from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from .dvae import Group
from .dvae import DiscreteVAE, Encoder, DGCNN

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
import numpy as np
import torch.optim as optim
import copy
from pointnet2_ops import pointnet2_utils
from .Point_BERT import *

_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([float('inf')]).to("cuda")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


#@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()
        
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100


    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_embeddings = self.encoder(neighborhood)  #  B G N
        group_input_embeddings = self.reduce_dim(group_input_embeddings)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_embeddings.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_embeddings.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_embeddings), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret


#@MODELS.register_module()
class PointTransformerGenerator(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config 
        try:
            self.mask_rand = config.mask_rand
        except:
            self.mask_rand = False
        self.mask_ratio = config.mask_ratio
        self.mask_token_id = config.dvae_config.num_tokens + 1
        self.choice_temperature = config.choice_temperture
        self.replace_pob = config.replace_pob
        self.trans_dim = config.trans_dim
        self.depth = config.depth 
        self.drop_path_rate = config.drop_path_rate 
        #self.cls_dim = config.cls_dim 
        self.num_heads = config.num_heads 

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.npoints = config.npoints
        # grouper
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  config.encoder_dims
        # self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        # self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        # add transformer and encoder
        self.dvae = DiscreteVAE(config.dvae_config)
        self._prepare_dvae()
        
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        # self.cls_head_finetune = nn.Sequential(
        #    nn.Linear(self.trans_dim * 2, 256),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(0.5),
        #    nn.Linear(256, self.cls_dim)
        #)

        self.build_loss_func()
        
    
    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
    
    def get_loss_acc(self, pred, gt, smoothing=True):
        # import pdb; pdb.set_trace()
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path)
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]


        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print_log('missing_keys', logger = 'Transformer')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger = 'Transformer'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger = 'Transformer')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger = 'Transformer'
            )

        print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger = 'Transformer')


    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Point_BERT')


    def _mask_center(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p = 2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G

        return bool_masked_pos


    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos


    def _mask_by_ratio(self, center, mask_ratio):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        bool_masked_pos = (torch.rand(center.shape[:2]) < mask_ratio).bool().to(center.device)

        return bool_masked_pos


    def forward(self, pts):
        # no need for training for now
        pass

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError


    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking


    def _tokens_to_logits(self, input, pos):
        x = self.blocks(input, pos)
        x = self.norm(x)
        logits = self.lm_head(x)
        return x, logits

    # TODO: not used, delete when finished
    def get_input_token(self, center, input_embeddings, discrete_label_index, lr=500):
        epochs = 10
        #discrete_label = nn.functional.one_hot(discrete_label_index, self.num_tokens)[:, 0, :]
        input_embeddings = input_embeddings.detach().clone()
        input_embeddings.requires_grad = True
        #lm_head = nn.Linear(self.trans_dim, self.num_tokens)
        #lm_head.load_state_dict(self.lm_head)
        #lm_head = copy.deepcopy(self.lm_head)
        #lm_head.requires_grad = False
        criterion = nn.CrossEntropyLoss(ignore_index=self.mask_token_id)
        optimizer = optim.SGD([input_embeddings], lr=lr)
        running_loss = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            #pred = lm_head(input_token)
            #pred_prob = F.softmax(pred, dim=-1)

            pred = self.dvae.dgcnn_1(input_embeddings, center)
            loss = criterion(pred.reshape(-1, pred.size(-1)), discrete_label_index.reshape(-1,))
            #loss = criterion(pred_prob, discrete_label_index[:, 0])
            running_loss += loss.item() * input_embeddings.shape[0]

            # Backward pass
            loss.mean().backward()
            optimizer.step()
        return self.reduce_dim(input_embeddings)


    def _sample_with_scheduler(self, center, neighborhood, mask_ratio, T=10):

        # get the groud truth labels from the dVAE
        gt_logits = self.dvae.encoder(neighborhood)  #  B G N
        gt_logits = self.dvae.dgcnn_1(gt_logits, center)
        gt_dvae_label = gt_logits.argmax(-1).long() # B G 
        
        # TODO: check if need to change temperture
        #soft_one_hot_gt = F.gumbel_softmax(gt_logits, tau = 1, dim = 2, hard = True) # B G N
        #sampled_gt = torch.einsum('b g n, n c -> b g c', soft_one_hot_gt, self.dvae.codebook) # B G C
        
      
       # add noise to centers
       # center = center + torch.normal(mean=0, std=0.05, size=center.shape).cuda()

       # generate initial mask
        bool_masked_pos = self._mask_by_ratio(center, mask_ratio) # B G
        # encode the input cloud blocks
        input_embeddings = self.dvae.encoder(neighborhood)  #  B G N
        group_input_embeddings = self.reduce_dim(input_embeddings)

        # mask all tokens
        #bool_masked_pos = torch.zeros(bool_masked_pos.shape, dtype=torch.bool)

        #replaced_group_input_embeddings, overall_mask = self._random_replace(group_input_embeddings, bool_masked_pos.clone(), noaug = False)
        batch_size, seq_len, _ = group_input_embeddings.size()
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        # mask for testing
        #w[:, 0:50, :] = 1
        # replace the masked locations with the mask token
        masked_group_input_embeddings = group_input_embeddings * (1 - w) + mask_token * w
        #center = center + torch.normal(mean=0, std=0.1, size=center.shape).cuda()
        # add pos embedding
        pos = self.pos_embed(center)
        gamma = self.gamma_func("cosine")
        unknown_number_in_the_beginning = torch.sum(w.squeeze() == True, dim=-1) # B
        #cur_ids = masked_group_input_embeddings.argmax(-1)
        # TODO: need to clone and detach so gt_dvae_label will be preserved
        cur_ids = gt_dvae_label
        cur_ids[w.squeeze().bool()] = self.mask_token_id

        for t in range(T):
            pred_tokens, logits = self._tokens_to_logits(masked_group_input_embeddings, pos)
            
            #sampled_catagorical_ids = torch.distributions.categorical.Categorical(logits=logits).sample().long()
            sampled_max_ids = torch.argmax(logits, -1)
            # TODO: check the influence of sampling the max (expecting to have less diversity but more reconstruction accuracy)
            # TODO: check why gumbel_softmax yeilds different results from the max sampling
            #soft_one_hot = F.gumbel_softmax(logits, tau = 1, dim = 2, hard = True) # B G N
            #sampled_dvae = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.dvae.codebook) # B G C

            # which tokens need to be sampled -> bool [8, 257]
            unknown_map = (cur_ids == self.mask_token_id)  
            # replace all masked embeddings with their samples and leave the others untouched [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_max_ids, cur_ids)  
            sampled_dvae_codebook = self.dvae.codebook[sampled_ids]
            feature = self.dvae.dgcnn_2(sampled_dvae_codebook, center)
            coarse, fine = self.dvae.decoder(feature)

            whole_fine = (fine + center.unsqueeze(2)).reshape(batch_size, -1, 3)
            points = whole_fine

            if self.npoints == 1024:
                point_all = 1200

       
            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, self.npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

            neighborhood, center = self.dvae.group_divider(points)
            # encode the input cloud blocks
            input_embeddings = self.dvae.encoder(neighborhood)  #  B G N
            group_input_embeddings = self.reduce_dim(input_embeddings)        # encode the input cloud blocks

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)
            # convert logits into probs [8, 257, 1024]
            probs = F.softmax(logits, dim=-1)  
            # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  
            # ignore tokens which are already sampled [8, 257]
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS) 

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  
            # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            id_masking = self.mask_by_random_topk(mask_len, selected_probs, temperature= self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(id_masking, self.mask_token_id, sampled_ids)
            # print((cur_ids == 8192).count_nonzero())
            #masked_group_input_embeddings = torch.where(masked_group_input_embeddings == mask_token ,mask_token , masked_group_input_embeddings)
            token_masking = id_masking.unsqueeze(-1).expand(batch_size, seq_len, self.trans_dim)
            # find the point embeddings corresponding to the discrete point tokens sampled 
            #discrete_group_input_embeddings = self.get_input_token(center, input_embeddings, cur_ids)
            masked_group_input_embeddings = torch.where(token_masking , mask_token, group_input_embeddings)    
            # TODO: Check if it helps
            pos = self.pos_embed(center)
        
        sampled_dvae_codebook = self.dvae.codebook[cur_ids]
        return sampled_dvae_codebook, gt_logits


    #@torch.no_grad()
    def log_cloud(self, pts, mask_ratio=1):

        # add random noise to the point cloud
        # pts = pts + torch.normal(mean=0, std=0.01, size=pts.shape).cuda()

        num_itr_steps = 10
#        generate_with = "scheduler"
        generate_with = "scheduler"
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.dvae.group_divider(pts)

        if generate_with == "scheduler":
            sampled_tokens, gt_logits = self._sample_with_scheduler(center, neighborhood, mask_ratio=mask_ratio, T=num_itr_steps)
        else:
            raise NotImplementedError(f'{generate_with} sampling strategy type is not supported')

        feature = self.dvae.dgcnn_2(sampled_tokens, center)
        coarse, fine = self.dvae.decoder(feature)

        whole_fine = (fine + center.unsqueeze(2)).reshape(pts.size(0), -1, 3)
        whole_coarse = (coarse + center.unsqueeze(2)).reshape(pts.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, gt_logits, center)
        return ret


class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate 
#        self.cls_dim = config.transformer_config.cls_dim 
        self.replace_pob = config.transformer_config.replace_pob
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[Transformer args] {config.transformer_config}', logger = 'dVAE BERT')
#        self.encoder_dims =  config.dvae_config.encoder_dims
        # define the encoder
#        self.encoder = Encoder(encoder_channel = self.encoder_dims)
#        self.dgcnn1 = DGCNN(encoder_channel=self.encoder_dims, output_channel=self.trans_dim)
#        self._prepare_tokenizer()
#        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
#        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)
        try:
            self.mask_rand = config.transformer_config.mask_rand
        except:
            self.mask_rand = False
        
        # define the learnable tokens
#        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
#        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
#        self.codebook = nn.Parameter(torch.randn(config.dvae_config.num_tokens, config.dvae_config.tokens_dims))
        self._prepare_codebook(config.dvae_config.ckpt)

        # pos embedding for each patch 
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # head for token classification
        self.num_tokens = config.dvae_config.num_tokens
        self.lm_head = nn.Linear(self.trans_dim, self.num_tokens)
        # head for cls contrast
#        self.cls_head = nn.Sequential(
#            nn.Linear(self.trans_dim, self.cls_dim),
#            nn.GELU(),
#            nn.Linear(self.cls_dim, self.cls_dim)
#        )  
        # initialize the learnable tokens
#        trunc_normal_(self.cls_token, std=.02)
#        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def _prepare_codebook(self, dvae_ckpt):
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        codebook_ckpt = {k.replace("codebook.", ""): v for k, v in base_ckpt.items() if 'codebook' in k}
        
        #self.codebook.load_state_dict(codebook_ckpt, strict=True)
        self.codebook = nn.Parameter(copy.deepcopy(codebook_ckpt['codebook']))
        print_log(f'[Codebook] Successful Loading the ckpt for codebook from {dvae_ckpt}', logger = 'dVAE BERT')

#    def _prepare_tokenizer(self, dvae_ckpt):
#        ckpt = torch.load(dvae_ckpt, map_location='cpu')
#        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#        encoder_ckpt = {k.replace("encoder.", ""): v for k, v in base_ckpt.items() if 'encoder' in k}
#        dgcnn1_ckpt = {k.replace("dgcnn1.", ""): v for k, v in base_ckpt.items() if 'dgcnn1' in k}
#        
#        self.encoder.load_state_dict(encoder_ckpt, strict=True)
#        self.dgcnn1.load_state_dict(dgcnn1_ckpt, strict=True)
#        print_log(f'[Encoder] Successful Loading the ckpt for encoder from {dvae_ckpt}', logger = 'dVAE BERT')

#    def _prepare_dvae(self):
#        dvae_ckpt = self.config.dvae_config.ckpt
#        ckpt = torch.load(dvae_ckpt, map_location='cpu')
#        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#        self.dvae.load_state_dict(base_ckpt, strict=True)
#        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Point_BERT')


    def _mask_center(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p = 2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            mask_num = int(ratio * len(idx))

            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(center.shape[:2]).bool()
        
        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(center.shape[:2]) < ratio).bool().to(center.device)

        return bool_masked_pos


    def _mask_token_rand(self, input_tokens, noaug=False):
        '''
            input_tokens : B G N
            --------------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio[1] == 0:
            return torch.zeros(input_tokens.shape[:2]).bool()
        
        ratio = random.random() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]
        bool_masked_pos = (torch.rand(input_tokens.shape[:2]) < ratio).bool().to(input_tokens.device)

        return bool_masked_pos


    def _random_replace(self, group_input_embeddings, bool_masked_pos, noaug = False):
        '''
            group_input_embeddings : B G C
            bool_masked_pos : B G  
            -----------------
            replaced_grodvae.dvae.up_input_tokens: B G C
        '''
        # TODO: adapt to the new inputs (tokens, instead of centers) which cause the bool_masked_pos to be of different size
        # skip replace
        if noaug or self.replace_pob == 0:
            return group_input_embeddings, bool_masked_pos
        
        replace_mask = (torch.rand(group_input_embeddings.shape[:2]) < self.replace_pob).to(bool_masked_pos.device).bool()
        replace_mask = (replace_mask & ~bool_masked_pos)  # do not replace the mask pos

        overall_mask = (replace_mask + bool_masked_pos).bool().to(bool_masked_pos.device)  #  True for flake input

        detached_group_input_embeddings = group_input_embeddings.detach()
        flatten_group_input_embeddings = detached_group_input_embeddings.reshape(detached_group_input_embeddings.size(0) * detached_group_input_embeddings.size(1), detached_group_input_embeddings.size(2))
        idx = torch.randperm(flatten_group_input_embeddings.shape[0])
        shuffled_group_input_embeddings = flatten_group_input_embeddings[idx].reshape(detached_group_input_embeddings.size(0), detached_group_input_embeddings.size(1), detached_group_input_embeddings.size(2))

        replace_mask = replace_mask.unsqueeze(-1).type_as(detached_group_input_embeddings)
        replaced_group_input_embeddings = group_input_embeddings * (1 - replace_mask) + shuffled_group_input_embeddings * replace_mask
        return replaced_group_input_embeddings, overall_mask

    def forward(self, center, input_tokens, return_all_tokens = False, noaug = False):
#        input_tokens = self.codebook[input_tokens_ids]
        # generate mask
        # TODO: when using at inference, the input is already masked... handle this case
        if self.mask_rand:
            bool_masked_pos = self._mask_center_rand(input_tokens, noaug = noaug) # B G
#            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
#            bool_masked_pos = self._mask_center(center, noaug = noaug) # B G
            bool_masked_pos = self._mask_center(input_tokens, noaug = noaug) # B G
        # encoder the input cloud blocks
        #group_input_embeddings = self.encoder(neighborhood)  #  B G N
        #group_input_embeddings = self.reduce_dim(group_input_embeddings)
        # TODO: Handle the case where there is no neighborhood at the input, only centers
#        with torch.no_grad():
#            gt_logits = self.dvae.encoder(neighborhood)  #  B G N
#            gt_logits = self.dvae.dgcnn_1(gt_logits, center)
#            dvae_tokens = gt_logits.argmax(-1).long() # B G 

        # replace the point
        replaced_group_input_tokens, overall_mask = self._random_replace(input_tokens, bool_masked_pos.clone(), noaug = noaug)
        batch_size, seq_len, _ = replaced_group_input_tokens.size()
        # prepare cls and mask
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        masked_group_input_tokens = replaced_group_input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = masked_group_input_tokens
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        logits = self.lm_head(x)
        if return_all_tokens:
            return logits
        else:
            # return the predicted logits
            return logits[~overall_mask], logits[overall_mask], overall_mask # reduce the Batch dim        


@MODELS.register_module()
class MaskGPCT_compressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Masked-GPCT] build dVAE_BERT ...', logger ='Masked-GPCT')
        self.config = config

        self.moco_loss = config.transformer_config.moco_loss
        self.dvae_loss = config.transformer_config.dvae_loss
        self.cutmix_loss = config.transformer_config.cutmix_loss
        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[Masked-GPCT] Masked-GPCT calc the loss for all token ...', logger ='Masked-GPCT')
        else:
            print_log(f'[Masked-GPCT] Masked-GPCT [NOT] calc the loss for all token ...', logger ='Masked-GPCT')
        
        self.transformer_q = MaskTransformer(config)
        self.mask_token_id = config.dvae_config.num_tokens + 1
 #       self.transformer_q._prepare_encoder(self.config.dvae_config.ckpt)
        
        self.dvae = DiscreteVAE(config.dvae_config)
        self._prepare_dvae()

        for param in self.dvae.parameters():
            param.requires_grad = False

        self.group_size = config.dvae_config.group_size
        self.num_group = config.dvae_config.num_group
        # TODO: check how it affects pc quality
        self.choice_temperature = 2

        print_log(f'[Masked-GPCT Group] cutmix_BERT divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Masked-GPCT')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # loss
        self.build_loss_func()

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Masked-GPCT')


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

    #TODO: currently not used, might need to later delete
    def forward_eval(self, pts):
        # TODO: pass the no_grad as a decorator
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            #neighborhood, center = self.group_divider(pts)
            gt_logits = self.dvae.encoder(neighborhood)  #  B G N
            gt_logits = self.dvae.dgcnn_1(gt_logits, center)
            gt_dvae_label = gt_logits.argmax(-1).long() # B G 
            ##############################
            input_tokens = self.dvae.codebook[gt_dvae_label]
            bool_masked_pos = self._mask_by_ratio(center, mask_ratio=random.uniform(0, 1)) # B G

            batch_size, seq_len, _ = input_tokens.size()
            mask_token = self.transformer_q.mask_token.expand(batch_size, seq_len, -1)
            # mask the input tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)

            masked_group_input_tokens = input_tokens * (1 - w) + mask_token * w
            pos = self.transformer_q.pos_embed(center)
            pred_tokens, logits = self._token_to_logits(masked_group_input_tokens.long(), pos)
            
            #sampled_catagorical_ids = torch.distributions.categorical.Categorical(logits=logits).sample().long()
            sampled_max_ids = torch.argmax(logits, -1)
            ##############################
            #return torch.sum(sampled_max_ids == gt_dvae_label, dim=-1)
            return w, sampled_max_ids, gt_dvae_label
            
            
    def _mixup_pc(self, neighborhood, center, dvae_label):
        '''
            neighborhood : B G M 3
            center: B G 3
            dvae_label: B G
            ----------------------
            mixup_ratio: /alpha:
                mixup_label = alpha * origin + (1 - alpha) * flip

        '''
        mixup_ratio = torch.rand(neighborhood.size(0))
        mixup_mask = torch.rand(neighborhood.shape[:2]) < mixup_ratio.unsqueeze(-1)
        mixup_mask = mixup_mask.type_as(neighborhood)
        mixup_neighborhood = neighborhood * mixup_mask.unsqueeze(-1).unsqueeze(-1) + neighborhood.flip(0) * (1 - mixup_mask.unsqueeze(-1).unsqueeze(-1))
        mixup_center = center * mixup_mask.unsqueeze(-1) + center.flip(0) * (1 - mixup_mask.unsqueeze(-1))
        mixup_dvae_label = dvae_label * mixup_mask + dvae_label.flip(0) * (1 - mixup_mask)

        return mixup_ratio.to(neighborhood.device), mixup_neighborhood, mixup_center, mixup_dvae_label.long()


    def forward(self, pts, noaug = False, **kwargs):
        if noaug:
            return self.forward_eval(pts)
        else:
            # divide the point cloud in the same form. This is important
            neighborhood, center = self.group_divider(pts)
            # produce the gt point tokens
            with torch.no_grad():
                gt_logits = self.dvae.encoder(neighborhood) 
                gt_logits = self.dvae.dgcnn_1(gt_logits, center) #  B G N
                input_token_ids= gt_logits.argmax(-1).long() # B G 
                input_tokens = self.dvae.codebook[input_token_ids]
            # forward the query model in mask style 1.
            if self.return_all_tokens:
                pred_logits = self.transformer_q(center, input_tokens, return_all_tokens = self.return_all_tokens) # logits :  N G C 
            else:
                real_logits, pred_logits, mask = self.transformer_q(center, input_tokens, return_all_tokens = self.return_all_tokens) # logits :  N' C where N' is the mask.sum() 
            if self.dvae_loss:
                if self.return_all_tokens:
                    dvae_loss = self.loss_ce(pred_logits.reshape(-1, pred_logits.size(-1)), gt_logits.reshape(-1,))# + \
#                             self.loss_ce(mixup_logits.reshape(-1, mixup_logits.size(-1)), mix_dvae_label.reshape(-1,))
                else:
#                    dvae_loss = self.loss_ce(pred_logits, gt_logits[mask]) #+ \
                    dvae_loss = self.loss_ce(pred_logits.reshape(-1, pred_logits.size(-1)), input_token_ids[mask])
#                        self.loss_ce(mixup_flake_logits, mix_dvae_label[mixup_mask])
            else:
                dvae_loss = torch.tensor(0.).to(pts.device)
            return dvae_loss


    def _mask_by_ratio(self, input_tokens, mask_ratio):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        bool_masked_pos = (torch.rand(input_tokens.shape[:2]) < mask_ratio).bool().to(input_tokens.device)

        return bool_masked_pos


    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError


    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        # TODO: Check if noise is really needed
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        #confidence = torch.log(probs)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking


    def _token_to_logits(self, input_tokens, pos_emb):
        x = self.transformer_q.blocks(input_tokens, pos_emb)
        x = self.transformer_q.norm(x)
        logits = self.transformer_q.lm_head(x)
        return logits


    def _sample_with_scheduler(self, center, input_token_ids, mask_ratio, T=10):
        '''
            Sample the input_tokens iteratively according to a mask scheduling function
        '''
        input_tokens = self.dvae.codebook[input_token_ids]
        # get the mask locations according to the mask ratio
        bool_masked_pos = self._mask_by_ratio(center, mask_ratio) # B G
        # encode the input cloud blocks

        # mask all tokens
        #bool_masked_pos = torch.zeros(bool_masked_pos.shape, dtype=torch.bool)

        #replaced_group_input_embeddings, overall_mask = self._random_replace(group_input_embeddings, bool_masked_pos.clone(), noaug = False)
        batch_size, seq_len, _ = input_tokens.size()
        # adjust the mask tensor to match the input size 
        mask_token = self.transformer_q.mask_token.expand(batch_size, seq_len, -1)
        # mask the input tokens
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        # mask for testing
        # replace the masked locations with the mask token
        cur_tokens = input_tokens * (1 - w) + mask_token * w
        # add pos embedding
        pos = self.transformer_q.pos_embed(center)
        gamma = self.gamma_func("cosine")
        unknown_number_in_the_beginning = torch.sum(w.squeeze() == True, dim=-1) # B
        #cur_token_ids = masked_group_input_embeddings.argmax(-1)
        cur_token_ids = input_token_ids
        cur_token_ids[w.squeeze().bool()] = self.mask_token_id

        for t in range(T):
            logits = self._token_to_logits(cur_tokens.long(), pos)
            #sampled_catagorical_ids = torch.distributions.categorical.Categorical(logits=logits).sample().long()
            #sampled_max_ids = torch.distributions.categorical.Categorical(logits=logits).sample().long()
            sampled_max_ids = torch.argmax(logits, -1) # B G
            sampled_max_tokens = self.dvae.codebook[sampled_max_ids] # B G C
            # TODO: check the influence of sampling the max (expecting to have less diversity but more reconstruction accuracy)
            # TODO: check why gumbel_softmax yeilds different results from the max sampling
            #soft_one_hot = F.gumbel_softmax(logits, tau = 1, dim = 2, hard = True) # B G N
            #sampled_max_tokens = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.dvae.codebook) # B G C

            # which tokens need to be sampled -> bool [8, 257]
            unknown_map_ids = (cur_token_ids == self.mask_token_id)  
            unknown_map_tokens = unknown_map_ids.unsqueeze(-1).expand(batch_size, seq_len, self.transformer_q.trans_dim)
            # replace all masked tokens with their samples and leave the others untouched 
            mixed_sampled_ids = torch.where(unknown_map_ids, sampled_max_ids, cur_token_ids)  # B G
            mixed_sampled_tokens = torch.where(unknown_map_tokens, sampled_max_tokens, cur_tokens)  # B G
            #discrete_group_sampled_tokens = self.dvae.codebook[sampled_ids] # B G C

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)
            # convert logits into probs
            probs = F.softmax(logits, dim=-1)  # B G N
            # get probability for selected tokens in categorical call, also for already sampled ones 
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(mixed_sampled_ids, -1), -1), -1)   # B G
            # ignore tokens which are already sampled 
            selected_probs = torch.where(unknown_map_ids, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS) 

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map_ids, dim=-1, keepdim=True)-1, mask_len))  
            # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            id_masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_token_ids = torch.where(id_masking, self.mask_token_id, mixed_sampled_ids)
            # print((cur_token_ids == 8192).count_nonzero())
            # Expand the id_masking to match the token size
            token_masking = id_masking.unsqueeze(-1).expand(batch_size, seq_len, self.transformer_q.trans_dim)
            # find the point embeddings corresponding to the discrete point tokens sampled 
            cur_tokens = torch.where(token_masking , mask_token, mixed_sampled_tokens)    
        
        sampled_dvae_codebook = self.dvae.codebook[cur_token_ids]
        return sampled_dvae_codebook, input_tokens


    @torch.no_grad()
    def log_cloud(self, points, mask_ratio=1, num_itr_steps=10):
        # TODO: handle the case where points == None
        # add random noise to the point cloud
        # pts = pts + torch.normal(mean=0, std=0.01, size=pts.shape).cuda()

        # divide the point cloud in the same form. This is important
        neighborhood, center = self.dvae.group_divider(points)
        gt_logits = self.dvae.encoder(neighborhood)  #  B G N
        gt_logits = self.dvae.dgcnn_1(gt_logits, center)
        input_token_ids = gt_logits.argmax(-1).long() # B G 
        sampled_tokens, gt_logit_ids = self._sample_with_scheduler(center, input_token_ids, mask_ratio=mask_ratio, T=num_itr_steps)
        feature = self.dvae.dgcnn_2(sampled_tokens, center)
        coarse, fine = self.dvae.decoder(feature)

        whole_fine = (fine + center.unsqueeze(2)).reshape(points.size(0), -1, 3)
        whole_coarse = (coarse + center.unsqueeze(2)).reshape(points.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, gt_logit_ids, center)
        return ret


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output