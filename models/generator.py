from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
from .dvae import DiscreteVAE

from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils.misc import accuracy
import random
import math
import numpy as np
import torch.optim as optim
import copy
#from pointnet2_ops import pointnet2_utils
from .MaskGPCT import *
from .transformer import Transformer
from utils.dist_utils import get_dist_info


_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([float('inf')]).to("cuda")

# TODO: change all the places where device="cuda" to a general device type so it will support CPU as well
#@MODELS.register_module()
class Generator(nn.Module):

    def __init__(self, args, config):
        super().__init__()
        # TODO: when loading the MaskGPCT, it also initialize the dVAE unnecessarily, shoue be changed
        self.MaskGPCT = MaskGPCT(config)
        self.load_model_from_ckpt(args.ckpts)
        print_log(f'[generator] build MaskGPCT...', logger ='generator')
        self.config = config

        self.normalize_before = config.transformer_config.normalize_before
        self.aux_loss = config.transformer_config.aux_loss
        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[generator] generator calc the loss for all token ...', logger ='generator')
        else:
            print_log(f'[generator] generator [NOT] calc the loss for all token ...', logger ='generator')
        # Queries should match the number of centers
        self.num_queries = config.dvae_config.num_group
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.dvae_config.group_size
        self.gamma = self.gamma_func("cosine")
        self.choice_temperature = config.transformer_config.choice_temperature
        self.mask_token_id = config.dvae_config.num_tokens + 1
        self.mask_center_id = config.dvae_config.num_tokens + 1.
#        hidden_dim = self.trans_dim
#        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
#        self.query_embed = nn.Embedding(self.num_queries, self.trans_dim)
#        self.center_head = MLP(self.trans_dim, hidden_dim, 3, 3)
#        self.token_head = nn.Linear(self.trans_dim, config.dvae_config.num_tokens)
#
#        # TODO: might want to simply pass config as an argument instead
#        self.transformer = Transformer(self.trans_dim, config.transformer_config.num_heads, 
#                                       config.transformer_config.encoder_depth, config.transformer_config.decoder_depth,
#                                       config.transformer_config.mlp_ratio, dropout=config.transformer_config.dropout,
#                                       normalize_before=self.normalize_before)
#        
#        self.dvae = DiscreteVAE(config.dvae_config)
#        self.pos_embed = nn.Sequential(
#            nn.Linear(3, 128),
#            nn.GELU(),
#            nn.Linear(128, config.transformer_config.trans_dim)
#        )  
#
#        print_log(f'[generator Group] divide point cloud into G{config.dvae_config.num_group} x S{config.dvae_config.group_size} points ...', logger ='generator')
#        # TODO: try different initializations
#        #nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)
#        # initialize the learnable tokens
#        #trunc_normal_(self.mask_token, std=.02)
#        #trunc_normal_(self.query_embed.weight, std=.05)
#        #self.apply(self._init_weights)
#        # loss
#        # TODO: adapt to the new task and add init for the weights if needed
#        #self.build_loss_func()

    def load_model_from_ckpt(self, MaskGPCT_ckpt_path):
        ckpt = torch.load(MaskGPCT_ckpt_path)
        base_ckpt = {k.replace("module", "MaskGPCT"): v for k, v in ckpt['base_model'].items()}
#       for k in list(base_ckpt.keys()):
#            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
#                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
#            if k.startswith('base_model'):
#                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
#            del base_ckpt[k]

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
        print_log(f'[Transformer] Successful Loading the ckpt from {MaskGPCT_ckpt_path}', logger = 'Transformer')

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

    def get_mask_token_idx(self, token_ids):
        """ Returns a random mask according to gamma mask scheduling function.
            - id_mask: A bool tensor of shape (bs, num_queries) with True in places to be masked.
        """
        mask_idx = []
        # TODO: need to avoid the for loop
        for token_id in token_ids:
            # number of token_ids to mask
            gamma_r = math.floor(self.gamma(np.random.uniform()) * token_id.shape[0])
            sample = torch.rand(token_id.shape, device=token_ids.device).topk(gamma_r, dim=0).indices
            id_mask = torch.zeros(token_id.shape, dtype=torch.bool, device=token_ids.device)
            id_mask.scatter_(dim=0, index=sample, value=True)
            mask_idx.append(id_mask)
        bool_masked_pos = torch.stack(mask_idx) 
        return bool_masked_pos

    def _mask_center(self, center):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0) # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p = 2 ,dim = -1)  # 1 1 3 - 1 G 3 -> 1 G
    
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            #ratio = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
            #mask_num = int(ratio * len(idx))
            mask_num = math.floor(self.gamma(np.random.uniform()) * len(idx))
    
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())
        
        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B G
        return bool_masked_pos 
    
    def create_input_mask(self,  num_clouds=1, mask_ratio=None, token_ids=None, tokens=None, centers=None):
        # TODO: change description
        """ creates an input with all tokens masked.
            gets as input:
            - num_clouds: number of clouds to be generated.
            returns:
            - bool_masks: a boolean tensor  of size: (num_clouds, num_queries) with True in all masked locations.
            - masked_centers: a tensor of size (num_clouds, num_queries, 3) with all elements initialized as mask_center_id.
            - mask_tokens: a tensor of size: (num_clouds, num_queries, trans_dim) which contains only the mask_token at every location.
            - masked_token_ids: a tensor of size: (num_clouds, num_queries) with the mask_id at all locations.
        """

        if tokens == None and token_ids == None and centers == None:
            assert mask_ratio is None
            blank_tokens = torch.ones((num_clouds, self.num_queries, self.trans_dim)).to(self.mask_token.device)
            blank_token_ids = torch.ones((num_clouds, self.num_queries), dtype=torch.int64).to(self.mask_token.device)
            token_ids = self.mask_token_id * blank_token_ids
            bool_masks = torch.ones((num_clouds, self.num_queries)).bool().to(self.mask_token.device)
            tokens = self.mask_token * blank_tokens
            centers = torch.full((num_clouds, self.num_queries, 3), self.mask_center_id, dtype=self.mask_token.dtype).to(self.mask_token.device)
        elif tokens != None and  token_ids != None and centers != None:
            # TODO: not really accurate, perhaps think of a more accurate version
            bool_masks = (torch.rand(num_clouds, self.num_queries) < mask_ratio).bool().to(self.mask_token.device)
        else:
            raise NotImplementedError
        
        w_token = bool_masks.unsqueeze(-1).expand(tokens.shape).type_as(self.mask_token)
        w_token_id = bool_masks.type_as(token_ids)
        w_center = bool_masks.unsqueeze(-1).expand(centers.shape).type_as(self.mask_token)

        # replace the masked locations with the mask token (unchanged in the case of unconditional generation)
        masked_tokens = tokens * (1 - w_token) + self.mask_token * w_token
        masked_token_ids = token_ids * (1 - w_token_id) + self.mask_token_id * w_token_id
        mask_center = centers  * (1 - w_center) + self.mask_center_id * w_center

        return mask_center, bool_masks, masked_tokens, masked_token_ids
        
    
    # TODO: Add description
    @torch.no_grad()
    def patches_to_tokens(self, center, neighborhood):
        # produce the ground truth point tokens
        gt_logits = self.dvae.encoder(neighborhood) 
        gt_logits = self.dvae.dgcnn_1(gt_logits, center) #  B G N
        token_ids = gt_logits.argmax(-1).long() # B G 
        tokens = self.dvae.codebook[token_ids]
        return gt_logits, token_ids, tokens

    def tokens_to_outputs(self, input_tokens, bool_masking, center):
        """ Computes the predicted token centers.
            Gets as input:
            - input_tokens
            - bool_center_masking: a tensor of size(num_clouds, num_center) which is used as an indicator of which centers got masked.
            - center: a tensor of size (num_clouds, num_centers) where in the locations of masked centers, it has the mask_token_id
            Returns:
            - output_tokens
            - output_centers
        """
        num_tokens_unmasked = torch.sum(bool_masking == False)
        
        bool_center_unmasked = bool_masking.unsqueeze(-1).expand(center.shape)
        bool_token_unmasked = bool_masking.unsqueeze(-1).expand(input_tokens.shape)
        # add positional embeddings only to non masked tokens
        if num_tokens_unmasked != 0:
            unmasked_center = torch.masked_select(center, ~bool_center_unmasked).view((-1, center.shape[-1]))
            pos = self.pos_embed(unmasked_center)
            pos_embed = torch.zeros_like(input_tokens)
            w = bool_token_unmasked.type_as(input_tokens)
            pos_embed[(w == 0)] = pos.reshape(-1)
        else:
            pos_embed = None
        hs = self.transformer(input_tokens, self.query_embed.weight, pos_embed)[0]
        # TODO: might want to check different activation functions
        #outputs_center = F.tanh(self.center_head(hs))
        outputs_center = self.center_head(hs)
        outputs_token_logits = self.token_head(hs)
        return outputs_token_logits[-1], outputs_center[-1]

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        """ Generates a bool mask of size (bs, num_tokens) according to probs at a length of mask_len  with an added noise for diversity.
        """
        # TODO: Check if noise is really needed
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    def sample_with_scheduler(self, mask_ratio, T, num_clouds, token_ids, tokens, center):
        '''
            Sample the input_tokens iteratively according to a mask scheduling function
        '''
        if tokens == None:
            assert center is None
            # for unconditional generation
            mask_centers, bool_mask, input_tokens_masking, input_token_ids = self.create_input_mask(num_clouds)
        else:
            # for cloud completion
            mask_centers, bool_mask, input_tokens_masking, input_token_ids = self.create_input_mask(num_clouds=tokens.shape[0], mask_ratio=mask_ratio,
                                                                                                    token_ids=token_ids, tokens=tokens, centers=center)
        # adjust the mask tensor dimensions to match the input size
        batch_size, seq_len, _ = input_tokens_masking.size()
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        gamma = self.gamma_func("cosine")
        #unknown_number_in_the_beginning = torch.sum(w.squeeze() == True, dim=-1) # B
        unknown_number_in_the_beginning = torch.sum(bool_mask == True, dim=-1) # B 
        cur_token_ids = input_token_ids
        cur_tokens = input_tokens_masking
        cur_centers = mask_centers
        
        for t in range(T):
            output_token_logits, output_centers = self.tokens_to_outputs(cur_tokens, bool_mask, cur_centers)
            sampled_ids = torch.distributions.categorical.Categorical(logits=output_token_logits).sample().long()
            #sampled_max_ids = torch.argmax(output_token_logits, -1) # B G
            sampled_max_tokens = self.dvae.codebook[sampled_ids] # B G C
            # TODO: check the influence of sampling the max (expecting to have less diversity but more reconstruction accuracy)
            # TODO: check why gumbel_softmax yeilds different results from the max sampling
            # which tokens need to be sampled
            unknown_map_ids = (cur_token_ids == self.mask_token_id)  
            unknown_map_tokens = unknown_map_ids.unsqueeze(-1).expand(batch_size, seq_len, self.trans_dim)
            unknown_map_centers = unknown_map_ids.unsqueeze(-1).expand(mask_centers.shape)
            # replace all masked tokens with their samples and leave the others untouched 
            mixed_sampled_ids = torch.where(unknown_map_ids, sampled_ids, cur_token_ids)  # B G
            mixed_sampled_tokens = torch.where(unknown_map_tokens, sampled_max_tokens, cur_tokens)  # B G
            mixed_sampled_centers = torch.where(unknown_map_centers, output_centers, cur_centers)

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)
            # convert logits into probs
            probs = F.softmax(output_token_logits, dim=-1)  # B G N
            # get probability for selected tokens in categorical call, also for already sampled ones 
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(mixed_sampled_ids, -1), -1), -1)   # B G
            # ignore tokens which are already sampled 
            selected_probs = torch.where(unknown_map_ids, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS) 

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map_ids, dim=-1, keepdim=True)-1, mask_len))  
            # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            bool_mask = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # print((cur_token_ids == 8192).count_nonzero())
            center_masking = bool_mask.unsqueeze(-1).expand(output_centers.shape)
            token_masking = bool_mask.unsqueeze(-1).expand(batch_size, seq_len, self.trans_dim)
            # Masks tokens with lower confidence.
            cur_token_ids = torch.where(bool_mask, self.mask_token_id, mixed_sampled_ids)
            cur_centers = torch.where(center_masking, mask_centers, mixed_sampled_centers)
            cur_tokens = torch.where(token_masking , mask_token, mixed_sampled_tokens)
        return cur_tokens, cur_token_ids, cur_centers

    @torch.no_grad()
    def log_cloud(self, T=8, num_clouds=1, pc=None, mask_ratio=0):
        """ Generates a point cloud. 
            to generate a cloud unconditionally, simply pass T, num_cloud.
            to generate a completion to a cloud, pass also the point cloud (pc) and mask_ratio wanted
                                             
            inputs:
            - num_clouds: the amout of point clouds to be generated. only used when input_tokens=None.
            - T: the number of iterations used for the generation proccess.

        """
        if pc == None:
            # in case of unconditional generation
            input_tokens = None
            input_centers = None
            input_token_ids = None
        else:
            # divide the point cloud to centers and it's surrounding neighborhoods
            neighborhood, input_centers = self.dvae.group_divider(pc)
            # produce the ground truth point tokens
            gt_logits, input_token_ids, input_tokens = self.patches_to_tokens(input_centers, neighborhood)

        pred_tokens, _, centers = self.sample_with_scheduler(num_clouds=num_clouds, mask_ratio=mask_ratio, 
                                                             T=T, token_ids=input_token_ids, tokens=input_tokens, center=input_centers)
        feature = self.dvae.dgcnn_2(pred_tokens, centers)
        coarse, fine = self.dvae.decoder(feature)

        whole_fine = (fine + centers.unsqueeze(2)).reshape(centers.size(0), -1, 3)
        whole_coarse = (coarse + centers.unsqueeze(2)).reshape(centers.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, input_centers, None, centers)
        return ret

    def forward(self):
        # generate a point-cloud uncoditionally
        self.log_cloud(self, T=8, num_clouds=1)