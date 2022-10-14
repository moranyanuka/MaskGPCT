from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from .dvae import DiscreteVAE
from models.matcher import build_matcher

from .build import MODELS
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from utils.misc import accuracy
import random
import math
import numpy as np
import torch.optim as optim
from .transformer import Transformer
from utils.dist_utils import get_dist_info

# TODO: change all the places where device="cuda" to a general device type so it will support CPU as well

_CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([float('inf')]).to("cuda")

@MODELS.register_module()
class MaskGPCT(nn.Module):
    """ The Class that is used for training the transoformer.
        An added functionality of point-cloud generation and competion is added for testing during training.
        This funcitionality is provided in the log_cloud function.
    """
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskGPCT] build dVAE ...', logger ='MaskGPCT')
        self.config = config

        self.normalize_before = config.transformer_config.normalize_before
        self.aux_loss = config.transformer_config.aux_loss
        self.return_all_tokens = config.transformer_config.return_all_tokens
        if self.return_all_tokens:
            print_log(f'[MaskGPCT] Masked-GPCT calc the loss for all token ...', logger ='MaskGPCT')
        else:
            print_log(f'[MaskGPCT] Masked-GPCT [NOT] calc the loss for all token ...', logger ='MaskGPCT')
        # Queries should match the number of centers
        self.num_queries = config.dvae_config.num_group
        self.trans_dim = config.transformer_config.trans_dim
        self.group_size = config.dvae_config.group_size
        self.gamma = self.gamma_func("cosine")
        self.choice_temperature = config.transformer_config.choice_temperature
        self.mask_token_id = config.dvae_config.num_tokens + 1
        self.mask_center_id = config.dvae_config.num_tokens + 1.
        hidden_dim = self.trans_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.query_embed = nn.Embedding(self.num_queries, self.trans_dim)
        self.center_head = MLP(self.trans_dim, hidden_dim, 3, 3)
        self.token_head = nn.Linear(self.trans_dim, config.dvae_config.num_tokens)

        self.dvae = DiscreteVAE(config.dvae_config)
        # TODO: might want to simply pass config as an argument instead
        self.transformer = Transformer(self.trans_dim, config.transformer_config.num_heads, 
                                       config.transformer_config.encoder_depth, config.transformer_config.decoder_depth,
                                       config.transformer_config.mlp_ratio, dropout=config.transformer_config.dropout,
                                       normalize_before=self.normalize_before)
        
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, config.transformer_config.trans_dim)
        )  
        # TODO: might want to avoid loading the weights after training is done, 
        #       (since the weights are already being loaded form the ckpts) 
        self._prepare_dvae()
        # Freeze dVAE weights
        for param in self.dvae.parameters():
            param.requires_grad = False
        
        self.matcher = build_matcher(config.transformer_config)
        


        print_log(f'[MaskGPCT Group] divide point cloud into G{config.dvae_config.num_group} x S{config.dvae_config.group_size} points ...', logger ='MaskGPCT')
        # TODO: try different initializations
        #nn.init.uniform_(self.query_embed.weight, -1.0, 1.0)
        # initialize the learnable tokens
        #trunc_normal_(self.mask_token, std=.02)
        #trunc_normal_(self.query_embed.weight, std=.05)
        #self.apply(self._init_weights)
        # loss
        # TODO: adapt to the new task and add init for the weights if needed
        #self.build_loss_func()

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

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger ='Masked-GPCT')

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
        # TODO: avoid the for loop
        for token_id in token_ids:
            # number of tokens to mask
            gamma_r = math.floor(self.gamma(np.random.uniform()) * token_id.shape[0])
            sample = torch.rand(token_id.shape, device=token_ids.device).topk(gamma_r, dim=0).indices
            id_mask = torch.zeros(token_id.shape, dtype=torch.bool, device=token_ids.device)
            id_mask.scatter_(dim=0, index=sample, value=True)
            mask_idx.append(id_mask)
        bool_masked_pos = torch.stack(mask_idx) 
        return bool_masked_pos

    
    def create_input_mask(self,  num_clouds=1, mask_ratio=None, token_ids=None, tokens=None, centers=None):
        # TODO: change description
        """ creates an input with all tokens masked in the case of unconditional generation 
            or some of the tokens masked according to mask_ratio, in the case of completion.
            gets as input:
            - num_clouds: number of clouds to be generated.
            returns:
            - bool_masks: a boolean tensor  of size: (num_clouds, num_queries) with True in all masked locations.
            - masked_centers: a tensor of size (num_clouds, num_queries, 3) with all elements initialized as mask_center_id.
            - mask_tokens: a tensor of size: (num_clouds, num_queries, trans_dim) which contains only the mask_token at every location.
            - masked_token_ids: a tensor of size: (num_clouds, num_queries) with the mask_id at all locations.
        """
        # unconditional generation
        if tokens == None and token_ids == None and centers == None:
            assert mask_ratio is None
            blank_tokens = torch.ones((num_clouds, self.num_queries, self.trans_dim)).to(self.mask_token.device)
            blank_token_ids = torch.ones((num_clouds, self.num_queries), dtype=torch.int64).to(self.mask_token.device)
            token_ids = self.mask_token_id * blank_token_ids
            bool_masks = torch.ones((num_clouds, self.num_queries)).bool().to(self.mask_token.device)
            tokens = self.mask_token * blank_tokens
            centers = torch.full((num_clouds, self.num_queries, 3), self.mask_center_id, dtype=self.mask_token.dtype).to(self.mask_token.device)
        # conditional generation: shape completion case
        elif tokens != None and  token_ids != None and centers != None:
            # TODO: not really accurate, perhaps think of an accurate version
            bool_masks = (torch.rand(num_clouds, self.num_queries) < mask_ratio).bool().to(self.mask_token.device)
            #no_mask = torch.zeros(bool_masks.shape).type_as(bool_masks)
            ####################################
            # TODO: just for test, delete later
            bool_masks =  torch.ones((num_clouds, self.num_queries)).bool().to(self.mask_token.device)
            bool_masks[(centers[:, :, 1] < -0.3)] = False
            #bool_masks[:, 0] = False
            ####################################
            #centers, tokens, token_ids = self.match_known_tokens(bool_masks, centers, tokens, token_ids)
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
        
    def match_known_tokens(self, bool_mask, centers, tokens, token_ids):

        output_token_logits, output_centers = self.tokens_to_outputs(tokens, bool_mask, centers)

        bool_center = bool_mask.unsqueeze(-1).expand(centers.shape)
        unmasked_centers = torch.masked_select(centers, ~bool_center).view((centers.shape[0], -1, centers.shape[-1]))
        bool_tokens = bool_mask.unsqueeze(-1).expand(tokens.shape)
        unmasked_tokens = torch.masked_select(tokens, ~bool_tokens).view((tokens.shape[0], -1, tokens.shape[-1]))
        unmasked_token_ids = torch.masked_select(token_ids, ~bool_mask).view((token_ids.shape[0], -1))

        outputs = {'pred_logits': output_token_logits, 'pred_centers': output_centers}
        targets = [{"labels" : seq[0], "centers" : seq[1]} for seq in zip(unmasked_token_ids, unmasked_centers)]
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        idx_src = self._get_src_permutation_idx(indices)
        idx_tgt = self._get_tgt_permutation_idx(indices)
        unmasked_centers = unmasked_centers[idx_tgt]
        unmasked_token_ids = unmasked_token_ids[idx_tgt]
        unmasked_tokens = unmasked_tokens[idx_tgt]
        centers[:] = self.mask_center_id
        centers[idx_src] = unmasked_centers
        tokens[:] = self.mask_token
        tokens[idx_src] = unmasked_tokens
        token_ids[:] = self.mask_token_id 
        token_ids[idx_src] = unmasked_token_ids
        bool_mask = torch.ones(bool_mask.shape, device=bool_mask.device).bool()
        bool_mask[idx_src] = False
        #target_centers = centers[idx].reshape(centers.shape)
        #target_tokens = tokens[idx].reshape(tokens.shape)
        #target_token_ids = token_ids[idx].reshape(token_ids.shape)
        return bool_mask, centers, tokens, token_ids


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


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
        
        center_mask = bool_masking.unsqueeze(-1).expand(center.shape)
        token_mask = bool_masking.unsqueeze(-1).expand(input_tokens.shape)
        # add positional embeddings only to non masked tokens
        if num_tokens_unmasked != 0:
            unmasked_center = torch.masked_select(center, ~center_mask).view((-1, center.shape[-1]))
            pos = self.pos_embed(unmasked_center)
            pos_embed = torch.zeros_like(input_tokens)
            w = token_mask.type_as(input_tokens)
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
        """ Generates a bool mask of size (bs, num_tokens) according to probs at a length of mask_len with an added noise for diversity.
        """
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(probs.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.gather(sorted_confidence, -1, mask_len.to(torch.long))
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking
        
    def sample_with_scheduler(self, mask_ratio, T, num_clouds, token_ids, tokens, center):
        '''
            Sample the input_tokens iteratively according to a mask scheduling function
        '''

        if tokens == None:
            # unconditional generation
            assert center is None
            masked_centers, bool_mask, masked_tokens, masked_token_ids = self.create_input_mask(num_clouds)
            bool_mask_out = bool_mask.detach().clone()
        else:
            # conditional generation - cloud completion
            masked_centers, bool_mask, masked_tokens, masked_token_ids = self.create_input_mask(num_clouds=tokens.shape[0], mask_ratio=mask_ratio,
                                                                                                  token_ids=token_ids, tokens=tokens, centers=center)
            bool_mask_out = bool_mask.detach().clone()
            bool_mask, masked_centers, masked_tokens, masked_token_ids = self.match_known_tokens(bool_mask, masked_centers, masked_tokens, masked_token_ids)
        batch_size, seq_len, _ = masked_tokens.size()
        # adjust the mask tensor dimensions to match the input size
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        gamma = self.gamma_func("cosine")
        unknown_number_in_the_beginning = torch.sum(bool_mask == True, dim=-1) # B 
        cur_token_ids = masked_token_ids
        cur_tokens = masked_tokens
        cur_centers = masked_centers
        
        for t in range(T):
            output_token_logits, output_centers = self.tokens_to_outputs(cur_tokens, bool_mask, cur_centers)
            sampled_ids = torch.distributions.categorical.Categorical(logits=output_token_logits).sample().long()
            #sampled_max_ids = torch.argmax(output_token_logits, -1) # B G
            sampled_max_tokens = self.dvae.codebook[sampled_ids] # B G C
            # which tokens need to be sampled
            unknown_map_ids = (cur_token_ids == self.mask_token_id)  
            unknown_map_tokens = unknown_map_ids.unsqueeze(-1).expand(batch_size, seq_len, self.trans_dim)
            unknown_map_centers = unknown_map_ids.unsqueeze(-1).expand(masked_centers.shape)
            # replace all masked tokens with their samples and leave the others untouched 
            mixed_sampled_ids = torch.where(unknown_map_ids, sampled_ids, cur_token_ids)  # B G
            mixed_sampled_tokens = torch.where(unknown_map_tokens, sampled_max_tokens, cur_tokens)  # B G
            mixed_sampled_centers = torch.where(unknown_map_centers, output_centers, cur_centers)

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)
            # convert logits into probs
            probs = F.softmax(output_token_logits, dim=-1)  # B G N
            # get probability for selected tokens in categorical call, also for already sampled ones 
            selected_probs = torch.squeeze(torch.gather(probs, -1, torch.unsqueeze(mixed_sampled_ids, -1)), -1)   # B G
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
            cur_centers = torch.where(center_masking, masked_centers, mixed_sampled_centers)
            cur_tokens = torch.where(token_masking , mask_token, mixed_sampled_tokens)
        return bool_mask_out, bool_mask, cur_tokens, cur_token_ids, cur_centers

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
            # divide the point cloud to centers and their surrounding neighborhoods
            neighborhood, input_centers = self.dvae.group_divider(pc)
            unnorm_neighborhood = input_centers.unsqueeze(2) + neighborhood
            # produce the ground truth point tokens
            gt_logits, input_token_ids, input_tokens = self.patches_to_tokens(input_centers, neighborhood)

        bool_mask_out, bool_mask, pred_tokens, _, centers = self.sample_with_scheduler(num_clouds=num_clouds, mask_ratio=mask_ratio, T=T,
                                                                        token_ids=input_token_ids, tokens=input_tokens, center=input_centers)
        feature = self.dvae.dgcnn_2(pred_tokens, centers)
        coarse, fine = self.dvae.decoder(feature)

        whole_fine = (fine + centers.unsqueeze(2)).reshape(centers.size(0), -1, 3)
        whole_coarse = (coarse + centers.unsqueeze(2)).reshape(centers.size(0), -1, 3)

        # get the partial cloud
        #############
        #partial_centers = torch.masked_select(input_centers, ~bool_mask.unsqueeze(-1).expand(input_centers.shape)).view((-1, centers.shape[-1]))
        #partial_neighborhood = torch.masked_select(neighborhood, ~bool_mask.unsqueeze(-1).unsqueeze(-1).expand(neighborhood.shape)).view((-1, *neighborhood.shape[2:]))
        #partial_cloud = (partial_centers.unsqueeze(1) + partial_neighborhood).view(input_centers.shape[0], -1, 3)

        #partial_cloud = torch.masked_select(unnorm_neighborhood, ~bool_mask.unsqueeze(-1).unsqueeze(-1).expand(unnorm_neighborhood.shape)).view(unnorm_neighborhood.shape[0] ,-1, 3)
        # list of the partial clouds
        partial_clouds = [torch.masked_select(unnorm_neighborhood[i], ~bool_mask_out[i].unsqueeze(-1).unsqueeze(-1).expand(unnorm_neighborhood.shape[1:])).view(-1, 3) 
                          for i in range(bool_mask_out.shape[0])]
        #############

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, input_centers, bool_mask_out, centers, partial_clouds)
        return ret

    def forward(self, pts):
        """ The forward expects a Tensor, which consists of:
               - input_points.tensor: batched point clouds, of shape [batch_size  N  3]
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (vocabulary_size + 1)]
               - "pred_center": The normalized coordinates coordinates for all queries, represented as
                               (x, y, z). These values are normalized in [-1, 1]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # divide the point cloud to centers and their surrounding neighborhoods
        neighborhood, center = self.dvae.group_divider(pts)
        # produce the ground truth point tokens
        gt_logits, input_token_ids, input_tokens = self.patches_to_tokens(center, neighborhood) 

        batch_size, seq_len, trans_dim = input_tokens.size()

        bool_token_id_masking = self.get_mask_token_idx(input_token_ids)
        bool_token_masking = bool_token_id_masking.unsqueeze(-1).expand(batch_size, seq_len, trans_dim)
        bool_center_masking = bool_token_id_masking.unsqueeze(-1).expand(batch_size, seq_len, center.shape[-1])
        w = bool_token_masking.type_as(self.mask_token)
        # replace the masked locations with the mask token
        mask_tokens = input_tokens * (1 - w) + self.mask_token * w
        # get only the centers which aren't masked
        unmasked_center = torch.masked_select(center, ~bool_center_masking).view((-1, center.shape[-1]))
        pos = self.pos_embed(unmasked_center)
        pos_embed = torch.zeros_like(mask_tokens)
        pos_embed[(w == 0)] = pos.reshape(-1)        # w == 0 whenever the corresponding token is not masked
        hs = self.transformer(mask_tokens, self.query_embed.weight, pos_embed)[0]
        # TODO: might want to check different activation functions
        #outputs_center = F.tanh(self.center_head(hs))
        outputs_center = self.center_head(hs)
        outputs_token = self.token_head(hs)
        out = {'pred_logits': outputs_token[-1], 'pred_centers': outputs_center[-1]}
        target = [{"labels" : seq[0], "centers" : seq[1]} for seq in zip(input_token_ids, center)]
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_token, outputs_center)
        return out, target
 
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion(nn.Module):
    """ This class computes the loss for Mask-GPCT.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    # TODO: probably no need for eos_coef, delete
    def __init__(self, vocabulary_size, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            vocabulary_size: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.vocabulary_size)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_center, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_centers]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_tokens_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # legacy, fits the case where there is a no_class token
        target_tokens = torch.full(src_logits.shape[:2], self.vocabulary_size,
                                    dtype=torch.int64, device=src_logits.device)
        target_tokens[idx] = target_tokens_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_tokens, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['token_error'] = 100 - accuracy(src_logits[idx], target_tokens_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_center):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_centers(self, outputs, targets, indices, num_centers):
        """Compute the losses related to the centers: the L1 regression loss.
           targets dicts must contain the key "centers" containing a tensor of dim [nb_target_centers, 3]
           The target boxes are expected in format (center_x, center_y, center_z), normalized by the image size.
        """
        assert 'pred_centers' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_centers = outputs['pred_centers'][idx]
        target_centers = torch.cat([t['centers'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_center = F.l1_loss(src_centers, target_centers, reduction='none')
        #loss_center = self.center_loss(src_centers, target_centers)
        losses = {}
        losses['loss_center'] = loss_center.sum() / num_centers

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_centers, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'centers': self.loss_centers
#            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_centers, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target centers accross all nodes, for normalization purposes
        num_centers = sum(len(t["labels"]) for t in targets)
        num_centers = torch.as_tensor([num_centers], dtype=torch.float, device=next(iter(outputs.values())).device)
        rank, world_size = get_dist_info()
        if rank != 0:
            torch.distributed.all_reduce(num_centers)
        num_centers = torch.clamp(num_centers / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_centers))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_centers, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x