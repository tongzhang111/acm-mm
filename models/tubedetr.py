# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TubeDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn
import math
import copy

import util.dist as dist
from util import box_ops
from util.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
from einops import rearrange
from .matcher import build_matcher

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    pred_is_referred = inputs.squeeze(2).log_softmax(dim=-1)
    loss = -(pred_is_referred * targets.squeeze(2))   #vidstg
    # prob = inputs.sigmoid()
    # ce_loss = F.binary_cross_entropy_with_logits(prob, targets, reduction="none")  #hcstvg
    # p_t = prob * targets + (1 - prob) * (1 - targets)
    # loss = ce_loss * ((1 - p_t) ** gamma)
    # # #
    # if alpha >= 0:
    #     alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #     loss = alpha_t * loss

    return loss.sum(-1).mean() / num_boxes

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x


class TubeDETR(nn.Module):
    """This is the TubeDETR module that performs spatio-temporal video grounding"""

    def __init__(
        self,
        num_classes,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        video_max_len=200,
        stride=5,
        guided_attn=False,
        fast=False,
        fast_mode="",
        sted=True,
        num_feature_levels=3,
        init_ref_dim=2,
        box_refine=True,
    ):
        """
        :param backbone: visual backbone model
        :param transformer: transformer model
        :param num_queries: number of object queries per frame
        :param aux_loss: whether to use auxiliary losses at every decoder layer
        :param video_max_len: maximum number of frames in the model
        :param stride: temporal stride k
        :param guided_attn: whether to use guided attention loss
        :param fast: whether to use the fast branch
        :param fast_mode: which variant of fast branch to use
        :param sted: whether to predict start and end proba
        ssh -p 46037 root@region-41.seetacloud.com
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.hsbbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.class_embeding = nn.Linear(hidden_dim, num_classes)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bbox_query_embed = nn.Embedding(1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels[2], hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.video_max_len = video_max_len
        self.stride = stride
        self.guided_attn = guided_attn
        self.fast = fast
        self.fast_mode = fast_mode
        self.sted = sted
        if sted:
            self.sted_embed = MLP(hidden_dim, hidden_dim, 2, 2, dropout=0.5)

        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs-1):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            # the 3x3 conv with large input channels takes lots of params
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj_pre = nn.ModuleList(input_proj_list)
        else:
            self.input_proj_pre = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                )])
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        # nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        # nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # num_pred = transformer.decoder.num_layers
        # if box_refine:
        #     # self.class_embed = _get_clones(self.class_embed, num_pred)
        #     self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
        #     nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        # else:
        #     nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        #     # self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
        #     self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
        # # Note in our impl, decoder.bbox_embed is never None, for guiding the RoI extraction
        # self.transformer.decoder.bbox_embed = self.bbox_embed
        # self.ref_point_head = MLP(hidden_dim, hidden_dim, output_dim=init_ref_dim, num_layers=2)
        self.transformer.decoder.temp = self.sted_embed
        #self.box_refine = box_refine

    def forward(
        self,
        samples: NestedTensor,
        durations,
        captions,
        encode_and_save=True,
        memory_cache=None,
        samples_fast=None,
        meat_info=None,
        keep=None,
        bbox=None,
    ):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched frames, of shape [n_frames x 3 x H x W]
           - samples.mask: a binary mask of shape [n_frames x H x W], containing 1 on padded pixels
        It returns a dict with the following elements:
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            b = len(durations)
            t = max(durations)
            features, pos = self.backbone(
                samples
            )  # each frame from each video is forwarded through the backbone
            srcs = []
            masks = []
            pre_feature = features[:2]
            for l, feat in enumerate(pre_feature):
                src, mask = feat.decompose()
                srcs.append(self.input_proj_pre[l](src))
                masks.append(mask)
                assert mask is not None
            src, mask = features[
                -1
            ].decompose()  # src (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32)); mask (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))
            src = self.input_proj(src)
            srcs.append(src)
            masks.append(mask)
            if self.fast:
                with torch.no_grad():  # fast branch does not backpropagate to the visual backbone
                    features_fast, pos_fast = self.backbone(samples_fast)
                src_fast, mask_fast = features_fast[-1].decompose()
                src_fast = self.input_proj(src_fast)
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)

            ms_feats = [torch.cat([src_[keep], pos_[keep]], dim=1) for src_, pos_ in zip(srcs, pos)]  # (bs, 2*c, H, W)

            assert len(ms_feats) == self.num_feature_levels
            if self.num_feature_levels == 1:
                ms_feats = None
                src, mask, pos = srcs[0], masks[0], pos[0]
            elif self.num_feature_levels in [3, 4]:  # only the /32 scale is processed by the encoder
                src, mask, pos = srcs[2], masks[2], pos[2]
            else:
                raise NotImplementedError


            # temporal padding pre-encoder
            # src = self.input_proj(src)
            _, f, h, w = src.shape
            meat_info["src_size"] = src.shape
            f2 = pos.size(1)
            device = src.device
            tpad_mask_t = None
            fast_src = None
            if not self.stride:
                tpad_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask = torch.ones(b, t, h, w).bool().to(device)
                pos_embed = torch.zeros(b, t, f2, h, w).to(device)
                cur_dur = 0
                for i_dur, dur in enumerate(durations):
                    tpad_src[i_dur, :dur] = src[cur_dur : cur_dur + dur]
                    tpad_mask[i_dur, :dur] = mask[cur_dur : cur_dur + dur]
                    pos_embed[i_dur, :dur] = pos[cur_dur : cur_dur + dur]
                    cur_dur += dur
                tpad_src = tpad_src.view(b * t, f, h, w)
                tpad_mask = tpad_mask.view(b * t, h, w)
                tpad_mask[:, 0, 0] = False  # avoid empty masks
                pos_embed = pos_embed.view(b * t, f2, h, w)
            else:  # temporal sampling
                n_clips = math.ceil(t / self.stride)
                tpad_src = src
                tpad_mask = mask
                pos_embed = pos
                if self.fast:
                    fast_src = torch.zeros(b, t, f, h, w).to(device)
                tpad_mask_t = (
                    torch.ones(b, t, h, w).bool().to(device)
                )  # temporally padded mask for all frames, will be used for the decoding
                cum_dur = 0  # updated for every video
                cur_dur = 0
                cur_clip = 0
                for i_dur, dur in enumerate(durations):
                    if self.fast:
                        fast_src[i_dur, :dur] = src_fast[cum_dur : cum_dur + dur]
                        tpad_mask_t[i_dur, :dur] = mask_fast[cum_dur : cum_dur + dur]
                    else:
                        for i_clip in range(math.ceil(dur / self.stride)):
                            clip_dur = min(self.stride, dur - i_clip * self.stride)
                            tpad_mask_t[
                                i_dur, cur_dur - cum_dur : cur_dur - cum_dur + clip_dur
                            ] = mask[cur_clip : cur_clip + 1].repeat(clip_dur, 1, 1)
                            cur_dur += clip_dur
                            cur_clip += 1
                    cum_dur += dur
                tpad_src = tpad_src.view(b * n_clips, f, h, w)
                tpad_mask = tpad_mask.view(b * n_clips, h, w)
                pos_embed = pos_embed.view(b * n_clips, f, h, w)
                tpad_mask_t = tpad_mask_t.view(b * t, h, w)
                if self.fast:
                    fast_src = fast_src.view(b * t, f, h, w)
                tpad_mask[:, 0, 0] = False  # avoid empty masks
                tpad_mask_t[:, 0, 0] = False  # avoid empty masks
            query_embed = self.query_embed.weight
            bbox_query_embed = self.bbox_query_embed.weight
            # video-text encoder
            memory_cache = self.transformer(
                tpad_src,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                tpad_mask,  # (n_clips)x(math.ceil(H/32))x(math.ceil(W/32))
                query_embed,  # num_queriesxF
                pos_embed,  # (n_clips)xFx(math.ceil(H/32))x(math.ceil(W/32))
                captions,  # list of length batch_size
                encode_and_save=True,
                durations=durations,  # list of length batch_size
                tpad_mask_t=tpad_mask_t,  # (n_frames)x(math.ceil(H/32))x(math.ceil(W/32))
                fast_src=fast_src,  # (n_frames)xFx(math.ceil(H/32))x(math.ceil(W/32))
                meta_info=meat_info,
                ms_feats=ms_feats,
                keep=keep,
                bbox_query_embed=bbox_query_embed,
                bbox=bbox,
            )

            return memory_cache

        else:
            assert memory_cache is not None
            # space-time decoder
            hs = self.transformer(
                img_memory=memory_cache[
                    "img_memory"
                ],  # (math.ceil(H/32)*math.ceil(W/32) + n_tokens)x(BT)xF
                mask=memory_cache[
                    "mask"
                ],  # (BT)x(math.ceil(H/32)*math.ceil(W/32) + n_tokens)
                pos_embed=memory_cache["pos_embed"],  # n_tokensx(BT)xF
                query_embed=memory_cache["query_embed"],  # (num_queries)x(BT)xF
                query_mask=memory_cache["query_mask"],  # Bx(Txnum_queries)
                encode_and_save=False,
                text_memory=memory_cache["text_memory"],
                text_mask=memory_cache["text_attention_mask"],
                meta_info=memory_cache["meta_info"],
                ms_feats=memory_cache["ms_feats"],
                keep=memory_cache["keep"],
                bbox_query_embed=memory_cache["bbox_query_embed"],
                bbox=memory_cache["bbox"],

            )
            if self.guided_attn:
                hs, weights, cross_weights, bbox_hs, outputs_sted = hs
            out = {}
            # outputs_coord=outputs_coord.transpose(1,2)

            # outputs heads
            # if self.sted:
            #     outputs_sted = self.sted_embed(hs)

            # hs = hs.flatten(1, 2)  # n_layersxbxtxf -> n_layersx(b*t)xf
            pred_logits=[]
            for lvl in range(hs.shape[0]):
                outputs_class = self.class_embeding(hs[lvl])
                pred_logits.append(outputs_class)

            out.update({"hs_feature": hs[-1]})
            out.update({"boxhs_feature": bbox_hs[-1]})
            out.update({"pred_logits": pred_logits[-1]})
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update({"pred_boxes": outputs_coord[-1]})
            bbox_outputs_coord = self.bbox_embed(bbox_hs).sigmoid()
            out.update({"bbox_outputs_coord": bbox_outputs_coord[-1]})
            if self.sted:
                out.update({"pred_sted": outputs_sted[-1]})
            if self.guided_attn:
                out["weights"] = weights[-1]
                out["ca_weights"] = cross_weights[-1]

            # auxiliary outputs
            if self.aux_loss:
                out["aux_outputs"] = [
                    {
                        "pred_boxes": b,
                    }
                    for b in outputs_coord[:-1]
                ]
                for i_aux in range(len(out["aux_outputs"])):
                    out["aux_outputs"][i_aux]["pred_logits"] = pred_logits[i_aux]
                    if self.sted:
                        out["aux_outputs"][i_aux]["pred_sted"] = outputs_sted[i_aux]
                    if self.guided_attn:
                        out["aux_outputs"][i_aux]["weights"] = weights[i_aux]
                        out["aux_outputs"][i_aux]["ca_weights"] = cross_weights[i_aux]

            return out


class SetCriterion(nn.Module):
    """This class computes the loss for TubeDETR."""

    def __init__(self, num_classes, matcher, losses, axu_losses, sigma=1, focal_alpha=0.25):
        """Create the criterion.
        Parameters:
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            sigma: standard deviation for the Gaussian targets in the start and end Kullback Leibler divergence loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses
        self.axu_losses = axu_losses
        self.sigma = sigma
        self.focal_alpha = focal_alpha
        self.criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}



    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def loss_labels(self, outputs, num_boxes, inter_idx, positive_map, time_mask=None, indices=None, keep=None, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].transpose(0,1).unsqueeze(0)
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')

        # judge the valid frames
        valid_indices = []
        # valids = [target['valid'] for target in targets]
        valid = torch.ones(nf).cuda()
        valids = [valid]
        for valid, (indice_i, indice_j) in zip(valids, indices):
            valid_ind = valid.nonzero().flatten()
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices) # NOTE: use valid indices
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])
        # target_classes_o = torch.cat(0 for i in range(len(valid)))
        target_classes_o = torch.zeros(nf)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if self.num_classes == 1: # binary referred
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
        losses = {'loss_label': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            pass
        return losses

    def loss_cross_sim(self, outputs, num_boxes, inter_idx, positive_map, time_mask=None, indices=None, keep=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "hs_feature" in outputs
        assert "boxhs_feature" in outputs
        # cos = nn.CosineSimilarity(dim=2,eps=1e-6)
        idx = self._get_src_permutation_idx(indices)
        hs_feature = outputs["hs_feature"].transpose(0,1)
        nf,nq,_ = hs_feature.shape
        bboxhs_feature = outputs["boxhs_feature"].transpose(0,1)
        targets = torch.zeros([1,nq]).cuda()
        targets[idx]=1
        targets = targets.repeat(nf,1)
        # cos_sim = cos(hs_feature, bboxhs_feature)
        bboxhs_feature = F.normalize(bboxhs_feature,dim=-1)
        hs_feature = F.normalize(hs_feature,dim=-1)
        cross_sim = hs_feature.bmm(bboxhs_feature.transpose(1,2))
        cross_sim = cross_sim.squeeze(2)
        cross_sim = nn.LogSoftmax(dim=1)(cross_sim / 0.1)
        losses_ce = self.criterions['KLDiv'](cross_sim, targets)
        losses = {'loss_cross_sim': losses_ce}

        # ps_loss = cross_sim[:,idx[1]]
        # np_loss = cross_sim.masked_fill(targets.squeeze(2).bool()[None, :], 0)
        # np_loss = torch.max(np_loss)
        # cr = torch.tensor(0.1).cuda() + np_loss - ps_loss
        # loss_ce = torch.max(torch.tensor(0).cuda(),cr)
        # loss_ce = loss_ce.mean()
        # src_boxes = outputs["pred_boxes"].transpose(0,1).unsqueeze(0)
        # bs, nf, nq = src_boxes.shape[:3]
        # src_boxes = src_boxes.transpose(1, 2)
        # idx = self._get_src_permutation_idx(indices)
        # src_boxes = src_boxes[idx]
        # src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]
        # target_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
        # loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        #
        # losses = {}
        # losses["loss_bbox"] = loss_bbox.sum() / max(num_boxes, 1)
        #
        # loss_giou = 1 - torch.diag(
        #     box_ops.generalized_box_iou(
        #         box_ops.box_cxcywh_to_xyxy(src_boxes),
        #         box_ops.box_cxcywh_to_xyxy(target_boxes),
        #     )
        # )
        # losses["loss_giou"] = loss_giou.sum() / max(num_boxes, 1)
        return losses

    def loss_boxes(self, outputs, targets, num_boxes, indices, keep=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        src_boxes = outputs["pred_boxes"].transpose(0,1).unsqueeze(0)
        bs, nf, nq = src_boxes.shape[:3]
        src_boxes = src_boxes.transpose(1, 2)
        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]
        target_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / max(num_boxes, 1)

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / max(num_boxes, 1)
        return losses



    def loss_bboxhs(self, outputs, targets, num_boxes, indices, keep=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        src_boxes = outputs["bbox_outputs_coord"].transpose(0,1)
        bs, nf, nq = src_boxes.shape[:3]
        # src_boxes = src_boxes.transpose(1, 2)
        # idx = self._get_src_permutation_idx(indices)
        # src_boxes = src_boxes[idx]
        src_boxes = src_boxes.flatten(0, 1)  # [b*t, 4]
        target_boxes = torch.cat([t["boxes"] for t in targets], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bboxhs"] = loss_bbox.sum() / max(num_boxes, 1)

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giouhs"] = loss_giou.sum() / max(num_boxes, 1)
        return losses

    def loss_sted(self, outputs, num_boxes, inter_idx, positive_map, time_mask=None, indices=None, keep=None):
        """Compute the losses related to the start & end prediction, a KL divergence loss
        targets dicts must contain the key "pred_sted" containing a tensor of logits of dim [T, 2]
        """
        assert "pred_sted" in outputs
        sted = outputs["pred_sted"]
        idx = self._get_src_permutation_idx(indices)
        # sted = sted[:,idx[1],:]
        sted = sted[idx[1]]
        losses = {}

        target_start = torch.tensor([x[0] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        target_end = torch.tensor([x[1] for x in inter_idx], dtype=torch.long).to(
            sted.device
        )
        sted = sted.masked_fill(
            ~time_mask[:, :, None], -1e32
        )  # put very low probability on the padded positions before softmax
        eps = 1e-6  # avoid log(0) and division by 0

        sigma = self.sigma
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
        pred_start_prob = (sted[:, :, 0]).softmax(1)
        loss_start = (
            pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
        )  # KL div loss
        loss_start = loss_start * time_mask  # not count padded values in the loss

        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
        pred_end_prob = (sted[:, :, 1]).softmax(1)
        loss_end = (
            pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
        )  # KL div loss
        loss_end = loss_end * time_mask  # do not count padded values in the loss

        loss_sted = loss_start + loss_end
        losses["loss_sted"] = loss_sted.mean()

        return losses

    def loss_guided_attn(
        self, outputs, num_boxes, inter_idx, positive_map, time_mask=None, indices=None, keep=None
    ):
        """Compute guided attention loss
        targets dicts must contain the key "weights" containing a tensor of attention matrices of dim [B, T, T]
        """
        weights = outputs["weights"]  # BxTxT
        idx = self._get_src_permutation_idx(indices)
        # sted = sted[:,idx[1],:]
        weights = weights[idx[1]]
        positive_map = positive_map + (
            ~time_mask
        )  # the padded positions also have to be taken out
        eps = 1e-6  # avoid log(0) and division by 0

        loss = -(1 - weights + eps).log()
        loss = loss.masked_fill(positive_map[:, :, None], 0)
        nb_neg = (~positive_map).sum(1) + eps
        loss = loss.sum(2) / nb_neg[:, None]  # sum on the column
        loss = loss.sum(1)  # mean on the line normalized by the number of negatives
        loss = loss.mean()  # mean on the batch

        losses = {"loss_guided_attn": loss}
        return losses

    def get_loss(
        self,
        loss,
        outputs,
        targets,
        indices,
        num_boxes,
        inter_idx,
        positive_map,
        time_mask,
        keep,
        **kwargs,
    ):
        loss_map = {
            "label": self.loss_labels,
            "boxes": self.loss_boxes,
            "sted": self.loss_sted,
            "guided_attn": self.loss_guided_attn,
            "cross_sim": self.loss_cross_sim,
            "boxes_hs": self.loss_bboxhs,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss in ["label","sted","guided_attn","cross_sim"]:
            return loss_map[loss](
                outputs, num_boxes, inter_idx, positive_map, time_mask, indices, keep, **kwargs
            )
        return loss_map[loss](outputs, targets, num_boxes, indices, **kwargs)

    def forward(self, outputs, targets, inter_idx=None, time_mask=None, keep=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == n_annotated_frames.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             inter_idx: list of [start index of the annotated moment, end index of the annotated moment] for each video
             time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the loss computation
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        if inter_idx is not None and time_mask is not None:
            # construct a map such that positive_map[k, i] = True iff num_frame i lies inside the annotated moment k
            positive_map = torch.zeros(time_mask.shape, dtype=torch.bool)
            for k, idx in enumerate(inter_idx):
                if idx[0] < 0:  # empty intersection
                    continue
                positive_map[k][idx[0] : idx[1] + 1].fill_(True)

            positive_map = positive_map.to(time_mask.device)
        elif time_mask is None:
            positive_map = None

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    indices,
                    num_boxes,
                    inter_idx,
                    positive_map,
                    time_mask,
                    keep,
                )
            )

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.axu_losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_boxes,
                        inter_idx,
                        positive_map,
                        time_mask,
                        keep,
                        **kwargs,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build(args):
    if args.binary:
        num_classes = 1
    else:
        if args.combine_datasets == "vidstg":
            num_classes = 79
        else:
            num_classes = 1  # for coco
    device = torch.device(args.device)

    backbone = build_backbone(args)
    matcher = build_matcher(args)

    transformer = build_transformer(args)

    model = TubeDETR(
        num_classes,
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        video_max_len=args.video_max_len_train,
        stride=args.stride,
        guided_attn=args.guided_attn,
        fast=args.fast,
        fast_mode=args.fast_mode,
        sted=args.sted,
    )
    weight_dict = {
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_bboxhs": args.hsbbox_loss_coef,
        "loss_giouhs": args.hsgiou_loss_coef,
        "loss_sted": args.sted_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_cross_sim": args.cross_sim,
    }
    if args.guided_attn:
        weight_dict["loss_guided_attn"] = args.guided_attn_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["boxes", "sted", "label","cross_sim","boxes_hs"] if args.sted else ["boxes", "label","cross_sim","boxes_hs"]
    axu_losses = ["boxes", "sted", "label"] if args.sted else ["boxes", "label"]
    if args.guided_attn:
        losses += ["guided_attn"]
        axu_losses = axu_losses + ["guided_attn"]

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        losses=losses,
        axu_losses=axu_losses,
        sigma=args.sigma,
    )
    criterion.to(device)

    return model, criterion, weight_dict