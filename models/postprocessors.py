# Adapted from
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform TubeDETR output according to the downstream task"""
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops


class PostProcessSTVG(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, frames_id=None, durations=None, video_ids=None, time_mask=None):
        """
        :param outputs: must contain a key pred_sted mapped to a [B, T, 2] tensor of logits for the start and end predictions
        :param frames_id: list of B lists which contains the increasing list of frame ids corresponding to the indexes of the decoder outputs
        :param video_ids: list of B video_ids, used to ensemble predictions when video_max_len_train < video_max_len
        :param time_mask: [B, T] tensor with False on the padded positions, used to take out padded frames from the possible predictions
        :return: list of B [start_frame, end_frame] for each video
        """
        out_sted = outputs["pred_sted"]  # BxTx2
        b, t, _ = out_sted.shape
        device = out_sted.device
        temp_prob_map = torch.zeros(b, t, t).to(device)
        inf = -1e32
        for i_b in range(len(durations)):
            duration = durations[i_b]
            sted_prob = (torch.ones(t, t) * inf).tril(0).to(device)
            sted_prob[duration:, :] = inf
            sted_prob[:, duration:] = inf
            temp_prob_map[i_b, :, :] = sted_prob

        temp_prob_map += F.log_softmax(out_sted[:, :, 0], dim=1).unsqueeze(2) + \
                         F.log_softmax(out_sted[:, :, 1], dim=1).unsqueeze(1)

        pred_steds = []
        for i_b in range(b):
            prob_map = temp_prob_map[i_b]  # [T * T]
            frame_id_seq = frames_id[i_b]
            prob_seq = prob_map.flatten(0)
            max_tstamp = prob_seq.max(dim=0)[1].item()
            start_idx = max_tstamp // t
            end_idx = max_tstamp % t
            pred_sted = [frame_id_seq[start_idx], frame_id_seq[end_idx] + 1]
            pred_steds.append(pred_sted)
        return pred_steds


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs["pred_boxes"]
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        boxes = boxes.clamp(min=0)
        results = [{"boxes": b} for b in boxes]

        return results


def build_postprocessors(args, dataset_name) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}

    if dataset_name in ["vidstg", "hcstvg"]:
        postprocessors[dataset_name] = PostProcessSTVG()

    return postprocessors