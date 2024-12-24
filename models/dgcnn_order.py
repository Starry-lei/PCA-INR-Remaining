#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""

import torch
import torch.nn as nn
import torch_cluster

class DGCNN_encoder_order(nn.Module):
    def __init__(self, latent_dim):
        super(DGCNN_encoder_order, self).__init__()
        self.num_neighs = 8 #27
        self.latent_dim = latent_dim
        self.input_features =(3+1) * 2
        self.only_true_neighs = True
        self.depth = 4
        # self.depth = 3
        bb_size = 24
        output_dim = self.latent_dim # 768 # 128
        self.convs = []
        for i in range(self.depth):
            in_features = self.input_features if i == 0 else bb_size * (2 ** (i+1)) * 2
            out_features = bb_size * 4 if i == 0 else in_features
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, bias=False), nn.BatchNorm2d(out_features), nn.LeakyReLU(negative_slope=0.2),
            )
        )
        last_in_dim = bb_size * 2 * sum([2 ** i for i in range(1,self.depth + 1,1)])
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False), nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
            )
        )
        self.convs = nn.ModuleList(self.convs)

    def forward_per_point(self, x, start_neighs=None):


        self.num_points = x.shape[1]
        x = x.transpose(1, 2)  # DGCNN assumes BxFxN


        # Add position encoding as additional features
        batch_size, num_features, num_points = x.shape
        position_ids = torch.arange(0, num_points, dtype=torch.float32, device=x.device)
        position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)  # [B, N]
        position_encoding = (position_ids / num_points).unsqueeze(1)  # [B, 1, N]


        # Concatenate position encoding as additional feature
        x = torch.cat([x, position_encoding], dim=1)  # [B, F+1, N] # see x shape here: torch.Size([8, 4, 1024])

     
        
        x = get_graph_feature(x, k=self.num_neighs, idx=start_neighs, only_intrinsic=False)#only_intrinsic=self.hparams.only_intrinsic)
        other = x[:,:3,:,:]

        outs = [x]
        for conv in self.convs[:-1]:
            if(len(outs) > 1):
                x = get_graph_feature(outs[-1], k=self.num_neighs, idx=None if not self.only_true_neighs else start_neighs)
            x = conv(x)
            outs.append(x.max(dim=-1, keepdim=False)[0])

        x = torch.cat(outs[1:], dim=1)
        features = self.convs[-1](x)
        return features.transpose(1,2)


    # Example: each point has 2 neighbors: [i-1, i+1], with boundary clamping
    def build_index_neighbors(self, num_points, k=2):
        """
        Returns a (num_points, k) Tensor containing the index-based neighbors
        for each point in a single sample.
        """
        idxs = []
        half_win = k // 2
        for i in range(num_points):
            local_neighbors = []
            for offset in range(-half_win, half_win + 1):
                if offset == 0:
                    continue
                neigh = i + offset
                neigh = max(0, min(num_points - 1, neigh))  # clamp boundaries
                local_neighbors.append(neigh)
            idxs.append(local_neighbors)
        # shape -> (num_points, k)
        return torch.tensor(idxs, dtype=torch.long)


    def forward(self, x, return_neighs=False):
        self.num_points = x.shape[1]
        batch_size = x.shape[0]
        sigmoid_for_classification=True
        # neigh_idx = torch.stack(
        #     [edge_index[i][1].reshape(x.shape[1], -1) for i in range(x.shape[0])]
        # )
        # Build index-based neighbors for each sample in the batch
        neigh_idx_list = []
        for i in range(batch_size):
            neighs_i = self.build_index_neighbors(num_points=self.num_points, k=self.num_neighs)
            neigh_idx_list.append(neighs_i)
        
        # Stack them to shape (B, 1024, k)
        neigh_idx = torch.stack(neigh_idx_list, dim=0)  # (B, N, k)
        features_per_point = self.forward_per_point(x, start_neighs=neigh_idx) # dense_output_feature, B N F
        global_feature, _ = torch.max(features_per_point.transpose(1,2), 2)


        global_feature = global_feature.view(batch_size, -1)
        if return_neighs:
            return global_feature, features_per_point, neigh_idx
        else:
            return global_feature, features_per_point





def create_feature_neighs(x, neigh_idxs):
    # The output of the function is BxNumxNeighsXF
    batch_size, num_points, num_features = x.shape
    num_neighs = neigh_idxs.shape[-1]
    x = x.transpose(1, 2)
    idx_base = torch.arange(0, batch_size, device=neigh_idxs.device).view(-1, 1, 1) * num_points

    idx = neigh_idxs + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, num_neighs, num_dims)
    return feature



def get_graph_feature(x, k, idx=None, only_intrinsic=False, permute_feature=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if(len(idx.shape)==2):
        idx = idx.unsqueeze(0).repeat(batch_size,1,1)
    idx = idx[:, :, :k]
    k = min(k,idx.shape[-1])

    num_idx = idx.shape[1]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.contiguous()
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_idx, k, num_dims)


    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if only_intrinsic is True:
        feature = feature - x
    else:
        local_diff = feature - x
        seq_weight = 0.5
        feature = torch.cat((local_diff, x), dim=3)  # Keep original dimensionality

    if permute_feature:
        feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature



class DGCNN(nn.Module):
    def __init__(self, hparams, output_channels=40, latent_dim=None):
        super(DGCNN, self).__init__()


    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = default_arg_parser(parents=[parent_parser], is_lowest_leaf=is_lowest_leaf)


        parser.add_argument(
            "--only_true_neighs", nargs="?", default=True, type=str2bool, const=True, help="Use the grpah neightborhood in all dgcnn steps or only at the first iteration",
        )
        parser.add_argument(
            "--use_inv_features", nargs="?", default=False, type=str2bool, const=True, help="Evaluate sensetivity to noise",
        )
        parser.add_argument("--concat_xyz_to_inv", nargs="?", default=False, type=str2bool, const=True,)

        parser.add_argument(
            "--DGCNN_latent_dim",
            type=int,
            default=512,
        )
        parser.add_argument("--bb_size", default=32, type=int, help="the building block size")
        parser.add_argument("--nn_depth", default=4, type=int, help="num of convs")

        return parser

import argparse
import os
from argparse import ArgumentDefaultsHelpFormatter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def default_arg_parser(description="", conflict_handler="resolve", parents=[], is_lowest_leaf=False):
    """
        Generate the default parser - Helper for readability
        
        Args:
            description (str, optional): name of the parser - usually project name. Defaults to ''.
            conflict_handler (str, optional): wether to raise error on conflict or resolve(take last). Defaults to 'resolve'.
            parents (list, optional): [the name of parent argument managers]. Defaults to [].
        
        Returns:
            [type]: [description]
        """
    description = (
        parents[0].description + description
        if len(parents) != 0 and parents[0] != None and parents[0].description != None
        else description
    )
    parser = argparse.ArgumentParser(
        description=description,
        add_help=is_lowest_leaf,
        formatter_class=ArgumentDefaultsHelpFormatter,
        conflict_handler=conflict_handler,
        parents=parents,
    )

    return parser
