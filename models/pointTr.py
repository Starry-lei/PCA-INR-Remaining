# Proposed model - Point2SSM
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.ae import PointNet_encoder
from models.dgcnn import DGCNN_encoder
from models.src.model import Encode2Points

# from models.AdaPoinTr import 
# from utils.model_utils import calc_cd
from torch_cluster import knn
import torch_cluster
# import xformers.ops
import os
import time
from models.Transformer_utils import *
import sys
from models.src.network.utils import normalize_3d_coordinate, ResnetBlockFC, normalize_coordinate
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
from pointnet2_utils import grouping_operation
from utils.model_utils import calc_cd
from timm.models.layers import DropPath, trunc_normal_
from functools import partial, reduce



criterion_mse = torch.nn.MSELoss(reduction='sum') # reduce=None #  with chamfer_loss reduction='sum': reduction='sum'
def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


######################################## Grouper ########################################  
class DGCNN_Grouper(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.num_features = 128
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f



class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
   
class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x


class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x


class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_input = args.num_input_points
        self.latent_dim = args.latent_dim
        self.num_output = args.num_output_points
        self.train_loss = args.loss
        self.device = args.device
        self.alpha = args.alpha

        self.encoder_name = args.model_name


        self.grouper = DGCNN_Grouper(k = 16)
        self.center_num= [512, 256]
        in_chans = 3
        embed_dim= 384
        global_feature_dim= 1024

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        ) 

        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim)
        )

        self.increase_dim = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        # print("ssss self.latent_dim:",self.latent_dim)
        if args.encoder == 'dgcnn':
            print("ssss use dgcnn:",self.latent_dim)
            self.encoder = DGCNN_encoder(self.latent_dim)
        elif args.encoder == 'local_pool_pointnet':
            self.encoder = Encode2Points(args)
        elif args.encoder == 'PointTransformerEncoderEntry':
            self.encoder = PointTransformerEncoderEntry(args.encoder_config)
        else:
            print("Unimplemented encoder: " + str(args.encoder))

        self.attn_module = Attention_Module(self.latent_dim, self.num_output)

        self.denoiser_module= denoiser_module(dim=3, c_dim=128,
                                              out_dim=3,
                                              hidden_size=256,
                                              n_blocks=5,
                                              leaky=True)

    
    
    def forward(self, x, gt=None, is_training=True):


        bs = x.size(0)
        print("show shape of x:",x.shape)
        # torch.Size([20, 2048, 3])
        coor, f = self.grouper(x, self.center_num) # b n c
        pe =  self.pos_embed(coor)
        print("show shape of ssspe:",pe.shape)# torch.Size([20, 256, 384])
        x = self.input_proj(f)
        print("show shape of aaaax:",x.shape)# torch.Size([20, 256, 384])
        print("show shape of coor:",coor.shape)# torch.Size([20, 256, 3])
       
        x = self.encoder(x + pe, coor)

        print("show shape of x:",x.shape) # 20, 256, 384

        # global_feature = self.increase_dim(x) # B 1024 N 
        # global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        # print("show shape of f:",f.shape)# torch.Size([20, 256, 128])
        
        print("show shape of x:",x.shape)# 20, 256, 384
        

        exit()

        # print("show shape of x:",x.shape)  # torch.Size([5, 2048, 3])
        z, features = self.encoder(x)



        # print("show shape of features:",features.shape)
        features = features.transpose(2, 1)  # [B, N, latent_dim]
        # print("show shape of transposed features:", features.shape)
        residuals = self.attn_module(features, coords=x) # [B, N, 3]
        # residuals = self.denoiser_module(x, features) # [B, N, 3]

        pred =  x + residuals
        if is_training:            
            cd_p, cd_t = calc_cd(pred, gt)
            recon_loss = cd_t
            loss= mean_flat(recon_loss) #+ self.alpha * mean_flat(neigh_loss)
            return pred, loss
        else:
            return pred
        

    # def forward(self, x, gt=None, is_training=True):
      
    #     out = self.encoder(x)
    #     points, normals = out
    #     pred= points-normals 
    #     # print("show shape of points:",points.shape)
    #     # density num_offset=7
    #     #show shape of points: torch.Size([16, 14336, 3])
    #     # print("show shape of normals:",normals.shape)
    #     # density num_offset=7
    #     # show shape of normals: torch.Size([16, 14336, 3])
    #     # density num_offset=1
    #     # show shape of normals: torch.Size([16, 2048, 3])
    #     if self.encoder_name == 'dgcnn':
    #         features = features.transpose(2, 1)

    #     # prob_map = self.attn_module(features)
    #     # pred = torch.sum(prob_map[:, :, :, None] * x[:, None, :, :], dim=2)

    #     if is_training:
    #         # cd_p, cd_t = criterion_mse(pred, gt)
    #         # loss_mse = criterion_mse(pred, gt)
    #         cd_p, cd_t = calc_cd(pred, gt)
    #         recon_loss = cd_t
    #         # recon_loss=  mean_flat(loss_mse)            
    #         # neigh_loss = self.get_neighbor_loss(pred, 10)
    #         loss= mean_flat(recon_loss) #+ self.alpha * mean_flat(neigh_loss)
    #         return pred, loss
    #     else:
   
    #         return pred
        



    def get_prob_map(self, x):
        z, features = self.encoder(x)

        if self.encoder_name == 'dgcnn':
            features = features.transpose(2, 1)

        prob_map = self.attn_module(features)

        pred = torch.sum(prob_map[:, :, :, None] * x[:, None, :, :], dim=2)

        return prob_map, pred


# class Attention_Module(nn.Module):
#     def __init__(self, latent_dim, num_output):
#         super(Attention_Module, self).__init__()
#         self.num_output = num_output
#         self.latent_dim = latent_dim
#         print("++++++++++++++++++++++++++++++++++++++++++checking latent_dim, num_output:", latent_dim,
#               num_output)  # 128 2048
#         time_rec_1 = time.time()
#         self.sa1 = cross_transformer(self.latent_dim, self.num_output)
#         self.sa2 = cross_transformer(self.num_output, self.num_output)
#         self.sa3 = cross_transformer(self.num_output, self.num_output)
#         self.softmax = nn.Softmax(dim=2)
#         print("cross atten using time:", str(time.time() - time_rec_1)[:4])

#     def forward(self, x):
#         x = self.sa1(x, x)
#         x = self.sa2(x, x)
#         x = self.sa3(x, x)
#         prob_map = self.softmax(x)
#         return prob_map


class ImNet(nn.Module): # develop a transformation based IMNet?
    """ImNet layer pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=128,
        out_features=3,
        nf=16,
        nonlinearity="leakyrelu",
    ):
        """Initialization.

        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = nn.LeakyReLU()
        self.fc0 = nn.Linear(self.dimz, nf * 16)
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        # A coord-condition MLP
        """Forward method. 
        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)

        x_tmp = self.activ(self.fc4(x_tmp))

        x_tmp = self.fc5(x_tmp)
        return x_tmp



class Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output, n_blocks= 3):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim


        self.n_blocks=n_blocks

        

        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        # self.sa3 = cross_transformer(self.num_output,self.num_output)


        self.coord_cond_mlp = ImNet(dim=3,in_features=self.num_output, out_features=3, nf=32)


       
        # Final projection to residuals [B, N, latent_dim] -> [B, N, 3]
        # self.residual_proj = nn.Linear(num_output, 3)


        for m in self.coord_cond_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                # nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.constant_(m.bias, val=0)


    def forward(self, x, coords=None):

        x = self.sa1(x,x)
        x = self.sa2(x,x)
        # x = self.sa3(x,x)
        

        x = x.permute(0,2,1)
        # print("show shape of x:",x.shape) # 5, 2048, 256
        # print("show shape of coords:",coords.shape)# 5, 2048, 3
        net_input = torch.cat([coords, x], dim=-1)
        out = self.coord_cond_mlp(net_input)


        # out = self.residual_proj(nn.LeakyReLU(net))
        print("show shape of out:",out.shape) # e([5, 2048, 3])  

        # exit()

        return out
    



class Residual_Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output, n_blocks= 3):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim

        self.n_blocks=n_blocks
        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        # self.sa3 = cross_transformer(self.num_output,self.num_output)
        # Final projection to residuals [B, N, latent_dim] -> [B, N, 3]
        self.residual_proj = nn.Linear(num_output, 3)
    def forward(self, x, coords=None):

        x = self.sa1(x,x)
        x = self.sa2(x,x)
        # x = self.sa3(x,x)
        

        # x = x.permute(0,2,1)
        # print("show shape of x:",x.shape) # 5, 2048, 256
        # print("show shape of coords:",coords.shape)# 5, 2048, 3
        # net_input = torch.cat([coords, x], dim=-1)
        # out = self.coord_cond_mlp(net_input)

        out = self.residual_proj(x)
        # print("show shape of out:",out.shape) # e([5, 2048, 3])  

        return out
    

class denoiser_module(nn.Module):
    def __init__(self, dim=3, c_dim=128, out_dim=3, hidden_size=256, n_blocks=5, leaky=False):
        super(denoiser_module, self).__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        self.in_dim = dim + self.c_dim

        self.fc_in = nn.Linear(self.in_dim, hidden_size)
        



        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)


        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size, activation=self.actvn) for _ in range(n_blocks)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(n_blocks)
        ])


        self.fc_out = nn.Linear(hidden_size, out_dim)
        self.activation = self.actvn

        # self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        # self.sa2 = cross_transformer(self.num_output,self.num_output)
        # self.sa3 = cross_transformer(self.num_output,self.num_output)

        

       
        # Final projection to residuals [B, N, latent_dim] -> [B, N, 3]
        # self.residual_proj = nn.Linear(num_output, 3)


    def forward(self, coords, local_feats):

        B, N, _ = coords.shape

        # Concatenate coords + local_feats => (B, N, 3 + F)

        # print("show shape of coords:",coords.shape)
        # print("show shape of local_feats:",local_feats.shape)
        # exit()
        net_input = torch.cat([coords, local_feats], dim=-1)  # => (B, N, in_dim)

        net_input = net_input.view(B*N, -1)

        net = self.fc_in(net_input)          # => (B*N, hidden_size)
        net = self.activation(net)


        # for block in self.blocks:
        #     net = block(net)

        for i, block in enumerate(self.blocks):
            net = block(net)         # residual block
            net = self.actvn(net)    # activation
            net = self.norms[i](net) # layer normalization

        out = self.fc_out(self.activation(net))  # => (B*N, out_dim)
        
        # reshape to (B, N, out_dim)
        residuals = out.view(B, N, -1)

        # print("show shape of residuals:",residuals.shape)
        # exit()

        # x = self.sa1(x,x)
        # x = self.sa2(x,x)
        # x = self.sa3(x,x)

        # residuals = self.residual_proj(x)  # [B, N, 3]
        return residuals




# PointAttN: You Only Need Attention for Point Cloud Completion
# https://github.com/ohhhyeahhh/PointAttN
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=8, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # self.cross_attn = MultiHeadCrossAttention(d_model_out, nhead, attn_drop=dropout)
        # self.multihead_attn1 = FlashAttentionCrossAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # transformer
    def forward(self, src1, src2, if_act=False):

        # print("\n new**start:**********************************see shape of src1:",src1.shape)
        #  torch.Size([5, 128, 2048])  

        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        # print("\n **start:**********************************see shape of src1:",src1.shape)
        # # [5, 256, 2048]
        # print("****************************************see shape of src2:",src2.shape)
        # # [5, 256, 2048]

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]
        
        # print("********************************************************see shape of src12:",src12.shape)
        # #  torch.Size([2048, 5, 256])
        # print("********************************************************see shape of src1:",src1.shape)
        # #  torch.Size([2048, 5, 256])
        # print("********************************************************see shape of src2:",src2.shape)
        # #  torch.Size([2048, 5, 256])

        # src12 = self.cross_attn(x=src1, cond=src2)
        # print("********************************************************see shape of src12:",src12.shape)
        # cross_attn: see shape of src12: torch.Size([2048, 16, 2048]
        # multihead_attn1 see shape of src12: torch.Size([2048, 16, 2048])
        

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        # print("************************************see shape of src1:",src1.shape)
        # 2048, 5, 256])

        # exit()

        src1 = src1.permute(1, 2, 0)

        return src1



