#modified from Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor
import copy
import os
import torch.nn.functional as F


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv3d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :,:] = self.partial_conv3(x[:, :self.dim_conv3, :, :,:])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x
    
class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv3d(in_channels=in_channels*2, out_channels=out_channels*2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.InstanceNorm3d(out_channels*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w, d = x.size()
        # print(x.size())
        r_size = x.size()
        x = x.to(torch.float32)

        ffted = torch.fft.rfft2(x, dim=(2, 3, 4), norm='ortho')         # (batch, c, h, w/2+1)

        # print('1',ffted.shape) #[1, 16, 24, 13]
        # (batch, c, 2, h, w/2+1)
        # ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        # ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = torch.view_as_real(ffted) #[1, 16, 24, 13, 2] # (batch, c, h, w, d/2+1,2)
        ffted = ffted.permute(0, 1, 5, 2, 3, 4).contiguous() 
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])# (batch, c*2, h, w, d/2+1)

        # print('2',ffted.shape)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w, d/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 5, 2).contiguous()  # (batch,c, h, w， d/2+1, 2)
        # print('3',ffted.shape) #[1, 16, 24, 13, 2]
        ffted = torch.view_as_complex(ffted) # (batch,c, h,w,d/2+1)
        # print('4',ffted.shape) #[1, 16, 24, 13, 2]

        output = torch.fft.irfft2(ffted, s= r_size[2:], dim=(2, 3, 4), norm='ortho')

        return output

class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv3d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv3d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            # print(self.layer_scale.shape)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward
        # print(self.drop_path)
        # self.fft = FourierUnit(dim, dim,)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)#+self.fft(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)#+self.fft(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        # self.reduction = (nn.MaxPool3d(patch_stride2))
        self.reduction = nn.Conv3d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x

class upsampling(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        # self.reduction = (nn.MaxPool3d(patch_stride2))
        self.reduction = nn.ConvTranspose3d(dim, int(dim/2), kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x



class FasterNet(nn.Module):

    def __init__(self,
                 in_chans=1,
                 num_classes=14,
                 embed_dim=32,
                 decoder_dim = 64,
                 dims = (32, 64, 128, 256),
                 depths=(2, 2, 2, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4, # patch embededing
                 patch_stride=4, 
                 patch_size2=2,  # for subsequent layers downsampling
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280, # for classfication
                 drop_path_rate=0.1,  # deactive when test
                 layer_scale_init_value=0,
                 norm_layer='IN',
                 act_layer='RELU',
                 fork_feat=True, # for dense prediction
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm3d
        if norm_layer == 'IN':
            norm_layer = nn.InstanceNorm3d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1)) # # for classfication
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage),
                               n_div=n_div,
                               depth=depths[i_stage],
                               mlp_ratio=self.mlp_ratio,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               layer_scale_init_value=layer_scale_init_value,
                               norm_layer=norm_layer,
                               act_layer=act_layer,
                               pconv_fw_type=pconv_fw_type
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** i_stage),
                                 norm_layer=norm_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        # build defcoder layers
        self.decode_stages_list = nn.ModuleList()
        self.trans_up = nn.ModuleList()

        for j_stage in range(self.num_stages): 
                # print(j_stage)
                 # upsample layer
                j_stage = (self.num_stages-1) - j_stage #3,2,1,0
                if j_stage == 0:
                    self.outconv0 = nn.ConvTranspose3d(
                    embed_dim * 2 ** j_stage , num_classes, kernel_size=4, stride=4,)
                else:
                    self.trans_up.append(upsampling(patch_size2=patch_size2,
                                 patch_stride2=patch_stride2,
                                 dim=int(embed_dim * 2 ** j_stage),
                                 norm_layer=norm_layer))
                

                if  j_stage > 0:
                    # print(j_stage)
                    # print(len(self.decode_stages_list))

                    j_stage = j_stage-1
                    decode_stages = BasicStage(dim=int(embed_dim * 2 ** (j_stage)),
                                    n_div=n_div,
                                    depth=depths[j_stage],
                                    mlp_ratio=self.mlp_ratio,
                                    drop_path=dpr[sum(depths[:j_stage]):sum(depths[:j_stage + 1])],
                                    layer_scale_init_value=layer_scale_init_value,
                                    norm_layer=norm_layer,
                                    act_layer=act_layer,
                                    pconv_fw_type=pconv_fw_type
                                    )
                    # print('555')
                    self.decode_stages_list.append(decode_stages)

                # print(len(self.decode_stages_list))



        self.fork_feat = fork_feat

        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv3d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv3d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv3d(decoder_dim, num_classes, 1),
        )

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()


    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        origin_input = x
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                # print(f'norm{idx}')
                norm_layer1 = getattr(self, f'norm{idx}')
                # print(type(norm_layer1))
                x_out = norm_layer1(x)
                outs.append(x_out)
        # segforer decoeder
        fused = [to_fused(output) for output, to_fused in zip(outs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        y = self.to_segmentation(fused)
        outs = F.interpolate(y, size=origin_input.shape[-3:], mode='trilinear', align_corners=True)




        #dense decoder
        # print(len(self.trans_up)) 
        # print(len(self.decode_stages_list)) 

        # up2 = self.trans_up[0](outs[3]) # encoder, decoder stage index 顺序相反 
        # dec2 = self.decode_stages_list[0](up2 + outs[2])

        # up1 = self.trans_up[1](dec2)
        # dec1 = self.decode_stages_list[1](up1 + outs[1])

        # up0 = self.trans_up[2](dec1)
        # dec0 = self.decode_stages_list[2](up0 + outs[0])

        # outs = self.outconv0(dec0)



        
        return outs

        # return outs

if __name__ == '__main__':   
    Unet = FasterNet().cuda()
    # x = torch.ones((1,1,192,192,192)).cuda()#注意h w 尺寸大小,magesize/ patchsize = segment_dimension,embed_dim 要能整除segment_dim
    x = torch.ones((1,1,128,128,128)).cuda()#注意h w 尺寸大小,magesize/ patchsize = segment_dimension,embed_dim 要能整除segment_dim

    y = Unet(x)
    print(y.shape)

    # print(len(y))
    # for i in y:
    #      print(i.shape)