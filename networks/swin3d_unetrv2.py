# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from networks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from networks.swin_transformer_3d import SwinTransformer3D
import pdb

class SwinUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 48,
        patch_size: int = 2,
        depths: Tuple[int, int, int, int] = [2, 2, 2, 2],
        num_heads: Tuple[int, int, int, int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = [7, 7, 7],
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.swinViT = SwinTransformer3D(
            pretrained=None,
            pretrained2d=False,
            patch_size=(patch_size, patch_size, patch_size),
            in_chans=in_channels,
            embed_dim=feature_size,
            depths = depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16*feature_size,
            out_channels=16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block)

        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=16*feature_size,
            out_channels=8*feature_size,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            stride=1,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights['state_dict']:
                print(i)
            self.swinViT.patch_embed.proj.weight.copy_(weights['state_dict']['module.patch_embed.proj.weight'])
            self.swinViT.patch_embed.proj.bias.copy_(weights['state_dict']['module.patch_embed.proj.bias'])
            
            # layer1
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.loadFrom(weights, n_block=bname, layer='layers1')
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers1.0.downsample.reduction.weight'])
            self.swinViT.layers1[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers1.0.downsample.norm.weight'])
            self.swinViT.layers1[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers1.0.downsample.norm.bias'])
            # layer2
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.loadFrom(weights, n_block=bname, layer='layers2')
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers2.0.downsample.reduction.weight'])
            self.swinViT.layers2[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers2.0.downsample.norm.weight'])
            self.swinViT.layers2[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers2.0.downsample.norm.bias'])
            # layer3
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.loadFrom(weights, n_block=bname, layer='layers3')
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers3.0.downsample.reduction.weight'])
            self.swinViT.layers3[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers3.0.downsample.norm.weight'])
            self.swinViT.layers3[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers3.0.downsample.norm.bias'])
            # layer4
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.loadFrom(weights, n_block=bname, layer='layers4')
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(weights['state_dict']['module.layers4.0.downsample.reduction.weight'])
            self.swinViT.layers4[0].downsample.norm.weight.copy_(weights['state_dict']['module.layers4.0.downsample.norm.weight'])
            self.swinViT.layers4[0].downsample.norm.bias.copy_(weights['state_dict']['module.layers4.0.downsample.norm.bias'])


            # last norm layer of transformer
            self.swinViT.norm.weight.copy_(weights['state_dict']['module.norm.weight'])
            self.swinViT.norm.bias.copy_(weights['state_dict']['module.norm.bias'])
            
    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in)
        enc0 = self.encoder1(x_in)
        x1 = hidden_states_out[0]
        enc1 = self.encoder2(x1)
        x2 = hidden_states_out[1]
        enc2 = self.encoder3(x2)
        x3 = hidden_states_out[2]
        enc3 = self.encoder4(x3)
        x4 = hidden_states_out[3]
        enc4 = x4
        dec4 = hidden_states_out[4]
        dec4 = self.encoder10(dec4)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits
