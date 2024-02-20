import timm
import torch
import torch.nn as nn
import math

from einops import rearrange
from timm.models.vision_transformer import Block, Attention
from utils.util import ChannelAttention, SpatialAttention
from utils.util import DeformConv2d, Depth_wise_separable_conv
#from util import ChannelAttention, SpatialAttention
#from util import DeformConv2d, Depth_wise_separable_conv


class HookBlock:
    def __init__(self):
        self.output = []

    def __call__(self, module, input, output):
        self.output.append(output)

    def clear(self):
        self.output = []


class Multi_branch(nn.Module):
    def __init__(self,
                 dim,
                 factor=[256, 256, 256],
                 qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.factor = factor
        self.ca = ChannelAttention(in_channel=dim)
        self.sa = SpatialAttention(k=3)

        self.dconv = DeformConv2d(dim, factor[0])
        self.dconv_proj = nn.Linear(factor[0], factor[0])

        self.dwconv = Depth_wise_separable_conv(dim, factor[1])
        self.max_pool = nn.MaxPool2d(2)
        # self.ups = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.ups = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.attn = Attention(dim, qkv_bias=qkv_bias)
        self.attn_proj = nn.Linear(dim, factor[2])

        self.ups_proj = nn.Linear(36, 49)

    def forward(self, x):
        B, L, _ = x.shape
        _x = rearrange(x, 'b (h w) c -> b c h w',
                       h=int(math.sqrt(L)),
                       w=int(math.sqrt(L)))
        # ca and sa
        residual_x = self.ca(_x)
        residual_x = self.sa(residual_x)
        residual_x = rearrange(residual_x, 'b c h w -> b (h w) c')

        # branch1
        # b1 = self.dconv(residual_x).reshape(B, self.factor[0], -1)
        b1 = self.dconv(_x).reshape(B, self.factor[0], -1)
        b1 = b1.transpose(-2, -1)
        b1 = self.dconv_proj(b1)
        # print(b1.shape)

        # branch2
        b2 = self.max_pool(_x)
        b2 = self.dwconv(b2)
        b2 = self.ups(b2).reshape(B, self.factor[1], -1)
        if b2.shape[2] != b1.shape[1]:
            b2 = self.ups_proj(b2)

        b2 = b2.transpose(-2, -1)
        # print(b2.shape)

        # branch3
        x = rearrange(_x, 'b c h w -> b (h w) c')
        b3 = self.attn(x)
        b3 = self.attn_proj(b3)
        # print(b3.shape)

        # cat
        out = torch.cat([b1, b2, b3], dim=2)
        out = out + residual_x
        return out


class Patch_down_sample(nn.Module):
    def __init__(self, dim):
        super(Patch_down_sample, self).__init__()
        self.down_conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        B, L, _ = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w',
                      h=int(math.sqrt(L)),
                      w=int(math.sqrt(L)))  
        x = self.down_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class NIQA(nn.Module):
    def __init__(self,
                 dim=768, enc_dim=768,
                 #layers=[0, 1, 2, 6],
                 layers=[1, 2, 5, 6],
                 depth=[1, 5, 3],
                 factor=[[256, 256, 256],
                         [192, 192, 384],
                         [48, 48, 672]]):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.hook = HookBlock()
        self.layers = layers
        self.dim = dim
        self.factor = factor
        self.enc_dim = enc_dim

        self.feature_proj = nn.Linear(len(self.layers) * dim, dim, bias=False)

        self.block = nn.Sequential(
            self.build_block(dim, depth=depth[0], factor=factor[0]),
            Patch_down_sample(dim),
            self.build_block(dim, depth=depth[1], factor=factor[1]),
            Patch_down_sample(dim),
            self.build_block(dim, depth=depth[2], factor=factor[2]),
            # Patch_down_sample(dim),
        )

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.dim, 1)
        )
        
        self.fea_proj = nn.Sequential(
            nn.Conv2d(1792, enc_dim, 1, 1)
        )

        self.W_Q = nn.Linear(enc_dim, enc_dim, bias=False)
        self.W_K = nn.Linear(enc_dim, enc_dim, bias=False)
        self.W_V = nn.Linear(enc_dim, enc_dim, bias=False)

        self.scale = enc_dim ** -0.5
    
        self.drop_attn = nn.Dropout(p=0.1)
        
        self.ffn = nn.Sequential(
            nn.Linear(enc_dim, 4*enc_dim, bias=False),
            nn.GELU(),
            nn.Linear(4*enc_dim, enc_dim, bias=False),
            nn.GELU()
        )

        for blk in self.vit.modules():
            if isinstance(blk, Block):
                blk.register_forward_hook(self.hook)

        for p in self.vit.parameters():
            p.requires_grad = False

    def build_block(self, dim, depth, factor):
        return nn.Sequential(*[Multi_branch(dim, factor) for _ in range(depth)])

    def feature_extract(self, layers):
        features = []
        for layer in layers:
            features.append(self.hook.output[layer][:, 1:])

        out = torch.cat([features[i] for i in range(len(features))], dim=2)
        return out

    def forward(self, x, fea):
        fea = self.fea_proj(fea)
        fea = rearrange(fea, 'b c h w -> b (h w) c')

        _x = self.vit(x)

        out = self.feature_extract(self.layers)
        self.hook.clear()
        out = self.feature_proj(out)
        out = self.block(out)

        Q = self.W_Q(fea)
        K = self.W_K(out)
        V = self.W_V(out)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop_attn(attn)

        x = (attn @ V)
        token = x + self.ffn(x)

        B, L, _ = token.shape
        token = rearrange(token, 'b (h w) c -> b c h w',
                          h=int(math.sqrt(L)),
                          w=int(math.sqrt(L)))
        
        out = self.avg(token).reshape(B, -1)
        out = self.fc(out) 
        return out


if __name__ == '__main__':
    # inp = torch.randn(5, 196, 768)
    # mb = Multi_branch(768, factor=[48, 48, 672])
    # print(mb(inp).shape)

    inp = torch.randn(5, 3, 224, 224)
    enc = torch.randn(5, 1792, 7, 7)
    model = NIQA()
    print(model(inp, enc))


    trainable_total_params = sum(p.numel() for p in model.parameters() \
                                 if p.requires_grad)   
#    trainable_total_params = sum(p.numel() for p in model.parameters())                                
    print(f'Params(M): {trainable_total_params/1000/1000:.3f}M')
