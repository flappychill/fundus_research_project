import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import build_backbone, feature_channels

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learn_p=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p) if learn_p else torch.tensor([p], requires_grad=False)
        self.eps = eps
    def forward(self, x):
        x = torch.clamp(x, min=self.eps).pow(self.p)
        return x.mean(dim=(-1,-2)).pow(1.0/self.p)

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(c//r, c, 1, bias=False), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, groups=1, act=True):
        super().__init__()
        if p is None: p = (k//2)*d
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, dilation=d, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SpatialMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, pool_stride=2, dropout=0.0):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.pool_stride = pool_stride
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B, C, H, W = x.shape
        q = x.flatten(2).transpose(1,2)
        if self.pool_stride > 1:
            kvin = F.avg_pool2d(x, kernel_size=self.pool_stride, stride=self.pool_stride)
        else:
            kvin = x
        kv = kvin.flatten(2).transpose(1,2)
        q = self.q_proj(q); k = self.k_proj(kv); v = self.v_proj(kv)
        q = q.transpose(0,1); k = k.transpose(0,1); v = v.transpose(0,1)
        out,_ = self.mha(q,k,v,need_weights=False)
        out = out.transpose(0,1)
        out = self.out(out); out = self.drop(out)
        return out.transpose(1,2).reshape(B, C, H, W)

class PlainHead(nn.Module):
    def __init__(self, num_classes, backbone_name, pooling="gap", dropout=0.3):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        out_ch = self.backbone.feature_info[-1]['num_chs']
        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.forward_pool = lambda x: self.pool(x).flatten(1)
        else:
            self.pool = GeM()
            self.forward_pool = lambda x: self.pool(x)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_ch, num_classes)
    def forward(self, x):
        f = self.backbone(x)[-1]
        z = self.forward_pool(f); z = self.bn(z); z = self.drop(z)
        return self.fc(z)

class StrongHead(nn.Module):
    def __init__(self, num_classes, backbone_name, pooling="gem", dropout=0.3, use_se=True):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        out_ch = self.backbone.feature_info[-1]['num_chs']
        self.se = SE(out_ch) if use_se else nn.Identity()
        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.forward_pool = lambda x: self.pool(x).flatten(1)
        else:
            self.pool = GeM()
            self.forward_pool = lambda x: self.pool(x)
        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_ch, num_classes)
    def forward(self, x):
        f = self.backbone(x)[-1]
        f = self.se(f)
        z = self.forward_pool(f); z = self.bn(z); z = self.drop(z)
        return self.fc(z)

class FPNHead(nn.Module):
    def __init__(self, num_classes, backbone_name, feat_ch=256, pooling="gem", dropout=0.3, use_se=True):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        fi = self.backbone.feature_info
        idxs, chs = feature_channels(fi, 3)
        self.idxs = idxs
        self.lateral = nn.ModuleList([nn.Conv2d(c, feat_ch, 1, bias=False) for c in chs])
        self.refine  = nn.ModuleList([nn.Conv2d(feat_ch, feat_ch, 3, padding=1, bias=False) for _ in chs])
        self.seblk   = nn.ModuleList([SE(feat_ch) if use_se else nn.Identity() for _ in chs])
        self.fuse    = nn.Conv2d(feat_ch*3, feat_ch, 1, bias=False)
        self.bn_fuse = nn.BatchNorm2d(feat_ch)
        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.forward_pool = lambda x: self.pool(x).flatten(1)
        else:
            self.pool = GeM()
            self.forward_pool = lambda x: self.pool(x)
        self.bn_fc = nn.BatchNorm1d(feat_ch)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(feat_ch, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        c3,c4,c5 = [feats[i] for i in self.idxs]
        p5 = self.lateral[2](c5)
        p4 = self.lateral[1](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lateral[0](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p5 = self.seblk[2](self.refine[2](p5)); p4 = self.seblk[1](self.refine[1](p4)); p3 = self.seblk[0](self.refine[0](p3))
        p4u = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5u = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        fm  = torch.cat([p3, p4u, p5u], dim=1); fm = self.bn_fuse(self.fuse(fm))
        z = self.forward_pool(fm); z = self.bn_fc(z); z = self.drop(z)
        return self.fc(z)

class RevPyrHead(nn.Module):
    def __init__(self, num_classes, backbone_name, feat_ch=256, pooling="gem", dropout=0.3, use_se=True, dilations=(2,3)):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        fi = self.backbone.feature_info
        idxs, chs = feature_channels(fi, 3)
        self.idxs = idxs
        self.l3 = nn.Conv2d(chs[0], feat_ch, 1, bias=False)
        self.l4 = nn.Conv2d(chs[1], feat_ch, 1, bias=False)
        self.l5 = nn.Conv2d(chs[2], feat_ch, 1, bias=False)
        self.ds3 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, s=2), ConvBNAct(feat_ch, feat_ch, k=3, s=2))
        self.ds4 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, s=2))
        self.ref3 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[0]), SE(feat_ch) if use_se else nn.Identity())
        self.ref4 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[0]), SE(feat_ch) if use_se else nn.Identity())
        self.ref5 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[1]), SE(feat_ch) if use_se else nn.Identity())
        self.fuse = nn.Sequential(nn.Conv2d(feat_ch*3, feat_ch, 1, bias=False), nn.BatchNorm2d(feat_ch), nn.SiLU(inplace=True), ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[1]))
        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.forward_pool = lambda x: self.pool(x).flatten(1)
        else:
            self.pool = GeM()
            self.forward_pool = lambda x: self.pool(x)
        self.bn_fc = nn.BatchNorm1d(feat_ch)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(feat_ch, num_classes)
    def forward(self, x):
        c3,c4,c5 = [self.backbone(x)[i] for i in self.idxs]
        p3 = self.l3(c3); p4 = self.l4(c4); p5 = self.l5(c5)
        p3d = self.ds3(p3); p4d = self.ds4(p4)
        p3d = self.ref3(p3d); p4d = self.ref4(p4d); p5r = self.ref5(p5)
        fm = torch.cat([p3d, p4d, p5r], dim=1); fm = self.fuse(fm)
        z = self.forward_pool(fm); z = self.bn_fc(z); z = self.drop(z)
        return self.fc(z)

class RPAttnHead(nn.Module):
    def __init__(self, num_classes, backbone_name, feat_ch=256, pooling="gem", dropout=0.3, use_se=True, dilations=(2,3), num_heads=4, attn_pool_stride=2):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        fi = self.backbone.feature_info
        idxs, chs = feature_channels(fi, 3)
        self.idxs = idxs
        self.l3 = nn.Conv2d(chs[0], feat_ch, 1, bias=False)
        self.l4 = nn.Conv2d(chs[1], feat_ch, 1, bias=False)
        self.l5 = nn.Conv2d(chs[2], feat_ch, 1, bias=False)
        self.ds3 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, s=2), ConvBNAct(feat_ch, feat_ch, k=3, s=2))
        self.ds4 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, s=2))
        self.ref3 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[0]), SE(feat_ch) if use_se else nn.Identity())
        self.ref4 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[0]), SE(feet_ch) if False else SE(feat_ch) if use_se else nn.Identity())
        self.ref5 = nn.Sequential(ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[1]), SE(feat_ch) if use_se else nn.Identity())
        self.fuse_conv = nn.Sequential(nn.Conv2d(feat_ch*3, feat_ch, 1, bias=False), nn.BatchNorm2d(feat_ch), nn.SiLU(inplace=True), ConvBNAct(feat_ch, feat_ch, k=3, d=dilations[1]))
        self.attn_proj = nn.Conv2d(feat_ch*3, feat_ch, 1, bias=False)
        self.attn = SpatialMHSA(dim=feat_ch, num_heads=num_heads, pool_stride=attn_pool_stride, dropout=0.0)
        self.gate = nn.Sequential(nn.Conv2d(feat_ch*2, feat_ch, 1, bias=False), nn.BatchNorm2d(feat_ch), nn.Sigmoid())
        if pooling == "gap":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.forward_pool = lambda x: self.pool(x).flatten(1)
        else:
            self.pool = GeM()
            self.forward_pool = lambda x: self.pool(x)
        self.bn_fc = nn.BatchNorm1d(feat_ch)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(feat_ch, num_classes)
    def forward(self, x):
        feats = self.backbone(x); c3,c4,c5 = [feats[i] for i in self.idxs]
        p3 = self.l3(c3); p4 = self.l4(c4); p5 = self.l5(c5)
        p3d = self.ds3(p3); p4d = self.ds4(p4)
        p3d = self.ref3(p3d); p4d = self.ref4(p4d); p5r = self.ref5(p5)
        cat = torch.cat([p3d, p4d, p5r], dim=1)
        conv_fused = self.fuse_conv(cat)
        attn_in = self.attn_proj(cat)
        attn_out = self.attn(attn_in)
        g = self.gate(torch.cat([conv_fused, attn_out], dim=1))
        fm = g*attn_out + (1-g)*conv_fused
        z = self.forward_pool(fm); z = self.bn_fc(z); z = self.drop(z)
        return self.fc(z)
