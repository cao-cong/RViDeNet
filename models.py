import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
try:
    from modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
from modules.cbam import CBAM
from modules.dual_attention import CAM_Module, TAM_Module
from modules.cc_attention import CrissCrossAttention

class ISP(nn.Module):

    def __init__(self):
        super(ISP, self).__init__()
        
        self.conv1_1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
                  
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
   
        self.upv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upv5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv5_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv6_1 = nn.Conv2d(32, 12, kernel_size=1, stride=1)
    
    def forward(self, x):
        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))   
        
        up4 = self.upv4(conv3)
        up4 = torch.cat([up4, conv2], 1)
        conv4 = self.lrelu(self.conv4_1(up4))
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv1], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        conv6 = self.conv6_1(conv5)
        out = nn.functional.pixel_shuffle(conv6, 2)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

    def lrelu(self, x):
        outt = torch.max(0.2*x, x)
        return outt

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock_noBN(nn.Module):

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class Predenoiser(nn.Module):

    def __init__(self, nf=64):
        super(Predenoiser, self).__init__()

        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.upv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv5_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.upv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.upv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv8_1 = nn.Conv2d(64, 4, kernel_size=1, stride=1)

    def forward(self, x):

        conv1 = self.lrelu(self.conv1_1(x))
        conv1 = self.lrelu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        
        conv2 = self.lrelu(self.conv2_1(pool1))
        conv2 = self.lrelu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)
        
        conv3 = self.lrelu(self.conv3_1(pool2))
        conv3 = self.lrelu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)
        
        conv4 = self.lrelu(self.conv4_1(pool3))
        conv4 = self.lrelu(self.conv4_2(conv4))
        
        up5 = self.upv5(conv4)
        up5 = torch.cat([up5, conv3], 1)
        conv5 = self.lrelu(self.conv5_1(up5))
        conv5 = self.lrelu(self.conv5_2(conv5))
        
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv2], 1)
        conv6 = self.lrelu(self.conv6_1(up6))
        conv6 = self.lrelu(self.conv6_2(conv6))
        
        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv1], 1)
        conv7 = self.lrelu(self.conv7_1(up7))
        conv7 = self.lrelu(self.conv7_2(conv7))
        
        conv8= self.conv8_1(conv7)
        out = conv8
 
        return out

    def lrelu(self, x):
        out = torch.max(0.2*x, x)
        return out

class Alignment(nn.Module):

    def __init__(self, nf=64, groups=1):
        super(Alignment, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised):
        '''align other neighboring frames to the reference frame in the feature level
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l_predenoised[2], ref_fea_l_predenoised[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea_predenoised = self.lrelu(self.L3_dcnpack(nbr_fea_l_predenoised[2], L3_offset))
        L3_fea_noisy = self.lrelu(self.L3_dcnpack(nbr_fea_l_noisy[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l_predenoised[1], ref_fea_l_predenoised[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea_predenoised = self.L2_dcnpack(nbr_fea_l_predenoised[1], L2_offset)
        L3_fea_predenoised = F.interpolate(L3_fea_predenoised, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_predenoised = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea_predenoised, L3_fea_predenoised], dim=1)))
        L2_fea_noisy = self.L2_dcnpack(nbr_fea_l_noisy[1], L2_offset)
        L3_fea_noisy = F.interpolate(L3_fea_noisy, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea_noisy = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea_noisy, L3_fea_noisy], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l_predenoised[0], ref_fea_l_predenoised[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea_predenoised = self.L1_dcnpack(nbr_fea_l_predenoised[0], L1_offset)
        L2_fea_predenoised = F.interpolate(L2_fea_predenoised, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea_predenoised = self.L1_fea_conv(torch.cat([L1_fea_predenoised, L2_fea_predenoised], dim=1))
        L1_fea_noisy = self.L1_dcnpack(nbr_fea_l_noisy[0], L1_offset)
        L2_fea_noisy = F.interpolate(L2_fea_noisy, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea_noisy = self.L1_fea_conv(torch.cat([L1_fea_noisy, L2_fea_noisy], dim=1))
        # Cascading
        offset = torch.cat([L1_fea_predenoised, ref_fea_l_predenoised[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea_noisy = self.lrelu(self.cas_dcnpack(L1_fea_noisy, offset))

        return L1_fea_noisy

class Non_Local_Attention(nn.Module):

    def __init__(self, nf=64, nframes=3):
        super(Non_Local_Attention, self).__init__()

        self.conv_before_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())      
        self.conv_before_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_before_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.recurrence = 2
        self.cca = CrissCrossAttention(nf)
        self.ca = CAM_Module()
        self.ta = TAM_Module()

        self.conv_after_cca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ca = nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                                   nn.ReLU())
        self.conv_after_ta = nn.Sequential(nn.Conv2d(nframes, nframes, 3, padding=1, bias=False),
                                   nn.ReLU())

        self.conv_final = nn.Conv2d(nf, nf, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  

        # spatial non-local attention
        cca_feat = self.conv_before_cca(aligned_fea.reshape(-1, C, H, W))
        for i in range(self.recurrence):
            cca_feat = self.cca(cca_feat)
        cca_conv = self.conv_after_cca(cca_feat).reshape(B, N, C, H, W)

        # channel non-local attention
        ca_feat = self.conv_before_ca(aligned_fea.reshape(-1, C, H, W))
        ca_feat = self.ca(ca_feat)
        ca_conv = self.conv_after_ca(ca_feat).reshape(B, N, C, H, W)

        # temporal non-local attention
        ta_feat = self.conv_before_ta(aligned_fea.permute(0, 2, 1, 3, 4).reshape(-1, N, H, W))
        ta_feat = self.ta(ta_feat)
        ta_conv = self.conv_after_ta(ta_feat).reshape(B, C, N, H, W).permute(0, 2, 1, 3, 4)

        feat_sum = cca_conv+ca_conv+ta_conv
        
        output = self.conv_final(feat_sum.reshape(-1, C, H, W)).reshape(B, N, C, H, W)
                
        return aligned_fea + output


class Temporal_Fusion(nn.Module):

    def __init__(self, nf=64, nframes=3, center=1):
        super(Temporal_Fusion, self).__init__()
        self.center = center

        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nonlocal_fea):
        B, N, C, H, W = nonlocal_fea.size()  

        emb_ref = self.tAtt_2(nonlocal_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(nonlocal_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1) 
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        cor_prob = cor_prob.view(B, -1, H, W)
        nonlocal_fea = nonlocal_fea.view(B, -1, H, W) * cor_prob

        fea = self.lrelu(self.fea_fusion(nonlocal_fea))

        att = self.lrelu(self.sAtt_1(nonlocal_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add

        return fea

class RViDeNet(nn.Module):
    def __init__(self, predenoiser, nf=16, nframes=3, groups=1, front_RBs=5, back_RBs=10, center=1):
        super(RViDeNet, self).__init__()
        self.center = center

        ResidualBlock_noBN_begin = functools.partial(ResidualBlock_noBN, nf=nf)
        ResidualBlock_noBN_end = functools.partial(ResidualBlock_noBN, nf=nf*4)

        self.pre_denoise = predenoiser

        self.conv_first = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_begin, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.align = Alignment(nf=nf, groups=groups)

        self.non_local_attention = Non_Local_Attention(nf=nf, nframes=nframes)

        self.temporal_fusion = Temporal_Fusion(nf=nf, nframes=nframes, center=self.center)

        self.recon_trunk = make_layer(ResidualBlock_noBN_end, back_RBs)
  
        self.cbam = CBAM(nf*4, 16)

        self.conv_last = nn.Conv2d(nf*4, 4, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # faster version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        ### predenoising
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        x = x.permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)
        predenoised_img = predenoised_img.view(B, N, C, H, W).permute(2, 0, 1, 3, 4).view(C*B, N, 1, H, W)

        #### extract noisy features
        #print(x[:, :, 0, :, :].contiguous().shape)
        L1_fea_noisy = self.lrelu(self.conv_first(x[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
        # L2
        L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
        L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
        # L3
        L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
        L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

        L1_fea_noisy = L1_fea_noisy.view(C*B, N, -1, H, W)
        L2_fea_noisy = L2_fea_noisy.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_noisy = L3_fea_noisy.view(C*B, N, -1, H // 4, W // 4)

        #### extract predenoised features
        L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:, :, 0, :, :].contiguous().view(-1, 1, H, W)))
        L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
        # L2
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
        L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
        # L3
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
        L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

        L1_fea_predenoised = L1_fea_predenoised.view(C*B, N, -1, H, W)
        L2_fea_predenoised = L2_fea_predenoised.view(C*B, N, -1, H // 2, W // 2)
        L3_fea_predenoised = L3_fea_predenoised.view(C*B, N, -1, H // 4, W // 4)

        #### align
        # ref feature list
        ref_fea_l_noisy = [
            L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
            L3_fea_noisy[:, self.center, :, :, :].clone()
        ]
        ref_fea_l_predenoised = [
            L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
            L3_fea_predenoised[:, self.center, :, :, :].clone()
        ]
        aligned_noisy_fea = []
        for i in range(N):
            nbr_fea_l_noisy = [
                L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                L3_fea_noisy[:, i, :, :, :].clone()
            ]
            nbr_fea_l_predenoised = [
                L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                L3_fea_predenoised[:, i, :, :, :].clone()
            ]
            
            aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
            aligned_noisy_fea.append(aligned_fea_noisy)

        aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
        
        #non-local attention
        non_local_feature = self.non_local_attention(aligned_noisy_fea)

        #temporal fusion
        fea = self.temporal_fusion(non_local_feature)# fea shape: (C*B, nf, H, W)
        _, nf, _, _ = fea.size()
        fusioned_fea_4channel = fea.view(C, B, nf, H, W).permute(1, 0, 2, 3, 4).view(B, C*nf, H, W)

        #spatial fusion
        out = self.recon_trunk(fusioned_fea_4channel)
        out = self.cbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out

    '''# old version
    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()
        ### predenoising
        predenoised_img = self.pre_denoise(x.view(-1, C, H, W))

        aligned_noisy_fea_4channel = []
        fusioned_fea_4channel = []

        for channel_index in range(C):

            #### extract noisy features
            L1_fea_noisy = self.lrelu(self.conv_first(x[:,:,channel_index,:,:].view(-1, 1, H, W)))
            L1_fea_noisy = self.feature_extraction(L1_fea_noisy)
            # L2
            L2_fea_noisy = self.lrelu(self.fea_L2_conv1(L1_fea_noisy))
            L2_fea_noisy = self.lrelu(self.fea_L2_conv2(L2_fea_noisy))
            # L3
            L3_fea_noisy = self.lrelu(self.fea_L3_conv1(L2_fea_noisy))
            L3_fea_noisy = self.lrelu(self.fea_L3_conv2(L3_fea_noisy))

            L1_fea_noisy = L1_fea_noisy.view(B, N, -1, H, W)
            L2_fea_noisy = L2_fea_noisy.view(B, N, -1, H // 2, W // 2)
            L3_fea_noisy = L3_fea_noisy.view(B, N, -1, H // 4, W // 4)

            #### extract predenoised features
            L1_fea_predenoised = self.lrelu(self.conv_first(predenoised_img[:,channel_index,:,:].view(-1, 1, H, W)))
            L1_fea_predenoised = self.feature_extraction(L1_fea_predenoised)
            # L2
            L2_fea_predenoised = self.lrelu(self.fea_L2_conv1(L1_fea_predenoised))
            L2_fea_predenoised = self.lrelu(self.fea_L2_conv2(L2_fea_predenoised))
            # L3
            L3_fea_predenoised = self.lrelu(self.fea_L3_conv1(L2_fea_predenoised))
            L3_fea_predenoised = self.lrelu(self.fea_L3_conv2(L3_fea_predenoised))

            L1_fea_predenoised = L1_fea_predenoised.view(B, N, -1, H, W)
            L2_fea_predenoised = L2_fea_predenoised.view(B, N, -1, H // 2, W // 2)
            L3_fea_predenoised = L3_fea_predenoised.view(B, N, -1, H // 4, W // 4)

            #### align
            # ref feature list
            ref_fea_l_noisy = [
                L1_fea_noisy[:, self.center, :, :, :].clone(), L2_fea_noisy[:, self.center, :, :, :].clone(),
                L3_fea_noisy[:, self.center, :, :, :].clone()
            ]
            ref_fea_l_predenoised = [
                L1_fea_predenoised[:, self.center, :, :, :].clone(), L2_fea_predenoised[:, self.center, :, :, :].clone(),
                L3_fea_predenoised[:, self.center, :, :, :].clone()
            ]
            aligned_noisy_fea = []
            for i in range(N):
                nbr_fea_l_noisy = [
                    L1_fea_noisy[:, i, :, :, :].clone(), L2_fea_noisy[:, i, :, :, :].clone(),
                    L3_fea_noisy[:, i, :, :, :].clone()
                ]
                nbr_fea_l_predenoised = [
                    L1_fea_predenoised[:, i, :, :, :].clone(), L2_fea_predenoised[:, i, :, :, :].clone(),
                    L3_fea_predenoised[:, i, :, :, :].clone()
                ]
                
                aligned_fea_noisy = self.align(nbr_fea_l_noisy, ref_fea_l_noisy, nbr_fea_l_predenoised, ref_fea_l_predenoised)
                aligned_noisy_fea.append(aligned_fea_noisy)

            aligned_noisy_fea = torch.stack(aligned_noisy_fea, dim=1)
            aligned_noisy_fea_4channel.append(aligned_noisy_fea) 
            
            #non-local attention
            non_local_feature = self.non_local_attention(aligned_noisy_fea)

            #temporal fusion
            fea = self.temporal_fusion(non_local_feature)
            fusioned_fea_4channel.append(fea)

        aligned_noisy_fea_4channel = torch.cat(aligned_noisy_fea_4channel, dim=2) 
        fusioned_fea_4channel = torch.cat(fusioned_fea_4channel, dim=1)
        
        #spatial fusion
        out = self.recon_trunk(fusioned_fea_4channel)
        out = self.cbam(out)
        out = self.conv_last(out)
        base = x_center
        out += base

        return out'''

