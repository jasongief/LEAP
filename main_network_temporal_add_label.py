import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets.MPAM import MultimodalPyramidAttentionalModule as MMP
from nets.Transformer import *



class AVLTransformer(nn.Module):
    # the proposed LEAP decoder
    def __init__(self, num_inputs, ffn_dim, lv_layer_num=2, la_layer_num=2, dropout=0.2, nhead=8):
        super(AVLTransformer, self).__init__()
        c = copy.deepcopy
        self.multiheadattn = VanillaMultiHeadAttention(nhead, num_inputs)
        self.feedforward = PositionwiseFeedForward(num_inputs, ffn_dim, dropout=dropout)

        lv_cma_layers = []
        la_cma_layers = []
        for i in range(lv_layer_num):
            lv_cma_layers += [VanillaTransformerLayer(num_inputs, c(self.multiheadattn), c(self.feedforward), dropout)]
            la_cma_layers += [VanillaTransformerLayer(num_inputs, c(self.multiheadattn), c(self.feedforward), dropout)]
        self.lv_cma_network = nn.Sequential(*lv_cma_layers)
        self.la_cma_network = nn.Sequential(*la_cma_layers)

    def forward(self, v, a, label):
        # v: [nbatch, 10, 512] a: [nbatch, 10, 512] label: [nbatch, 25, 512]
        f_lv_stage_list = []
        f_la_stage_list = []
        attn_lv_stage_list = []
        attn_la_stage_list = []
        lv_stage = label
        la_stage = label

        for i in range(len(self.lv_cma_network)):
            f_lv, lv_attn = self.lv_cma_network[i](lv_stage, v, v) # [b, 25, 512,]
            f_la, la_attn = self.la_cma_network[i](la_stage, a, a) # [b, 25, 512,]
            lv_stage = f_lv  # [nbatch, 10, 512]
            la_stage = f_la  # [nbatch, 10, 512]
            f_lv_stage_list.append(lv_stage)
            f_la_stage_list.append(la_stage)
            attn_lv_stage_list.append(lv_attn)
            attn_la_stage_list.append(la_attn)
        return f_lv_stage_list, attn_lv_stage_list, f_la_stage_list, attn_la_stage_list



class AVVPNet(nn.Module):
    # whole backbone for AVVP task: encoder (MM-Pyr) + decoder (LEAP) 
    def __init__(self, mmp_head, avl_head, hidden_size, ffn_dim, n_channels, label_dim, lv_layer_num, la_layer_num):
        super(AVVPNet, self).__init__()

        self.fc_a = nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.MMP = MMP(hidden_size, ffn_dim, n_channels, nhead=mmp_head)

        #! for label-semantic based projection (audio/visual - text cross attention)
        self.fc_l = nn.Linear(label_dim, 512)
        self.AVL = AVLTransformer(hidden_size, ffn_dim, lv_layer_num, la_layer_num, nhead=avl_head)
        self.fc_la = nn.Linear(512, 1)
        self.fc_lv = nn.Linear(512, 1)


    def forward(self, audio, visual, visual_st, label):
        # label_a/label_v: [bs, 25, dim]
        f_l = self.fc_l(label)
        f_a = self.fc_a(audio) # (b, 10, 512)

        # 2d and 3d visual feature fusion (b, 80, 2048), (b, 10, 512)
        # merge (b, 80, 2048) -> (b, 10, 512)
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(visual_st)
        f_v = torch.cat((vid_s, vid_st), dim=-1)
        f_v = self.fc_fusion(f_v) # (b, 10, 512)

        # MM-Pyramid encoder
        v_o, a_o = self.MMP(f_v, f_a) # (b, 10, 512)

        # LEAP decoder
        lv_f_list, lv_attn_list, la_f_list, la_attn_list = self.AVL(v_o, a_o, f_l)

        if len(lv_f_list) > 1: 
            #! or from the last layer
            lv_f = lv_f_list[-1] # [bs, 25, 512]
            la_f = la_f_list[-1] # [bs, 25, 512]
            lv_attn = lv_attn_list[-1] # [bs, nhead, 25, 10]
            la_attn = la_attn_list[-1] 
            #! or from the first layer
            # lv_f = lv_f_list[0] # [bs, 25, 512]
            # la_f = la_f_list[0] # [bs, 25, 512]
            # lv_attn = lv_attn_list[0] # [bs, nhead, 25, 10]
            # la_attn = la_attn_list[0] 
            #! average two layers
            # lv_f = torch.stack(lv_f_list, dim=2).mean(dim=2) # [bs, 25, list_num, 512] -> [bs, 25, 512]
            # la_f = torch.stack(la_f_list, dim=2).mean(dim=2) # [bs, 25, list_num, 512] -> [bs, 25, 512]
            # lv_attn = torch.stack(lv_attn_list, dim=1).mean(dim=1) #  [bs, list_num, nhead, 25, 10] -> [bs, nhead, 25, 10]
            # la_attn = torch.stack(la_attn_list, dim=1).mean(dim=1) # [bs, list_num, nhead, 25, 10] -> [bs, nhead, 25, 10]
            # pdb.set_trace()
        else:
            lv_f = lv_f_list[0] # [bs, 25, 512]
            la_f = la_f_list[0] # [bs, 25, 512]
            lv_attn = lv_attn_list[-1] # [bs, nhead, 25, 10]
            la_attn = la_attn_list[-1] # [bs, nhead, 25, 10]
        #! note that attn is obtained using Softmax not sigmoid
        lv_attn_tmp = torch.mean(lv_attn, dim=1) # [bs, nhead, 25, 10] -> [bs, 25, 10]
        lv_attn_tmp = lv_attn_tmp.permute(0, 2, 1).contiguous()  # [bs, 10, 25]
        la_attn_tmp = torch.mean(la_attn, dim=1) # [bs, nhead, 25, 10] -> [bs, 25, 10]
        la_attn_tmp = la_attn_tmp.permute(0, 2, 1).contiguous()  # [bs, 10, 25]

        lv_f_tmp = self.fc_lv(lv_f).squeeze(-1) # [bs, 25, 1] -> [bs, 25]
        la_f_tmp = self.fc_la(la_f).squeeze(-1) # [bs, 25, 1] -> [bs, 25]
        v_prob = torch.sigmoid(lv_f_tmp) # [bs, 25]
        a_prob = torch.sigmoid(la_f_tmp) # [bs, 25]
        
        a_prob_tmp = (a_prob >= 0.5).float()
        v_prob_tmp = (v_prob >= 0.5).float()
        global_prob = a_prob_tmp + v_prob_tmp - a_prob_tmp * v_prob_tmp

        v_frame_prob = torch.sigmoid(lv_attn_tmp) # [bs, 10, 25]
        a_frame_prob = torch.sigmoid(la_attn_tmp) # [bs, 10, 25]
        return global_prob, a_prob, v_prob, a_frame_prob, v_frame_prob, a_o, v_o, la_f, lv_f 



if __name__ == "__main__":
    avl = AVLTransformer(num_inputs=512, ffn_dim=512, lv_layer_num=2, la_layer_num=2)
    b = 4
    a = torch.randn(b, 10, 512)
    v = torch.randn(b, 10, 512)
    l = torch.randn(b, 25, 512)

    f_lv_list, attn_lv_list, f_la_list, attn_la_list = avl(a, v, l)
    pdb.set_trace()
