import torch
import torch.nn as nn
from networks.gat_layer import GraphAttentionLayer
from networks.eat_layer import EmotionAttentionLayer
import torch.nn.functional as F
from config import DEVICE

class TSAMLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, configs):
        super(TSAMLayer, self).__init__()


        if configs.use_gat:
            assert  configs.feat_dim % configs.graph_att_heads == 0

            if configs.speaker_relation:
                self.gat_layer_same = GraphAttentionLayer(configs, configs.graph_att_heads, configs.feat_dim,
                                                          configs.feat_dim // configs.graph_att_heads, configs.dp)
                self.gat_layer_diff = GraphAttentionLayer(configs, configs.graph_att_heads, configs.feat_dim,
                                                          configs.feat_dim // configs.graph_att_heads, configs.dp)
            else:
                self.gat_layer = GraphAttentionLayer(configs, configs.graph_att_heads, configs.feat_dim,
                                                     configs.feat_dim // configs.graph_att_heads, configs.dp)
            self.speaker_relation = configs.speaker_relation

        if configs.use_emo_type == 'emo_att':
            self.eat_layer = EmotionAttentionLayer(configs.feat_dim, configs.emo_att_heads, configs.emo_dp)


        self.configs = configs


    def forward(self,outputs_gat,outputs_emo,emotion_embs, emotion_reps, adj_b,adj_same_b,adj_diff_b):

        gat_utters_h = None
        emo_utters_h = None

        if self.configs.use_gat:
            if self.speaker_relation:
                gat_utters_h_same = self.gat_layer_same(outputs_gat, adj_same_b)
                gat_utters_h_diff = self.gat_layer_diff(outputs_gat, adj_diff_b)
                gat_utters_h = gat_utters_h_same + gat_utters_h_diff
            else:
                gat_utters_h = self.gat_layer(outputs_gat, adj_b)  # (bs, curr_max_win_len,feat_dim)

        if self.configs.use_emo_type == 'emo':
            assert emotion_reps is not None #(batch_size, curr_max_win_len, emo_emb_dim=feat_dim)
            emo_utters_h = emotion_reps

        elif self.configs.use_emo_type == 'emo_att':
            assert emotion_embs is not None
            emo_utters_h = self.eat_layer(outputs_emo, emotion_embs,
                                         emotion_embs)  # (batch_size, curr_max_win_len, emo_emb_dim=feat_dim)


        return gat_utters_h, emo_utters_h