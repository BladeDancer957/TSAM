import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionAttentionLayer(nn.Module):

    def __init__(self, num_units, num_heads=1, dropout_rate=0):
        '''Applies multihead attention.
        Args:
            num_units: A scalar. Attention size.
            dropout_rate: A floating point number.
            num_heads: An int. Number of heads.
        '''
        super(EmotionAttentionLayer, self).__init__()

        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.Q_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(self.num_units, self.num_units), nn.ReLU())


        self.output_dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, queries, keys, values,last_layer = False):
        # keys, values: same shape of [bs, emo_num, feat_dim]
        # queries: A 3d Variable with shape of [bs,curr_max_win_len, feat_dim]
        # Linear projections
        Q = self.Q_proj(queries)  # [bs,curr_max_win_len, feat_dim]
        K = self.K_proj(keys)  # [bs, emo_num, feat_dim]
        V = self.V_proj(values)  # [bs, emo_num, feat_dim]
        # Split and concat
        Q_ = torch.cat(torch.chunk(Q, self.num_heads, dim=2), dim=0)  # (h*bs, curr_max_win_len, feat_dim/h)
        K_ = torch.cat(torch.chunk(K, self.num_heads, dim=2), dim=0)  # (h*bs, emo_num, feat_dim/h)
        V_ = torch.cat(torch.chunk(V, self.num_heads, dim=2), dim=0)  # (h*bs, emo_num, feat_dim/h)
        # Multiplication
        outputs = torch.bmm(Q_, K_.permute(0, 2, 1))  # (h*bs, curr_max_win_len, emo_num)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5) # (h*bs, curr_max_win_len, emo_num)

        # Activation
        if last_layer == False:
            outputs = F.softmax(outputs, dim=-1)  #(h*bs, curr_max_win_len, emo_num)
        '''
        # Query Masking  图注意力部分、输出部分都会排除掉填充部分， 这里填充部分正常处理即可
        query_masks = torch.sign(torch.abs(torch.sum(queries, dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = torch.unsqueeze(query_masks, 2).repeat(1, 1, keys.size()[1])  # (h*N, T_q, T_k)
        outputs = outputs * query_masks
        '''
        # Dropouts 中间层的emotion attention添加dropout  输出层做预测时不加
        outputs = self.output_dropout(outputs)  # (h*bs, curr_max_win_len, emo_num)
        if last_layer == True: # head=1 直接返回置信度  此时dropout=0
            return outputs #  (bs, curr_max_win_len, emo_num)
        # Weighted sum
        outputs = torch.bmm(outputs, V_)  # (h*bs, curr_max_win_len, feat_dim/h)
        # Restore shape
        outputs = torch.cat(torch.chunk(outputs, self.num_heads, dim=0), dim=2)  # (bs, curr_max_win_len, feat_dim)
        # Residual connection
        outputs += queries # (bs, curr_max_win_len, feat_dim)

        return outputs # (bs, curr_max_win_len, feat_dim)
