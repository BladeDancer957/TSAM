import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE,DATA_DIR
from transformers import RobertaModel, RobertaTokenizer
from networks.eat_layer import EmotionAttentionLayer
from networks.tsam_layer import TSAMLayer

class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()



        self.roberta = RobertaModel.from_pretrained(configs.roberta_cache_path) # 预训练roberta
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(configs.roberta_cache_path)  # 加载roberta分词器
        self.roberta.resize_token_embeddings(len(self.roberta_tokenizer) + 1)  # 添加了一个特殊token

        if configs.use_tuned_roberta: # 用自己微调后的roberta 进行覆盖
            self.roberta.load_state_dict(torch.load(configs.tuned_roberta_path))

        if configs.use_tsam:
            self.tsam = TSAM(configs)

            if configs.use_emo_type != 'none':
                assert configs.feat_dim == configs.emo_emb_dim

                self.EmotionEmbedding = nn.Embedding(configs.emo_num, configs.emo_emb_dim)

                if configs.emo_emb_initialize == 'random':  # 随机初始化
                    nn.init.xavier_uniform_(self.EmotionEmbedding.weight)
                elif configs.emo_emb_initialize == 'pretrain_emotion_word':  # roberta 情感词向量 初始化
                    pretrained_embeddings = np.load(configs.roberta_cache_path + '/pretrained_emotion_word_embeddings.npy')
                    self.EmotionEmbedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.pred = Pre_Predictions(configs) # 情感预测、原因预测模块

        self.configs = configs


    def forward(self, roberta_token_b, roberta_masks_b, roberta_clause_b, window_len_b, adj_b,adj_same_b,adj_diff_b,emo_index_b,y_emotions_b):


        roberta_output = self.roberta(input_ids=roberta_token_b.to(DEVICE),
                                attention_mask=roberta_masks_b.to(DEVICE)) # (batch_size, max_token_len, feat_dim)

        win_utters_h = self.batched_index_select(roberta_output, roberta_clause_b.to(DEVICE)) # (batch_size, curr_max_win_len, feat_dim) 选出每个子句对应的表示 填充部分子句的表示为第一个子句的表示

        roberta_utters_h = win_utters_h # (batch_size, curr_max_win_len, feat_dim)


        if self.configs.use_tsam:
            emotion_embs = None
            emotion_reps = None

            if self.configs.use_emo_type == 'emo_att':
                assert emo_index_b is not None
                emotion_embs = self.EmotionEmbedding(emo_index_b.to(DEVICE)) # (batch_size, emo_num, emo_emb_dim=feat_dim)

            elif self.configs.use_emo_type == 'emo':
                y_emotions_b = torch.LongTensor(y_emotions_b).to(DEVICE)
                y_emotions_b[y_emotions_b==-1] = 0 # -1 填充部分的情感标签  填充部分之后会被mask掉
                emotion_reps = self.EmotionEmbedding(y_emotions_b)

            win_utters_h = self.tsam(win_utters_h, window_len_b, emotion_embs, emotion_reps, adj_b,adj_same_b,adj_diff_b)


            win_utters_h = torch.cat([roberta_utters_h,win_utters_h],dim=-1)

        pred_c = self.pred(win_utters_h)
        return pred_c # （batch_size, curr_max_win_len）

    def batched_index_select(self, roberta_output, roberta_clause_b):
        hidden_state = roberta_output[0]
        dummy = roberta_clause_b.unsqueeze(2).expand(roberta_clause_b.size(0), roberta_clause_b.size(1), hidden_state.size(2))
        win_utters_h = hidden_state.gather(1, dummy)
        return win_utters_h

    def loss_cause(self, pred_c, y_causes, y_mask): # 只计算非填充部分的loss

        '''
        weight=torch.from_numpy(np.array([0.1,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])).float()
        weight = size 为C的 float Tensor 权重为每一类样本所占的比重的倒数
        loss =torch.nn.CrossEntropyLoss(weight=weight)
        a = torch.randn(2,3)

        target = torch.tensor([0,1])

        loss(a,target)

        '''

        y_mask = torch.ByteTensor(y_mask).to(DEVICE)   # 低版本torch 需要用Bytetensor
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)
        pred_c = pred_c.masked_select(y_mask)
        true_c = y_causes.masked_select(y_mask)

        count_pos = torch.sum(true_c)*1.0 + 1e-10  # 二分类必须使用权重
        count_neg = torch.sum(1-true_c)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos/(count_pos+count_neg)

        criterion = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=beta)
        loss_c = criterion(pred_c, true_c)*beta_back

        return loss_c


    def transport(self, pred_c, y_causes, y_mask):
        y_mask = torch.ByteTensor(y_mask).to(DEVICE)
        y_causes = torch.FloatTensor(y_causes).to(DEVICE)

        pred_c = pred_c.masked_select(y_mask) # 取出非填充分布的预测结果
        y_causes = y_causes.masked_select(y_mask) # 取出非填充分布的标签

        pred_c = F.sigmoid(pred_c)
        pred_c = (pred_c>0.5).float()

        return pred_c, y_causes # 一维结果




class TSAM(nn.Module):
    def __init__(self, configs):
        super(TSAM, self).__init__()

        self.layers = configs.tsam_layer_num

        self.tsam_layer_stack = nn.ModuleList()


        for i in range(self.layers):
            self.tsam_layer_stack.append(TSAMLayer(configs))

        if configs.use_gat and configs.use_emo_type!='none':
            if configs.use_interaction:
                self.affine1 = nn.Parameter(torch.Tensor(configs.feat_dim, configs.feat_dim))
                self.affine2 = nn.Parameter(torch.Tensor(configs.feat_dim, configs.feat_dim))
                self.drop = nn.Dropout(configs.emo_dp)
        self.configs = configs


    def forward(self, win_utters_h, window_len_b, emotion_embs, emotion_reps, adj_b,adj_same_b,adj_diff_b):
        batch, curr_max_win_len, _ = win_utters_h.size()

        assert max(window_len_b) == curr_max_win_len

        outputs_gat = win_utters_h
        outputs_emo = win_utters_h

        for i, tsam_layer in enumerate(self.tsam_layer_stack):
            gat_utters_h, emo_utters_h = tsam_layer(outputs_gat,outputs_emo, emotion_embs,emotion_reps, adj_b,adj_same_b,adj_diff_b) # (batch_size, curr_max_win_len, emo_emb_dim=feat_dim)

            if self.configs.use_gat and self.configs.use_emo_type != 'none' and self.configs.use_interaction:
                assert gat_utters_h is not None
                assert emo_utters_h is not None

                A1 = F.softmax(torch.bmm(torch.matmul(gat_utters_h, self.affine1), torch.transpose(emo_utters_h, 1, 2)), dim=-1)
                A2 = F.softmax(torch.bmm(torch.matmul(emo_utters_h, self.affine2), torch.transpose(gat_utters_h, 1, 2)), dim=-1)

                gat_utters_h1 = torch.bmm(A1, emo_utters_h)
                emo_utters_h1 = torch.bmm(A2, gat_utters_h)


                outputs_gat = self.drop(gat_utters_h1) if i < self.layers - 1 else gat_utters_h1
                outputs_emo = self.drop(emo_utters_h1) if i < self.layers - 1 else emo_utters_h1
            else:
                outputs_gat = gat_utters_h
                outputs_emo = emo_utters_h

        if self.configs.use_gat and self.configs.use_emo_type != 'none':
            assert outputs_gat is not None
            assert outputs_emo is not None
            outputs = torch.cat([outputs_gat,outputs_emo],dim=-1)
        elif self.configs.use_gat == False and self.configs.use_emo_type != 'none':
            assert outputs_gat is None
            assert outputs_emo is not None
            outputs = outputs_emo
        elif self.configs.use_gat and self.configs.use_emo_type == 'none':
            assert outputs_gat is not None
            assert outputs_emo is None
            outputs = outputs_gat

        return outputs




class Pre_Predictions(nn.Module):
    def __init__(self, configs):
        super(Pre_Predictions, self).__init__()
        self.feat_dim = configs.feat_dim

        if configs.use_tsam:
            if configs.use_gat and configs.use_emo_type!='none':
                self.feat_dim *= 3
            elif configs.use_gat==False and configs.use_emo_type!='none':
                self.feat_dim *= 2
            elif configs.use_gat and configs.use_emo_type=='none':
                self.feat_dim *= 2

        self.lin1 = nn.Linear(self.feat_dim, self.feat_dim//2)
        self.drop = nn.Dropout(configs.dp)
        self.lin2 = nn.Linear(self.feat_dim//2, 1)

    def forward(self, win_utters_h):
        hidden = self.drop(F.relu(self.lin1(win_utters_h)))
        pred_c = self.lin2(hidden)
        return pred_c.squeeze(2) # （batch_size, curr_max_win_len）
