import torch
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 10
path = os.path.abspath(__file__)
PATH = '/'.join(path.split('/')[:-2])

DATA_DIR = PATH + '/data'
TRAIN_FILE = 'window_all/train.json' 
VALID_FILE = 'window_all/valid.json'
TEST_FILE  = 'window_all/test.json' 
STOPWORDS_FILE = 'stopwords.json'

save_ckpt = True
ckpt_path = PATH + '/code/ckpt/'
ckpt_name_full = 'ex1_1_full.pt'
ckpt_name_roberta = 'ex1_1_roberta.pt'

class Config(object):
    def __init__(self):

        self.roberta_cache_path = PATH + '/code/roberta-large'
        if 'base' in self.roberta_cache_path:
            self.feat_dim = 768
        elif 'large' in self.roberta_cache_path:
            self.feat_dim = 1024

        self.tune_roberta = 'funetuning'  # 'funetuning' or 'fixed'

        self.use_tuned_roberta = False

        if self.use_tuned_roberta:
            self.tuned_roberta_path = ckpt_path + ckpt_name_roberta


        self.use_tsam = True

        if self.use_tsam:
            self.tsam_layer_num = 3

            self.use_gat = True  # 是否使用gat模块  True or False
            self.use_emo_type = 'emo_att'  # 'none'不使用情感信息， 'emo'拼接情感embedding， 'emo_att'情感注意力

            if self.use_gat:
                self.graph_att_heads = 8 # 头数
                self.speaker_relation = True # 是否考虑说话人关系

            if self.use_emo_type != 'none':
                self.emo_emb_dim = self.feat_dim
                self.emo_num = 7
                self.emo_emb_initialize = 'random'
                self.emo_dp = 0.1

                if self.use_emo_type=='emo_att':
                    self.emo_att_heads = 4


            if self.use_gat and self.use_emo_type!='none':
                self.use_interaction = True   # 是否使用交互模块


        self.epochs = 40
        self.lr = 1e-5
        self.batch_size = 2
        self.gradient_accumulation_steps = 2
        self.dp = 0.4
        self.l2 = 1e-5
        self.l2_roberta = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8



