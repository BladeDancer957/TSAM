import sys
sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
from config import *
from utils.utils import *
import random

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

config1 = Config()


def build_train_data(configs, shuffle=True): # 训练数据默认打乱
    train_dataset = MyDataset(configs, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, data_type): # 测试数据不打乱
    dataset = MyDataset(configs, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir #数据集路径
        self.data_type = data_type # train valid test

        #训练集、验证集和测试集
        self.train_file = join(data_dir, TRAIN_FILE)
        self.valid_file = join(data_dir, VALID_FILE)
        self.test_file = join(data_dir, TEST_FILE )
        self.stopwords_file = join(data_dir,STOPWORDS_FILE) # 停止词

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs


        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(configs.roberta_cache_path) # 加载roberta分词器
        special_tokens_dict = {'additional_special_tokens': ["[CLS]"]} #添加特殊token[CLS]
        self.roberta_tokenizer.add_special_tokens(special_tokens_dict)
        print("additional_special_tokens: ",self.roberta_tokenizer.additional_special_tokens)
        self.CLS_id =  self.roberta_tokenizer.additional_special_tokens_ids
        print("CLS id: ",self.CLS_id)


        self.emotion2index = {'neutral':0,'happy':1,'surprise':2,'anger':3,'sad':4,'disgust':5,'fear':6}

        self.y_causes_list, self.y_emotions_list, \
        self.window_len_list, self.window_id_list, \
        self.roberta_token_idx_list, self.roberta_clause_idx_list, \
        self.roberta_token_lens_list = self.read_data_file(self.data_type)




    def __len__(self):
        return len(self.y_causes_list)  # 窗口数量

    def __getitem__(self, idx):
        y_causes =  self.y_causes_list[idx]
        y_emotions = self.y_emotions_list[idx]

        window_len, window_id = self.window_len_list[idx], self.window_id_list[idx]
        roberta_token_idx, roberta_clause_idx = self.roberta_token_idx_list[idx], self.roberta_clause_idx_list[idx]
        roberta_token_lens = self.roberta_token_lens_list[idx]


        roberta_token_idx = torch.LongTensor(roberta_token_idx)
        roberta_clause_idx = torch.LongTensor(roberta_clause_idx)

        assert window_len == len(y_causes)
        assert window_len == len(y_emotions)

        return  y_causes,y_emotions, window_len, window_id, \
               roberta_token_idx, roberta_clause_idx, roberta_token_lens

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        window_id_list = [] # 每个窗口的id
        window_len_list = [] # 数据集中 每个窗口包含utterance的数量
        y_emotions_list, y_causes_list = [], [] # 每个utterance对应的情感 和 原因
        roberta_token_idx_list = []
        roberta_clause_idx_list = []

        roberta_token_lens_list = []

        data_list = read_json(data_file) # 读取json文件
        stopwords = read_json(self.stopwords_file)

        for window in data_list: # 遍历每个窗口（最后一个utterance是 non-neutral target utterance）
            window_id = window['window_id'] # 窗口id
            window_len = window['window_len'] # 窗口包含子句数量 / 文档长度
            cause_id = window['cause_utt_id'] # 原因utterance id
            window_id_list.append(window_id) # 存储每个窗口的id
            window_len_list.append(window_len) # 存储每个窗口的长度

            y_emotions, y_causes = [], []
            utterance_list = window['utterance_list'] # 窗口中的各个utterance [{},{}...]
            window_str = ''
            for i in range(window_len):  # 遍历窗口中的每个utterance
                cause_label = int(i + 1 in cause_id) # 当前utterance对应的原因标签 0/1
                y_causes.append(cause_label) # 窗口中每个utterance的原因标签 0/1 是否是原因utterance （包含原因短语）

                utterance = utterance_list[i] # 第i个utterance
                emotion_category = utterance['emotion_category'].lower()

                if emotion_category=='sadness':  # 少量情感类别标注不规范 或 超过了定义的七个类（标注错误）
                    emotion_category='sad'
                if emotion_category=='surprised':
                    emotion_category='surprise'
                if emotion_category=='happiness' or emotion_category=='happines' or emotion_category=='excited':
                    emotion_category='happy'
                if emotion_category=='angry':
                    emotion_category='anger'

                emotion_label = self.emotion2index[emotion_category] # utterance对应的情感索引
                y_emotions.append(emotion_label) # 窗口中每个utterance对应的情感标签

                utterance_id = utterance['utterance_id'] # 子句的id
                assert int(utterance_id) == i + 1
                window_str += '[CLS]' + utterance['utterance'] # [CLS]utterance_1[CLS]utterance_2...  # 不是每个utterance单独经过roberta 而是把所有utterance拼接后经过roberta，每个[CLS]token对应各个utterance的表示

            token_list = self.roberta_tokenizer.tokenize(window_str.strip())

            if len(token_list) > 512:
                #去除标点符号
                punctuation1 = "!#$%&\'()*+,-./:;=?@\\^_`{|}~"
                import re
                window_str = re.sub(r'[{}]+'.format(punctuation1), ' ', window_str)
                token_list = self.roberta_tokenizer.tokenize(window_str.strip())

                if len(token_list)>512: # 如果去除标点后 长度仍然大于512 则继续去除停止词
                    from nltk import word_tokenize
                    word_list = word_tokenize(window_str.strip())
                    fileter_words = [word for word in word_list if word not in stopwords]
                    window_str = ' '.join(fileter_words)
                    window_str = re.sub("\[ CLS \]","[CLS]",window_str)
                    token_list = self.roberta_tokenizer.tokenize(window_str.strip())

                assert len(token_list)<=512

            indexed_tokens = self.roberta_tokenizer.encode(window_str.strip(), add_special_tokens=False) # 把每个token转换为id


            utterance_indices = [i for i, x in enumerate(indexed_tokens) if x == self.CLS_id[0]] # 每个utterance CLS的位置 CLS的表示代表对应utterance的表示
            window_token_len = len(indexed_tokens) # 窗口长度 / 包含的utterance数量



            assert len(utterance_indices) == window_len

            roberta_token_idx_list.append(indexed_tokens) # 存储每个窗口对应的token id [[token_id],[]]
            roberta_clause_idx_list.append(utterance_indices) # 存储每个窗口中各个utterance对应的位置 [[0,30,...],[]]
            roberta_token_lens_list.append(window_token_len) # 存储每个窗口的token数量

            y_causes_list.append(y_causes) # 存储每个窗口对应的原因标签 [[1,0,...],[]]
            y_emotions_list.append(y_emotions) # 存储每个窗口对应的情感标签 [[0,5,6,...],[]]

        return y_causes_list, y_emotions_list,window_len_list, window_id_list, \
               roberta_token_idx_list, roberta_clause_idx_list, roberta_token_lens_list



def bert_batch_preprocessing(batch):
    y_causes_b, y_emotions_b, window_len_b, window_id_b, \
    roberta_token_b, roberta_clause_b, roberta_token_lens_b = zip(*batch)

    '''
    window_len_b： 每个窗口包含utterance的数量 [x,x,...]
    y_emotions_b: 每个窗口的各个utterance对应的情感标签 [[0,5,...],[]]
    y_causes_b: 每个窗口的各个utterance对应的原因标签 [[1,0,...],[]]
    '''
    y_mask_b, y_causes_b,y_emotions_b = pad_docs(window_len_b, y_causes_b,y_emotions_b) # [batch_size, curr_max_win_len]


    adj_b,adj_same_b,adj_diff_b = pad_matrices(window_len_b) # （batch_size, curr_max_win_len * curr_max_win_len）


    roberta_token_b = pad_sequence(roberta_token_b, batch_first=True, padding_value=1)  # (batch_size, max_token_len)  填充pad 索引1
    roberta_clause_b = pad_sequence(roberta_clause_b, batch_first=True, padding_value=0) # (batch_size, curr_max_win_len)  填充部分utterance的表示 为 第一个子句的表示

    bsz, max_len = roberta_token_b.size()
    roberta_masks_b = np.zeros([bsz, max_len], dtype=np.float)

    for index, seq_len in enumerate(roberta_token_lens_b): # seq_len 每个窗口实际长度/实际包含的token数量
        roberta_masks_b[index][:seq_len] = 1  # 非填充部分为1 填充部分为0

    roberta_masks_b = torch.FloatTensor(roberta_masks_b) # (batch_size, max_token_len)
    assert roberta_masks_b.shape == roberta_token_b.shape


    emo_index_b = None

    if config1.use_tsam:
        if config1.use_emo_type=='emo_att':
            emotion_index = torch.range(0, config1.emo_num-1).long()
            emo_index_b = emotion_index.repeat(bsz,1) # (batch_size, emo_size)

    return np.array(window_len_b), np.array(adj_b), np.array(adj_same_b),np.array(adj_diff_b), \
           np.array(y_causes_b), np.array(y_emotions_b), np.array(y_mask_b), window_id_b, \
           roberta_token_b, roberta_masks_b, roberta_clause_b, emo_index_b



def pad_docs(window_len_b, y_causes_b,y_emotions_b):

    max_window_len = max(window_len_b) # 一个batch中 窗口包含的最大utterance数量
    y_mask_b, y_causes_b_, y_emotions_b_ = [], [], []

    for y_causes,y_emotions in zip(y_causes_b,y_emotions_b):
        y_causes_ = pad_list(y_causes, max_window_len, -1) # 原因标签填充到最大的utterance数量  填充部分为-1
        y_emotions_ = pad_list(y_emotions, max_window_len, -1) # 情感标签填充到最大的utterance数量  填充部分为-1

        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_causes_)) #填充部分mask为0 # [1,1,1,...,0]

        y_mask_b.append(y_mask)  #  [[1,1,...0],[]]  (batch_size, curr_max_win_len)

        y_causes_b_.append(y_causes_) # (batch_size, curr_max_win_len)
        y_emotions_b_.append(y_emotions_)

    return y_mask_b, y_causes_b_, y_emotions_b_


def pad_matrices(window_len_b):
    N = max(window_len_b) # batch中 窗口包含的最大utterance数量
    adj_b = []
    adj_same_b = []  # 同一个说话人
    adj_diff_b = []  # 不同说话人
    # 二人对话 数据集 ， 说话人交替进行说话，不存在一个说话人连说几句的情况
    for window_len in window_len_b:
        adj = np.ones((window_len, window_len)) # 未填充部分的文档 彼此之间全连结 （邻接矩阵）
        nrr1 = np.zeros((1, window_len))
        nrr2 = np.zeros((1, window_len))
        adj1 = []
        for i in range(window_len):
            if i % 2 == 0:
                nrr1[0][i] = 1
                adj1.append(nrr1)
            else:
                nrr2[0][i] = 1
                adj1.append(nrr2)
        adj1 = np.concatenate(adj1, axis=0)
        adj2 = adj - adj1


        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32) # 填充部分全为0

        adj1 = sp.coo_matrix(adj1)
        adj1 = sp.coo_matrix((adj1.data, (adj1.row, adj1.col)),
                            shape=(N, N), dtype=np.float32)  # 填充部分全为0

        adj2 = sp.coo_matrix(adj2)
        adj2 = sp.coo_matrix((adj2.data, (adj2.row, adj2.col)),
                             shape=(N, N), dtype=np.float32)  # 填充部分全为0


        adj_b.append(adj.toarray())  #（batch_size, N*N）
        adj_same_b.append(adj1.toarray()) # (batch_size, N*N）
        adj_diff_b.append(adj2.toarray()) #（batch_size, N*N）

    return adj_b,adj_same_b,adj_diff_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad

