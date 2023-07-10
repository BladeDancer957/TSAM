import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")

import numpy as np
import random
import torch
from config import *

from data_loader import *
from networks.reccon import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *
from sklearn.metrics import classification_report,f1_score
import time

def main(configs):
    # 固定随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    #构建训练数据
    train_loader = build_train_data(configs)

    valid_loader = build_inference_data(configs, data_type='valid')
    test_loader = build_inference_data(configs, data_type='test')




    model = Network(configs).to(DEVICE)

    if configs.tune_roberta =='funetuning':

        params = model.parameters()
        params_roberta = model.roberta.parameters()
        params_rest = list(model.pred.parameters())


        if configs.use_tsam:
            params_rest += list(model.tsam.parameters())
            if configs.use_emo_type != 'none':
                params_rest += list(model.EmotionEmbedding.parameters())



        assert sum([param.nelement() for param in params]) == \
               sum([param.nelement() for param in params_roberta]) + sum([param.nelement() for param in params_rest])

        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': configs.l2_roberta, 'eps': configs.adam_epsilon},
            {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'eps': configs.adam_epsilon},
            {'params': params_rest,
             'weight_decay': configs.l2}
        ]

        optimizer = AdamW(params, lr=configs.lr)

        num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
        warmup_steps = int(num_steps_all * configs.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)
    elif configs.tune_roberta =='fixed':

        # 固定Roberta层
        freeze_layers = ['roberta.embeddings', 'roberta.encoder', 'roberta.pooler']
        # for name, param in model.named_parameters():
        #     print(name, param.size(), param.requires_grad)

        for name, param in model.named_parameters():
            for element in freeze_layers:
                if element in name:
                    param.requires_grad = False

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print(name, param.size(), param.requires_grad)


        optimizer = torch.optim.Adam(params=model.parameters(), lr=configs.lr, weight_decay=configs.l2)


    model.zero_grad()
    max_valid_f1 = -1
    correspond_test_f1 = -1

    early_stop_flag = None
    for epoch in range(1, configs.epochs+1):
        print("==========Epoch: {} ==============".format(epoch))
        for train_step, batch in enumerate(train_loader, 1):
            model.train()

            window_len_b, adj_b, adj_same_b, adj_diff_b, \
            y_causes_b, y_emotions_b,y_mask_b, window_id_b, \
            roberta_token_b, roberta_masks_b, roberta_clause_b, emo_index_b = batch


            pred_c = model(roberta_token_b, roberta_masks_b,
                                                              roberta_clause_b, window_len_b, adj_b,adj_same_b,adj_diff_b,emo_index_b,y_emotions_b.copy())


            loss = model.loss_cause(pred_c, y_causes_b, y_mask_b)


            loss = loss / configs.gradient_accumulation_steps

            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                if configs.tune_roberta =='funetuning':
                    scheduler.step()
                model.zero_grad()



        with torch.no_grad(): # 一个epoch验证一次
            model.eval()
            print("******************* Validation ***************")
            valid_f1 = inference_one_epoch(valid_loader, model)
            print("******************** Test ********************")
            test_f1 = inference_one_epoch(test_loader, model)

            if valid_f1 > max_valid_f1:
                early_stop_flag = 1
                max_valid_f1 = valid_f1
                correspond_test_f1 = test_f1
                if save_ckpt:
                    torch.save(model.state_dict(), ckpt_path + ckpt_name_full)
                    torch.save(model.roberta.state_dict(), ckpt_path + ckpt_name_roberta)

                #model.load_state_dict(torch.load(filename + "_critic"))
            else:
                early_stop_flag += 1


            print("correspond_test_f1:{}, max_valid_f1:{}".format(correspond_test_f1,max_valid_f1))

        if epoch > configs.epochs / 2 and early_stop_flag >= 10:
            break


    return correspond_test_f1, max_valid_f1


def inference_one_batch(batch, model):

    window_len_b, adj_b, adj_same_b, adj_diff_b, \
    y_causes_b, y_emotions_b, y_mask_b, window_id_b, \
    roberta_token_b, roberta_masks_b, roberta_clause_b, emo_index_b = batch


    pred_c = model(roberta_token_b, roberta_masks_b,
                                                      roberta_clause_b, window_len_b, adj_b,adj_same_b, adj_diff_b, emo_index_b, y_emotions_b)

    pred_c,label_c = model.transport(pred_c,y_causes_b,y_mask_b)

    return  to_np(pred_c), to_np(label_c)


def inference_one_epoch(batches, model):
    preds = []
    labels = []

    for batch in batches:
        pred_c, label_c = inference_one_batch(batch, model)
        preds.extend(pred_c.tolist())
        labels.extend(label_c.tolist())

    r = str(classification_report(labels, preds, digits=4))
    print(r)
    f1 = f1_score(labels,preds,average='macro')

    return f1


if __name__ == '__main__':

    configs = Config()
  
    correspond_test_f1,max_valid_f1 = main(configs)
    print("+"*30+"Final Result:"+"+"*30)
    print("correspond_test_f1:{}, max_valid_f1:{}".format(correspond_test_f1, max_valid_f1))
