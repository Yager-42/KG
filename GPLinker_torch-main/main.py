# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: token-pair范式的实体关系抽取pytorch实现
"""
import random

import torch
import json
import sys
import numpy as np
import torch.nn as nn
from nets.gpNet import EffiGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel
from utils.dataloader import data_generator, load_name, load_outside_name, train_outside_concat, cut
from torch.utils.data import DataLoader
import configparser
from utils.bert_optimization import BertAdam


con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf8')
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"])
k_fold_num=5

with open(args_path["schema_data"], 'r', encoding='utf-8') as f:
    schema = {}
    for idx, item in enumerate(f):
        item = json.loads(item.rstrip())
        schema[item["predicate"]] = idx
id2schema = {}
for k,v in schema.items(): id2schema[v]=k

def k_fold(k,i,data):
    #获得第i+1份的k交叉验证训练集和验证集
    length=len(data)
    fold_size=length//k#每一折的大小
    val_start=i*fold_size
    train_data=[]
    val_data=[]
    if(i==k-1):
        train_data=data[:val_start]
        val_data=data[val_start:length]
    else:
        val_end=(i+1)*fold_size
        train_data=data[:val_start]+data[val_end:]
        val_data=data[val_start:val_end]
    return train_data,val_data

train_content = load_name(args_path["train_file"])
outside_content = load_outside_name(args_path["outside_file"])

train=train_outside_concat(train_content,outside_content)
val_set = train[:200]
train_set = train[200:]
random.shuffle(train_set)

data_set=[]
for i in range(k_fold_num):
    data_set.append(random.choices(train_set,k=2800))


device = torch.device("cuda:0")


class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c
        self.encoder = encoder

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs


def set_optimizer(model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=1e-5,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer


def extract(tokenizer,text_list,model):
    spoes=set()
    for i in range(len(text_list)):
        text = text_list[i]
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=con.getint("para", "maxlen"),truncation=True)["offset_mapping"]
        new_span, entities = [], []
        for i in token2char_span_mapping:
            if i[0] == i[1]:
                new_span.append([])
            else:
                if i[0] + 1 == i[1]:
                    new_span.append([i[0]])
                else:
                    new_span.append([i[0], i[-1] - 1])
        threshold = 0.0
        encoder_txt = tokenizer.encode_plus(text, max_length=con.getint("para", "maxlen"),truncation=True)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        scores = model(input_ids, attention_mask, token_type_ids)
        outputs = [o[0].data.cpu().numpy() for o in scores]
        subjects, objects = set(), set()
        outputs[0][:, [0, -1]] -= np.inf
        outputs[0][:, :, [0, -1]] -= np.inf
        for l, h, t in zip(*np.where(outputs[0] > 0)):
            if l == 0:
                subjects.add((h, t))
            else:
                objects.add((h, t))
        for sh, st in subjects:
            for oh, ot in objects:
                p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                ps = set(p1s) & set(p2s)
                for p in ps:
                    spoes.add((
                        text[new_span[sh][0]:new_span[st][-1] + 1], new_span[sh][0], new_span[st][-1] + 1,
                        id2schema[p],
                        text[new_span[oh][0]:new_span[ot][-1] + 1], new_span[oh][0], new_span[ot][-1] + 1
                    ))
    return spoes

def evaluate(tokenizer,dataloader,model):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for i,batch in enumerate(dataloader):
        text_list=batch[0]
        spo_lists=batch[1]
        spoes = extract(tokenizer,text_list,model)
        target_spoes=set()
        for spo_list in spo_lists:
            for i in range(len(spo_list)):
                target_spoes.add((spo_list[i][0],spo_list[i][1][0],spo_list[i][1][1],
                                  spo_list[i][2],
                                  spo_list[i][3], spo_list[i][4][0],spo_list[i][4][1]
                                  ))
        X += len(spoes & target_spoes)
        Y += len(spoes)
        Z += len(target_spoes)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


for i in range(1):
    encoder = BertModel.from_pretrained(args_path["model_path"])
    mention_detect = EffiGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)  # 实体关系抽取任务默认不提取实体类型
    s_o_head = EffiGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False,tril_mask=False).to(device)
    s_o_tail = EffiGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False,tril_mask=False).to(device)
    net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)

    train_data=data_generator(data_set[i][0],tokenizer, max_len=con.getint("para", "maxlen"), schema=schema)
    train_loader = DataLoader(train_data , batch_size=con.getint("para", "batch_size"), shuffle=True, collate_fn=train_data.collate)

    val_data=data_generator(data_set[i][1],tokenizer, max_len=con.getint("para", "maxlen"), schema=schema)
    val_loader = DataLoader(val_data , batch_size=con.getint("para", "batch_size"), shuffle=True, collate_fn=val_data.collate)

    optimizer = set_optimizer(net, train_steps= (int(len(train_data) / con.getint("para", "batch_size")) + 1) * con.getint("para", "epochs"))
    maxf1=0.0
    for eo in range(con.getint("para", "epochs")):
        net.train()
        for idx, batch in enumerate(train_loader):
            text, spo_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
            batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
                batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
            logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
            loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1,mask_zero=True)
            loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2,mask_zero=True)
            loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3,mask_zero=True)
            loss = sum([loss1, loss2, loss3]) / 3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f]"%(eo, con.getint("para", "epochs"), loss.item()))
        if eo>30 :
            net.eval()
            f1, precision, recall=evaluate(tokenizer,val_loader,net)
            if(f1>maxf1):
                torch.save(net.state_dict(), f'./erenet_{i}th.pth')
                maxf1=f1
                print("max f1 is",maxf1)