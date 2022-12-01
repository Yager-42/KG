# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: token-pair范式的实体关系抽取pytorch实现
"""
import torch
import json
import sys
import numpy as np
import torch.nn as nn
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel
import configparser

con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf8')
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
# encoder = BertModel.from_pretrained(args_path["model_path"])

with open(args_path["schema_data"], 'r', encoding='utf-8') as f:
    schema = {}
    for idx, item in enumerate(f):
        item = json.loads(item.rstrip())
        schema[item["predicate"]] = idx
id2schema = {}
for k,v in schema.items(): id2schema[v]=k

device = torch.device("cuda:0")
# mention_detect = RawGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)#实体关系抽取任务默认不提取实体类型
# s_o_head = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
# s_o_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
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

# net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
# net.load_state_dict(torch.load('erenet.pth'))
# net.eval()
bagging_size=1
bagging = []
for i in range(bagging_size):
    encoder = BertModel.from_pretrained(args_path["model_path"])
    mention_detect = RawGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)  # 实体关系抽取任务默认不提取实体类型
    s_o_head = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False,
                                tril_mask=False).to(device)
    s_o_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False,
                                tril_mask=False).to(device)
    net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
    net.load_state_dict(torch.load(f'erenet.pth'))
    net.eval()
    bagging.append(net)
    bagging[i].eval()


with open(args_path["val_file"],encoding="utf-8") as f, open("./submit.json", 'w', encoding="utf-8") as wr:
    ID_list=[]
    text_list = []
    for text in f.readlines():
        ID_list.append(json.loads(text.rstrip())["ID"])
        text_list.append(json.loads(text.rstrip())["text"])
    for i in range(len(text_list)):
        text=text_list[i]
        ID=ID_list[i]
        token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=512)["offset_mapping"]
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
        encoder_txt = tokenizer.encode_plus(text, max_length=512)
        input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).to(device)
        token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).to(device)
        #scores = net(input_ids, attention_mask, token_type_ids)
        # scores_list = []
        # for i in range(bagging_size):
        #     scores = bagging[i](input_ids, attention_mask, token_type_ids)
        #     scores_list.append(scores)
        # scores = []
        # for i in range(3):
        #     temp=0
        #     for j in range(bagging_size):
        #         temp+=scores_list[j][i]
        #     scores.append(temp/bagging_size)
        voting_reslut = set()
        spoes_list = []
        for j in range(bagging_size):
            net = bagging[j]
            scores = net(input_ids, attention_mask, token_type_ids)
            outputs = [o[0].data.cpu().numpy() for o in scores]
            subjects, objects = set(), set()
            outputs[0][:, [0, -1]] -= np.inf
            outputs[0][:, :, [0, -1]] -= np.inf
            for l, h, t in zip(*np.where(outputs[0] > 0)):
                if l == 0:
                    subjects.add((h, t))
                else:
                    objects.add((h, t))
            spoes = set()
            for sh, st in subjects:
                for oh, ot in objects:
                    p1s = np.where(outputs[1][:, sh, oh] > threshold)[0]
                    p2s = np.where(outputs[2][:, st, ot] > threshold)[0]
                    ps = set(p1s) & set(p2s)
                    for p in ps:
                        spoes.add((
                            text[new_span[sh][0]:new_span[st][-1] + 1], new_span[sh][0],new_span[st][-1] + 1,
                            id2schema[p],
                            text[new_span[oh][0]:new_span[ot][-1] + 1], new_span[oh][0],new_span[ot][-1] + 1
                        ))
            spoes_list.append(spoes)
        temp= spoes_list[0]
        for k in range(1,len(spoes_list)):
            temp = temp.union(spoes_list[k])
        for temp_spo in list(temp):
            voting = 0
            for spoes in spoes_list:
                if(temp_spo in spoes):
                    voting+=1
                    if(voting>=bagging_size/2):
                        voting_reslut.add(temp_spo)
                        break


        spo_list = []
        for spo in list(voting_reslut):
            spo_list.append({"h":{"name":spo[0],"pos":[spo[1],spo[2]]}, "t":{"name":spo[4],"pos":[spo[5],spo[6]]},"relation":spo[3]
                             })
        wr.write(json.dumps({"ID":ID, "text":text, "spo_list":spo_list}, ensure_ascii=False))
        wr.write("\n")