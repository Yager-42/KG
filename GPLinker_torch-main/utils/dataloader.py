# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

def load_name(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            D.append({
                "text":line["text"],
                "spo_list":[(spo["h"]["name"],spo["h"]["pos"] ,spo["relation"], spo["t"]["name"] ,spo["t"]["pos"])
                            for spo in line["spo_list"]]
            })
        return D

def load_outside_name(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            D.append({
                "text":line["text"],
                "spo_list":[(line['h']['name'], line['h']['pos'], line['relation'], line['t']['name'], line['t']['pos'])]
            })
        return D

def train_outside_concat(train,outside):
    text_list = []
    for i in range(len(outside)):
        text_list.append(outside[i]['text'])
    for j,line in enumerate(train):
        text=line['text'][0:len(line['text'])-1]
        if(text_list.count(text)>0):
            k = text_list.index(text)
            outside.pop(k)
            text_list.pop(k)

    for i,line_train in enumerate(train):
        for j,line_outside in enumerate(outside):
            if(line_outside['text'] in line_train['text']):
                outside.pop(j)

    return train + outside

def cut(train_data):
    for i,line in enumerate(train_data):
        Min = 256
        if(len(line['text'])>256):
            for s, s_pos, r, o, o_pos in line['spo_list']:
                if(s_pos[1]>256 or o_pos[1]>256):
                    Min = min(s_pos[0],o_pos[0],Min)
            first = line['text'][0:Min]
            second = line['text'][Min:]
            first_spo_list = []
            second_spo_list = []
            for s, s_pos, r, o, o_pos in line['spo_list']:
                if (s_pos[1] > 256 or o_pos[1] > 256):
                    s_pos[0] -= Min
                    s_pos[1] -= Min
                    o_pos[0] -= Min
                    o_pos[1] -= Min
                    second_spo_list.append((s, s_pos, r, o, o_pos))
                else:
                    first_spo_list.append((s, s_pos, r, o, o_pos))
            train_data.pop(i)
            train_data.append({'text':first,'spo_list':first_spo_list})
            train_data.append({'text': second, 'spo_list': second_spo_list})
    return train_data

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        spo_list=item["spo_list"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, s_pos, p, o, o_pos in spo_list:
            p = self.schema[p]
            s_h = encoder_text.char_to_token(s_pos[0])
            o_h = encoder_text.char_to_token(o_pos[0])
            s_t=encoder_text.char_to_token(s_pos[1])
            o_t=encoder_text.char_to_token(o_pos[1])
            s__=self.tokenizer.decode(input_ids[s_h:s_t])
            o__=self.tokenizer.decode(input_ids[o_h:o_t])
            if(s_h!=None and s_t!=None and o_h!=None and o_t!=None):
                spoes.add((s_h, s_t-1, p, o_h, o_t-1))
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return text, spo_list, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        spo_list_list = []
        for item in examples:
            text, spo_list, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)
            spo_list_list.append(spo_list)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return text_list, spo_list_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels


