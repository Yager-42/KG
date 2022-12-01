import torch
from torch.utils import data
import pandas as pd
from config import *

def get_vocab():
    df=pd.read_csv(VOCAB_PATH,names=['word','id'])
    return list(df['word']),dict(df.values)

def get_label():
    df=pd.read_csv(LABEL_PATH,names=['label','id'])
    return list(df['label']),dict(df.values)


class DataSet(data.Dataset):
    def __init__(self,type='train',base_len=50):
        super().__init__()
        self.base_len=base_len
        sample_path=TRAIN_SAMPLE_PATH if type=='train' else TEST_SAMPLE_PATH
        self.df=pd.read_csv(sample_path,names=['word','label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_cutPoint()

    def get_cutPoint(self):
        self.cutPoints=[0]#记录分割点
        i=0#当前分割点
        while 1:
            if i+self.base_len>=len(self.df):
                self.cutPoints.append(len(self.df))
                break
            if self.df.loc[i+self.base_len,'label']=='O':
                i+=self.base_len
                self.cutPoints.append(i)
            else:
                i+=1
    def __len__(self):
        return len(self.cutPoints)-1

    def __getitem__(self, item):
        df=self.df[self.cutPoints[item]:self.cutPoints[item+1]]
        word_unk_id=self.word2id[WORD_UNK]
        label_o_id=self.label2id['O']
        input=[self.word2id.get(w,word_unk_id) for w in df['word']]
        target=[self.label2id.get(l,label_o_id) for l in df['label']]
        return input,target


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = len(batch[0][0])
    input = []
    target = []
    mask = []
    for item in batch:
        pad_len = max_len - len(item[0])
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        mask.append([1] * len(item[0]) + [0] * pad_len)
    return torch.tensor(input).cuda(), torch.tensor(target).cuda(), torch.tensor(mask).bool().cuda()


if __name__=='__main__':
    dataset=DataSet()
    loader=data.DataLoader(dataset,batch_size=10,collate_fn=collate_fn)


