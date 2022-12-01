import torch
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader

from config import *
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
#tokenizer.add_tokens(new_tokens=[' ','\r\n'])

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
        df=df.dropna()
        word_unk_id=self.word2id[WORD_UNK]
        label_o_id=self.label2id['O']
        input=[]
        target=[]
        num=df['word'].shape[0]
        #由于tokenizer会自动去除换行符和空格，导致label和token之间无法对齐，因此以下手动去除空格和换行符
        #这些换行符和空格应该是OCR出结果时出现的，因此应该不会对结果有影响
        for index in range(num):
            if df.iloc[index,0]==' ' or df.iloc[index,0] == '\r\n':
                continue
            input.append(df.iloc[index,0])
            target.append(self.label2id.get(df.iloc[index,1],label_o_id))
        # input=[w for w in df['word'] if w!=' ' or '\r\n']
        # target=[self.label2id.get(l,label_o_id) for l in df['label']]
        return input,target


def collate_fn(batch):
    '''batch.sort(key=lambda x: len(x[0]), reverse=True)
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
    '''
    tokens=[token[0] for token in batch]
    targets=[target[1] for target in batch]
    inputs=tokenizer.batch_encode_plus(tokens,padding=True,return_tensors='pt',is_split_into_words=True)

    for index, value in inputs.items():
        inputs[index] = value.cuda()

    max_len=inputs['input_ids'].shape[1]

    mask=[]

    for j in range(len(targets)):
        temp=[1]*len(targets[j])
        mask.append(temp)

    for i in range(len(targets)):
        pad_len=max_len-len(targets[i])-1
        targets[i]=[31]+targets[i]
        mask[i]=[1]+mask[i]
        targets[i]=targets[i]+pad_len*[31]
        mask[i]=mask[i]+[1]+(pad_len-1)*[0]
        # targets[i]=targets[i][:max_len]


    return inputs,torch.tensor(targets).cuda(),torch.tensor(mask).bool().cuda()



# if __name__=='__main__':
#     # dataset=DataSet()
#     # loader=DataLoader(dataset=dataset,batch_size=16,collate_fn=collate_fn,shuffle=True)
#     # for i,(tokens,inputs,targets) in enumerate(loader):
#     #     print(inputs['input_ids'][0])
#     #     print(tokenizer.decode(inputs['input_ids'][0]))
#     #     print(len(tokenizer.convert_tokens_to_ids(tokens[0])))
#     #     print(tokenizer.convert_tokens_to_ids(tokens[0]))
#     #     print(len(tokens[0]))
#     #     print(targets[0],targets[0].shape)
#     zidian=tokenizer.get_vocab()



