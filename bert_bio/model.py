from transformers import AutoModel
import torch
import torch.nn as nn
from torchcrf import  CRF

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert=AutoModel.from_pretrained('bert-base-multilingual-cased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.GRU=nn.GRU(768,256,batch_first=True,bidirectional=True)
        self.linear=nn.Linear(256*2,32)#双向所以是两倍
        self.crf=CRF(num_tags=32,batch_first=True)


    def reshape_remove_pad(self,out,target,attention_mask):
        #移除填充pad来降低pad对loss的影响
        out=out.reshape(-1,32)
        target=target.reshape(-1)

        select=attention_mask.reshape(-1)==1

        out=out[select]
        target=target[select]

        return out.softmax(dim=1),target

    def get_crf_in(self,input):
        #input为tokenizer编码结果
        with torch.no_grad():
            out=self.bert(**input).last_hidden_state
        out,_=self.GRU(out)
        crf_in=self.linear(out)
        return crf_in


    def predict(self,input,mask):
        crf_in=self.get_crf_in(input)
        crf_out=self.crf.decode(emissions=crf_in,mask=mask)
        return crf_out

    def forward(self,input,target,mask):
        #input为tokenizer编码结果
        crf_in=self.get_crf_in(input)
        #loss_in是用于计算crf前的模型损失
        loss_in=self.reshape_remove_pad(crf_in,target,input['attention_mask'])
        crf_out=self.crf.decode(emissions=crf_in,mask=mask)
        loss=self.loss_func(loss_in,crf_in,target,mask)
        return crf_out,loss

    def loss_func(self,loss_in,crf_in,target,mask):
        #设置两个loss，一个是Bert和双向GRU的交叉熵loss,一个是crf的loss，这里只做简单相加
        CEloss=torch.nn.CrossEntropyLoss()
        loss_before_crf=CEloss(loss_in[0],loss_in[1]) #loss_in[0]需要softmax？
        loss_crf=-self.crf(crf_in,target,mask,reduction='mean')
        loss_tot=loss_crf+loss_before_crf
        return loss_tot

if __name__=='__main__':
    model=model()
    print(model)