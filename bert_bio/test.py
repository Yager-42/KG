import torch

from model import*
from utils import *

if __name__=='__main__':
    dataset=DataSet('test')
    loader=data.DataLoader(dataset,batch_size=64,collate_fn=collate_fn,shuffle=True)


    with torch.no_grad():
        model=torch.load(MODEL_DIR+'model_19.pth')

        y_t_list=[]
        y_p_list=[]

        for i,(input,target,mask) in enumerate(loader):
            y_p,loss=model(input,target,mask)

            for label_p in y_p:
                y_p_list+=label_p
            for label_t,m in zip(target,mask):
                y_t_list+=label_t[m==1].tolist()

            print('>>batch: ',i,'loss: ',loss.item())

        y_t_tensor=torch.tensor((y_t_list))
        y_p_tensor=torch.tensor((y_p_list))
        acc=(y_t_tensor==y_p_tensor).sum()/len(y_t_tensor)
        print('accuracy: ',acc)
