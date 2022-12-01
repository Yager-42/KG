import torch.optim

from utils import *
from BiLstm_Crf_Model import *
from config import *

if __name__=='__main__':
    dataset=DataSet()
    loader=data.DataLoader(dataset=dataset,batch_size=100,shuffle=True,collate_fn=collate_fn)
    model=model().cuda()
    optim=torch.optim.Adam(model.parameters(),lr=LR)

    for epoch in range(6):
        for counter,(input,target,mask) in enumerate(loader):
            y_pre=model(input,mask)
            loss=model.loss_fn(input,target,mask)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if(counter%10==0):
                print('epoch:',epoch,' loss:',loss.item())

        torch.save(model, MODEL_DIR + f'model_{epoch}.pth')

