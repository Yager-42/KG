from torch.utils.data import DataLoader

from model import*
from utils import*
from torch import nn as nn
import torch

if __name__=='__main__':
     lr=5e-4
     epoches=100

     model=model().cuda()
     dataset=DataSet()
     loader=DataLoader(dataset=dataset,batch_size=64,collate_fn=collate_fn,shuffle=True)
     optim=torch.optim.Adam(model.parameters(),lr=lr)

     for epoch in range(epoches):
         for i,(inputs,targets,masks) in enumerate(loader):
              optim.zero_grad()
              output,loss=model(inputs,targets,masks)

              loss.backward()
              optim.step()

              if i%10==0:
                   print('>>epoch ',epoch,' loss:',loss.item())
         torch.save(model, MODEL_DIR + f'model_{epoch}.pth')




