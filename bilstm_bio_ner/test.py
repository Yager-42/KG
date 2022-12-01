from BiLstm_Crf_Model import *
from config import *
from utils import *

if __name__ == '__main__':
    dataset = DataSet('test')
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)

    with torch.no_grad():
        model = torch.load(MODEL_DIR + 'model_5.pth')
        y_true_list = []
        y_pred_list = []

        for b, (input, target, mask) in enumerate(loader):
            y_pred = model(input, mask)
            loss = model.loss_fn(input, target, mask)

            for lst in y_pred:
                y_pred_list += lst
            for y,m in zip(target, mask):
                y_true_list += y[m==True].tolist()

            print('>> batch:', b, 'loss:', loss.item())

        y_true_tensor = torch.tensor(y_true_list)
        y_pred_tensor = torch.tensor(y_pred_list)
        accuracy = (y_true_tensor == y_pred_tensor).sum()/len(y_true_tensor)
        print('>> total:', len(y_true_tensor), 'accuracy:', accuracy.item())
