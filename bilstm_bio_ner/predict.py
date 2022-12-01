# coding=UTF-8
from utils import *
from BiLstm_Crf_Model import *
from config import *

def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start:end + 1]])
        else:
            i += 1
    return res


if __name__ == '__main__':
    text = '许多健康问题及药物都可能使人患糖尿病的风险增加。相关药物包括：糖皮质激素、噻嗪类利尿剂、β受体阻滞剂、非典型抗精神病药物及他汀类药物。曾患妊娠期糖尿病的女性患上2型糖尿病的风险较高，而其他和2型糖尿病相关的健康问题还包括肢端肥大症、皮质醇增多症、甲状腺功能亢进症、嗜铬细胞瘤及某些癌症如胰高血糖素瘤。另外，睾酮缺乏与2型糖尿病也有很密切的关联。可能诱发糖尿病的药物包括：糖皮质激素、噻嗪类利尿剂、β受体阻滞剂、非典型抗精神病药物及他汀类药物'
    _, word2id = get_vocab()
    input = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in text]]).cuda()
    mask = torch.tensor([[1] * len(text)]).bool().cuda()

    model = torch.load(MODEL_DIR + 'model_5.pth')
    y_pred = model(input, mask)
    id2label, _ = get_label()
    label = [id2label[l] for l in y_pred[0]]
    # print(text)
    # print(label)
    print(extract(label, text))