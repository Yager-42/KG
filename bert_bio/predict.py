# coding=UTF-8

import torch

from utils import*
from model import *

def extract(label, text):
    i = 0
    res = []
    while i < len(label):
        if label[i] != 'O' and label[i] != 'U':
            prefix, name = label[i].split('-')
            start = end = i
            i += 1
            while i < len(label) and label[i] == 'I-' + name:
                end = i
                i += 1
            res.append([name, text[start-1:end]])
        else:
            i += 1
    return res


if __name__=='__main__':
    text='许多健康问题及药物都可能使人患糖尿病的风险增加。相关药物包括：糖皮质激素、噻嗪类利尿剂、β受体阻滞剂、非典型抗精神病药物及他汀类药物。曾患妊娠期糖尿病的女性患上2型糖尿病的风险较高，而其他和2型糖尿病相关的健康问题还包括肢端肥大症、皮质醇增多症、甲状腺功能亢进症、嗜铬细胞瘤及某些癌症如胰高血糖素瘤。另外，睾酮缺乏与2型糖尿病也有很密切的关联。可能诱发糖尿病的药物包括：糖皮质激素、噻嗪类利尿剂、β受体阻滞剂、非典型抗精神病药物及他汀类药物'
    # token=[list(text)]
    # t=str(token[0])
    # print(t)
    # print(len(text))
    input=tokenizer.encode_plus(text,padding=True,return_tensors='pt',add_special_tokens=True,is_split_into_words=False)
    # print(tokenizer.decode(input['input_ids'][0]))
    # print(input['input_ids'][0])
    # print(tokenizer.convert_tokens_to_ids(list(text)))
    for index, value in input.items():
        input[index] = value.cuda()

    mask=[]
    mask.append([1]*(len(input['input_ids'][0])))
    mask=torch.tensor((mask)).bool().cuda()
    model=torch.load(MODEL_DIR+'model_31.pth')
    y_pre=model.predict(input=input,mask=mask)

    id2label,_=get_label()
    label = [id2label[l] for l in y_pre[0]]

    print(extract(label, text))