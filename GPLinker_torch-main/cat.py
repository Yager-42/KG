import json

def load_file(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            D.append({
                "ID":line["ID"],
                "text":line["text"],
                "spo_list":[(spo["h"]["name"],spo["h"]["pos"] ,spo["relation"], spo["t"]["name"] ,spo["t"]["pos"])
                            for spo in line["spo_list"]]
            })
        return D


def cat(files):
    with open("./eval_submit.json", 'w', encoding="utf-8") as wr:
        text_list_len = len(files[0])
        files_len = len(files)
        for i in range(text_list_len):
            if (i == 471):
                print('hi')
            text = files[0][i]['text']
            ID = files[0][i]['ID']
            voting_reslut = set()
            spoes_list = []
            for j in range(files_len):
                temp_set = set()
                for spo in  files[j][i]['spo_list']:
                    temp_set.add((spo[0],spo[1][0],spo[1][1],spo[2],spo[3],spo[4][0],spo[4][1]))
                spoes_list.append(temp_set)
            temp = spoes_list[0]
            for j in range(1,files_len):
                temp = temp.union(spoes_list[j])
            for temp_spo in list(temp):
                voting = 0
                for spoes in spoes_list:
                    if (temp_spo in spoes):
                        voting += 1
                        if (voting >= 3):
                            voting_reslut.add(temp_spo)
                            break
            if(not voting_reslut):
                for temp_spo in list(temp):
                    voting_reslut.add(temp_spo)
            spo_list = []
            for spo in list(voting_reslut):
                spo_list.append(
                    {"h": {"name": spo[0], "pos": [spo[1], spo[2]]}, "t": {"name": spo[4], "pos": [spo[5], spo[6]]},
                     "relation": spo[3]
                     })
            wr.write(json.dumps({"ID": ID, "text": text, "spo_list": spo_list}, ensure_ascii=False))
            wr.write("\n")


files = []
files.append(load_file('pert_large_eval.json'))
files.append(load_file('uer_large_eval.json'))
files.append(load_file('roberta_eval.json'))
files.append(load_file('nezha_base_eval.json'))
files.append(load_file('pert_base_eval.json'))

cat(files)
