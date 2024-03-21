import json
import random
import os

def json_loads(path, name):
    file = path + "/" + name
    data = []
    with open(file,'r') as f:
        for line in f:
            a = json.loads(line)
            data.append(a)
    return data


# 读取原始数据
file_names = ["0.json", "1.json", "2.json", "3.json", "4.json"]

for fid, fname in enumerate(file_names):
    train_raw_data = json_loads('data/ace04_raw/train', fname)
    test_data = json_loads('data/ace04_raw/test', fname)

    # 随机打乱训练数据
    random.shuffle(train_raw_data)

    # 划分训练集和验证集
    split = int(0.15 * len(train_raw_data))
    train_data = train_raw_data[split:]
    dev_data = train_raw_data[:split]

    # 写入到 JSON 文件中
    output_lines = []
    def write_to_json(data, file_name):
        with open(file_name, 'w') as f:
            for item in data:
                json_string = json.dumps(item)
                f.write(json_string + "\n")

    outputdir = 'data/ace04/' + str(fid) + '/'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    train_oname = outputdir + 'train.txt'
    dev_oname = outputdir + 'dev.txt'
    test_oname = outputdir + 'test.txt'

    write_to_json(train_data, train_oname)
    write_to_json(dev_data, dev_oname)
    write_to_json(test_data, test_oname)