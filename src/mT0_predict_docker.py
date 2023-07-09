# author: Mingjie Huang
# create time：2023/5/21 0021 16:58
# -*- coding:UTF-8 -*-
import argparse
import json
import os, sys
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, Adafactor, MT5EncoderModel, MT5Tokenizer
import torch.utils.data as Data
import transformers

transformers.logging.set_verbosity_error()


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别


seed_torch()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 建立解析对象
parser = argparse.ArgumentParser()
# 给parser实例添加属性
parser.add_argument('--student_checkpoint', default='./mt0-small')
parser.add_argument('-i', '--test_data_path', help='INPUT-DIRECTORY')
parser.add_argument('-o', '--output_dir', help='OUTPUT-DIRECTORY')
parser.add_argument('--dataset', default='dataset1')
parser.add_argument('--max_length', default=256, type=int)
parser.add_argument('--dropout_rate', default=0)
parser.add_argument('--batch_size', default=1)
parser.add_argument('--student_weight_dir', default=r"./weights_t1")
# 传递属性给args实例
args = parser.parse_args()

tokenizer = MT5Tokenizer.from_pretrained(args.student_checkpoint)
max_length = args.max_length


def load_testdata(dir_path):
    file_list = os.listdir(dir_path)
    truth_list, problem_list = [], []
    for i in file_list:
        if i[-4:] == 'json':
            truth_list.append(i)
        else:
            problem_list.append(i)

    D = []
    for p in range(len(problem_list)):
        id = str(p + 1)
        problem_name = 'problem-{}.txt'.format(id)
        problem_path = os.path.join(dir_path, problem_name)

        try:
            with open(problem_path, "r", encoding='utf-8', newline="") as f_p:
                lines = f_p.readlines()
                punctuation_list = ['.', ':', '!', '?']
                new_lines = []
                str_temp = ''
                for line in lines:
                    line_temp = line
                    if line_temp.strip()[-1] in punctuation_list and len(str_temp) == 0:
                        new_lines.append(line)
                    elif line_temp.strip()[-1] in punctuation_list and len(str_temp) != 0:
                        new_lines.append(str_temp)
                        str_temp = ''
                    else:
                        str_temp += line.strip()
                lines = new_lines
        except Exception():
            continue

        for index in range(len(lines)-1):
            text1, text2 = lines[index].strip().replace('\t',''), lines[index+1].strip().replace('\t','')
            D.append(([text1, text2], int(id)))
    return D


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        pass

    def __getitem__(self, index):
        pairs = self.data[index][0]
        return pairs

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    sents = [i for i in data]
    # tokenizer对数据进行编码，得到input_ids, attention_mask
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=max_length,
                                       return_tensors='pt',
                                       return_length=True)
    # input_ids:编码之后的数字
    # attention_mask：补零的位置是0，其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']

    return input_ids, attention_mask


class StudentModel(nn.Module):
    def __init__(self, model, args):
        super(StudentModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(args.dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.dropout(out.last_hidden_state[:, 0])    # 取CLS计算
        out = self.fc(out)
        return out


def dataset_loader(dataset, args):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=args.batch_size,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=False)
    return loader


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def model_predict(model, loader_test, test_data, args):
    model.eval()
    out_list, id_list = [], []
    for i in test_data:
        id_list.append(i[1])

    for i, (input_ids, attention_mask) in tqdm(enumerate(loader_test), total=len(loader_test)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        out = F.softmax(logits, dim=1)
        out = out.argmax(dim=1)
        out = out.detach().cpu().numpy()
        out = list(out.flatten())
        out_list.extend(out)

    id_set = list(set(id_list))
    id_set.sort(key=id_list.index)

    result_list = []
    for id_now in id_set:
        result = []
        for index, id in enumerate(id_list):
            if int(id) == int(id_now):
                result.append(out_list[index])
            if int(id) > int(id_now):
                result_list.append(result)
                break
            if index == len(id_list)-1:
                result_list.append(result)

    for index, id in enumerate(id_set):
        with open(args.output_dir + '/solution-problem-{}.json'.format(str(id)), 'w', encoding='utf-8') as f:
            result_json = {"changes":result_list[index]}
            f.write(json.dumps(result_json, cls=NpEncoder))


def main():

    test_data = load_testdata(args.test_data_path)
    loader_test = dataset_loader(MyDataset(test_data), args)

    student_model = MT5EncoderModel.from_pretrained(args.student_checkpoint).to(device)
    student = StudentModel(student_model, args).to(device)
    print(student)

    student.load_state_dict(torch.load(args.student_weight_dir + '/best_student_weights.pth'))
    model_predict(student, loader_test, test_data, args)


if __name__ == '__main__':
    main()
