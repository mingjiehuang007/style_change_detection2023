# -*- coding:utf-8 -*-
import json
import os

def rearrange():
    dir_path = r"D:\NLP\datasets\authorship verification\PAN2023_datasets\release"
    new_path = r'../datum/style_change_detection_rearrange'
    dataset_list = os.listdir(dir_path)
    for data_type in ['train', 'validation']:
        for task, dataset in enumerate(dataset_list):
            train_set = os.path.join(dir_path, dataset, dataset + '-' + data_type)
            file_list = os.listdir(train_set)
            truth_list, problem_list = [], []
            for i in file_list:
                if i[-4:] == 'json':
                    truth_list.append(i)
                else:
                    problem_list.append(i)

            pairs_list = []
            for p in range(len(problem_list)):
                id = str(p+1)
                problem_name = 'problem-{}.txt'.format(id)
                truth_name = 'truth-problem-{}.json'.format(id)
                problem_path = os.path.join(dir_path, dataset, dataset + '-' + data_type, problem_name)
                truth_path = os.path.join(dir_path, dataset, dataset + '-' + data_type, truth_name)

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

                with open(truth_path, "r", encoding='utf-8') as f_t:
                    content = json.load(f_t)
                    changes = content['changes']
                    for index, label in enumerate(changes):
                        pairs = id + '\t' + lines[index].strip().replace('\t','') + '\t' + lines[index+1].strip().replace('\t','') + '\t' + str(label) + '\n'
                        pairs_list.append(pairs)

                if len(changes) != len(lines)-1:
                    print(problem_path, len(changes), len(lines))

            output_path = os.path.join(new_path, '{}{}.text'.format(data_type,str(task+1)))
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for i in pairs_list:
                    f_out.write(i)


if __name__ == '__main__':
    rearrange()
