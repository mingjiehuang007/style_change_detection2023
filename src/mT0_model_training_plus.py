# author: Mingjie Huang
# create time：2023/5/11 0011 13:44
# -*- coding:UTF-8 -*-
import argparse
import os, sys
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW, Adafactor, AutoTokenizer, MT5EncoderModel
import torch.utils.data as Data


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


# 将控制台的输出，保存到log文件中
class Logger(object):
    def __init__(self, filename='wes.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(stream=sys.stdout)  # 将控制台输出保存到文件中

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:{}".format(device))

# 建立解析对象
parser = argparse.ArgumentParser()
# 给parser实例添加属性
parser.add_argument('--teacher_checkpoint', default=r'D:\NLP\models_torch\mt0-base')
parser.add_argument('--student_checkpoint', default=r'D:\NLP\models_torch\mt0-small')
parser.add_argument('--train_data_path',
                    default=r'../datum/style_change_detection_rearrange/train1.text')
parser.add_argument('--valid_data_path', default=None)
parser.add_argument('--test_data_path',
                    default=r'../datum/style_change_detection_rearrange/validation1.text')
parser.add_argument('--train_valid_rate', default=0.8, help='length of new train set / length of total train set')
parser.add_argument('--freeze_bert', default=False)
parser.add_argument('--learning_rate', default=2e-4)
parser.add_argument('--weight_decay', default=1e-4)
parser.add_argument('--output_per_steps', default=500)
parser.add_argument('--max_length', default=256, type=int)
parser.add_argument('--dropout_rate', default=0.1)
parser.add_argument('--batch_size', default=4)
parser.add_argument('--train_epochs', default=10)
parser.add_argument('--distill_epochs', default=10)
parser.add_argument('--teacher_weight_dir', default=r"../weights_t1")
parser.add_argument('--student_weight_dir', default=r"../weights_s1")
parser.add_argument('--temperature', default=4)
parser.add_argument('--alpha', default=0.7, help='ground truth weights')
# 传递属性给args实例
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.teacher_checkpoint)
max_length = args.max_length
ce_loss_criterion = nn.CrossEntropyLoss()
kl_loss_criterion = nn.KLDivLoss(reduction='batchmean')


def random_(text):
    np.random.shuffle(text)


def load_data(fileName):
    # 加载数据
    D = []
    with open(fileName, encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            id, text1, text2, label = i.strip().split("\t")
            D.append(([text1, text2], int(label)))
    return D


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        pass

    def __getitem__(self, index):
        pairs = self.data[index][0]
        labels = self.data[index][1]
        return pairs, labels

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
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
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, labels


class TeacherModel(nn.Module):
    def __init__(self, model, args):
        super(TeacherModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(args.dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.dropout(out.last_hidden_state[:, 0])  # 取CLS计算
        out = self.fc(out)
        return out


class StudentModel(nn.Module):
    def __init__(self, model, args):
        super(StudentModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(args.dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        out = self.dropout(out.last_hidden_state[:, 0])  # 取CLS计算
        out = self.fc(out)
        return out


def dataset_loader(dataset, args):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=args.batch_size,
                                         collate_fn=collate_fn,
                                         shuffle=True,
                                         drop_last=True)
    return loader


def model_train(epoch_num, model, optimizer, args, loader_train, loader_valid=None):
    model.train()
    total_loss, total_accuracy, val_accuracy = 0, 0, 0
    total_preds = []

    for step, batch in tqdm(enumerate(loader_train), total=len(loader_train)):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        model.zero_grad()  # clear previously calculated gradients

        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask)  # get model predictions for the current batch
        preds = F.softmax(logits, dim=1)
        loss = ce_loss_criterion(logits, labels)  # compute the loss between actual and predicted values

        total_loss += loss.item()  # add on to the total loss

        loss.backward()  # backward pass to calculate the gradients

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()  # update parameters

        preds = preds.detach().cpu().numpy()  # model predictions are stored on GPU. So, push it to CPU
        total_preds.append(preds)  # append the model predictions

        if (step + 1) % args.output_per_steps == 0 and not step == 0 or (step + 1) == len(loader_train):
            if loader_valid is None:
                print(' Batch {} of {}. train_loss: {:.4f}'.format(step + 1, len(loader_train), loss.item()))
            else:
                correct, total = 0, 0
                total_val_loss = 0
                for i, (input_ids, attention_mask, labels) in enumerate(loader_valid):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        out = F.softmax(logits, dim=1)
                        val_loss = ce_loss_criterion(logits, labels)
                        total_val_loss += val_loss.item()
                    out = out.argmax(dim=1)
                    correct += (out == labels).sum().item()
                    total += len(labels)
                avg_val_loss = total_val_loss / len(loader_valid)
                val_accuracy = correct / total
                print(' Batch {} of {}. train_loss: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(step + 1,
                                                                                                    len(loader_train),
                                                                                                    loss.item(),
                                                                                                    avg_val_loss,
                                                                                                    correct / total))

    # compute the training loss of the epoch
    avg_loss = total_loss / len(loader_train)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    torch.save(model.state_dict(), args.teacher_weight_dir + '/model_weights_' + str(epoch_num) + '.pth')

    # returns the loss and predictions
    return avg_loss, total_preds, val_loss, val_accuracy


def model_test(model, loader_test):
    model.eval()
    correct, total = 0, 0

    for i, (input_ids, attention_mask, labels) in tqdm(enumerate(loader_test), total=len(loader_test)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
        out = F.softmax(logits, dim=1)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    test_accuracy = correct / total
    return test_accuracy


def distill(epoch_num, teacher_model, student_model, student_optimizer, args, loader_train, loader_valid=None):
    teacher_model.eval()
    student_model.train()

    total_loss, total_accuracy, avg_val_loss, val_accuracy = 0, 0, float('inf'), 0
    total_preds = []
    for step, (input_ids, attention_mask, labels) in tqdm(enumerate(loader_train), total=len(loader_train)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask)

        student_preds = F.softmax(student_logits, dim=1)
        student_loss = ce_loss_criterion(student_logits, labels)
        distillation_loss = kl_loss_criterion(input=F.log_softmax(student_logits / args.temperature, dim=1),
                                              target=F.softmax(teacher_logits / args.temperature, dim=1),
                                              )
        loss = args.alpha * student_loss + (1 - args.alpha) * distillation_loss

        total_loss += loss.item()

        student_optimizer.zero_grad()
        loss.backward()
        student_optimizer.step()

        student_preds = student_preds.detach().cpu().numpy()  # model predictions are stored on GPU. So, push it to CPU
        total_preds.append(student_preds)  # append the model predictions

        if (step + 1) % args.output_per_steps == 0 and not step == 0 or (step + 1) == len(loader_train):
            if loader_valid is None:
                print(' Batch {} of {}. train_loss: {:.4f}'.format(step + 1, len(loader_train), loss.item()))
            else:
                correct, total = 0, 0
                total_val_loss = 0
                for i, (input_ids, attention_mask, labels) in enumerate(loader_valid):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        logits = student_model(input_ids=input_ids, attention_mask=attention_mask)
                        val_loss = ce_loss_criterion(logits, labels)
                        total_val_loss += val_loss.item()
                    out = F.softmax(logits, dim=1)
                    out = out.argmax(dim=1)
                    correct += (out == labels).sum().item()
                    total += len(labels)
                avg_val_loss = total_val_loss / len(loader_valid)
                val_accuracy = correct / total
                print(' Batch {} of {}. train_loss: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(step + 1,
                                                                                                    len(loader_train),
                                                                                                    loss.item(),
                                                                                                    avg_val_loss,
                                                                                                    val_accuracy))
    # compute the training loss of the epoch
    avg_loss = total_loss / len(loader_train)
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)
    torch.save(student_model.state_dict(), args.student_weight_dir + '/model_weights_' + str(epoch_num) + '.pth')
    # returns the loss and predictions
    return avg_loss, total_preds, avg_val_loss, val_accuracy


def main():
    if args.valid_data_path is None:
        train_data_temp = load_data(args.train_data_path)
        random_(train_data_temp)
        test_data = load_data(args.test_data_path)
        train_data = train_data_temp[:int(len(train_data_temp) * args.train_valid_rate)]
        valid_data = train_data_temp[int(len(train_data_temp) * args.train_valid_rate):]
    else:
        train_data = load_data(args.train_data_path)
        random_(train_data)
        valid_data = load_data(args.valid_data_path)
        test_data = load_data(args.test_data_path)

    loader_train = dataset_loader(MyDataset(train_data), args)
    loader_valid = dataset_loader(MyDataset(valid_data), args)
    loader_test = dataset_loader(MyDataset(test_data), args)

    teacher_model = MT5EncoderModel.from_pretrained(args.teacher_checkpoint).to(device)
    student_model = MT5EncoderModel.from_pretrained(args.student_checkpoint).to(device)

    teacher = TeacherModel(teacher_model, args).to(device)
    optimizer = Adafactor(teacher.parameters(), scale_parameter=False, relative_step=False, warmup_init=False,
                          lr=args.learning_rate)
    print(teacher)

    student = StudentModel(student_model, args).to(device)
    student_optimizer = Adafactor(student.parameters(), scale_parameter=False, relative_step=False, warmup_init=False,
                                  lr=args.learning_rate)
    print(student)

    # train a teacher model
    avg_loss_min, accuracy_max = float('inf'), 0
    for i in range(args.train_epochs):
        print('----------Epoch {}----------'.format(i))
        avg_loss, _, avg_val_loss, val_accuracy = model_train(i, teacher, optimizer, args, loader_train, loader_valid)
        test_accuracy = model_test(teacher, loader_test)
        print('Epoch {} ends. avg_train_loss: {:.4f} test_acc: {:.4f}'.format(i, avg_loss, test_accuracy))
        # if avg_val_loss < avg_loss_min:
        #     avg_loss_min = avg_val_loss
        #     torch.save(teacher.state_dict(), args.teacher_weight_dir + 'best_teacher_weights.pth')
        if val_accuracy > accuracy_max:
            accuracy_max = val_accuracy
            torch.save(teacher.state_dict(), args.teacher_weight_dir + '/best_teacher_weights.pth')

    teacher.load_state_dict(torch.load(args.teacher_weight_dir + '/best_teacher_weights.pth'))
    test_accuracy = model_test(teacher, loader_test)
    print('final test_acc: {:.4f}'.format(test_accuracy))

    # distillation
    avg_loss_min, accuracy_max = float('inf'), 0
    for i in range(args.distill_epochs):
        print('----------Epoch {}----------'.format(i))
        avg_loss, _, avg_val_loss, val_accuracy = distill(i, teacher, student, student_optimizer, args, loader_train,
                                                          loader_valid)
        test_accuracy = model_test(student, loader_test)
        print('Epoch {} ends. avg_train_loss: {:.4f} test_acc: {:.4f}'.format(i, avg_loss, test_accuracy))
        # if avg_val_loss < avg_loss_min:
        #     avg_loss_min = avg_val_loss
        #     torch.save(student.state_dict(), args.student_weight_dir + 'best_student_weights.pth')
        if val_accuracy > accuracy_max:
            accuracy_max = val_accuracy
            torch.save(student.state_dict(), args.student_weight_dir + '/best_student_weights.pth')

    student.load_state_dict(torch.load(args.student_weight_dir + '/best_student_weights.pth'))
    test_accuracy = model_test(student, loader_test)
    print('final test_acc: {:.4f}'.format(test_accuracy))


if __name__ == '__main__':
    main()
