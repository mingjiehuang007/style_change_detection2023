# -*- coding:utf-8 -*-
import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW
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

# np.random.seed(1)
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# torch.backends.cudnn.deterministic = True  # 保证每次结果一样

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:{}".format(device))


# 建立解析对象
parser = argparse.ArgumentParser()
# 给parser实例添加属性
parser.add_argument('--teacher_checkpoint', default=r'D:\NLP\models_torch\uncased_L-12_H-768_A-12')
parser.add_argument('--student_checkpoint', default=r'D:\NLP\models_torch\uncased_L-12_H-768_A-12')
parser.add_argument('--train_data_path', default=r'..\datum\MSRP\msr_paraphrase_train.txt')
parser.add_argument('--valid_data_path', default=None)
parser.add_argument('--test_data_path', default=r'..\datum\MSRP\msr_paraphrase_test.txt')
parser.add_argument('--train_valid_rate', default=0.8, help='length of new train set / length of total train set')
parser.add_argument('--freeze_bert', default=False)
parser.add_argument('--learning_rate', default=2e-5)
parser.add_argument('--output_per_steps', default=100)
parser.add_argument('--max_length', default=128, type=int)
parser.add_argument('--dropout_rate', default=0.1)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--train_epochs', default=5)
parser.add_argument('--distill_epochs', default=5)
parser.add_argument('--teacher_weight_dir', default=r"../weights_t1")
parser.add_argument('--student_weight_dir', default=r"../weights_s1")
parser.add_argument('--temperature', default=4)
parser.add_argument('--alpha', default=0.3)
# 传递属性给args实例
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained(args.teacher_checkpoint)
max_length = args.max_length
ce_loss_criterion = nn.CrossEntropyLoss()
kl_loss_criterion = nn.KLDivLoss(reduction='batchmean')

def load_data(fileName):
# 加载数据
    D = []
    with open(fileName, encoding="utf-8") as f:
        flag = True
        for i in f:
            if flag == True:
                flag = False
                continue
            else:
                label, id1, id2, text1, text2 = i.strip().split("\t")
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
    # tokenizer对数据进行编码，得到input_ids, attention_mask, token_type_ids
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
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


class TeacherModel(nn.Module):
    def __init__(self, bert, args):
        super(TeacherModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(args.dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.dropout(out.last_hidden_state[:, 0])    # 取CLS计算
        out = self.fc(out)
        return out


class StudentModel(nn.Module):
    def __init__(self, bert, args):
        super(StudentModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(args.dropout_rate)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.dropout(out.last_hidden_state[:, 0])    # 取CLS计算
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
        input_ids, attention_mask, token_type_ids, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        model.zero_grad()  # clear previously calculated gradients

        logits = model(input_ids, attention_mask, token_type_ids)   # get model predictions for the current batch
        loss = ce_loss_criterion(logits, labels) # compute the loss between actual and predicted values

        total_loss += loss.item()   # add on to the total loss

        loss.backward() # backward pass to calculate the gradients

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()    # update parameters

        preds = F.softmax(logits, dim=1)
        preds = preds.detach().cpu().numpy()    # model predictions are stored on GPU. So, push it to CPU
        total_preds.append(preds)   # append the model predictions

        if (step+1) % args.output_per_steps == 0 and not step == 0 or (step+1)==len(loader_train):
            if loader_valid is None:
                print(' Batch {} of {}. train_loss: {:.4f}'.format(step+1, len(loader_train), loss.item()))
            else:
                correct, total = 0, 0
                total_val_loss = 0
                for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_valid):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        logits = model(input_ids, attention_mask, token_type_ids)
                        val_loss = ce_loss_criterion(logits, labels)
                        total_val_loss += val_loss.item()
                    out = F.softmax(logits, dim=1)
                    out = out.argmax(dim=1)
                    correct += (out == labels).sum().item()
                    total += len(labels)
                avg_val_loss = total_val_loss / len(loader_valid)
                val_accuracy = correct / total
                print(' Batch {} of {}. train_loss: {:.4f} val_loss: {:.4f} val_acc: {:.4f}'.format(step+1, len(loader_train), loss.item(), avg_val_loss, correct / total))

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

    for i, (input_ids, attention_mask, token_type_ids, labels) in tqdm(enumerate(loader_test), total=len(loader_test)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        out = F.softmax(out, dim=1)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    test_accuracy = correct / total
    return test_accuracy


def distill(epoch_num, teacher_model, student_model, args, loader_train, loader_valid=None):
    teacher_model.eval()
    student_model.train()
    student_optimizer = AdamW(student_model.parameters(), lr=args.learning_rate)

    total_loss, total_accuracy, avg_val_loss, val_accuracy= 0, 0, float('inf'), 0
    total_preds = []
    for step, (input_ids, attention_mask, token_type_ids, labels) in tqdm(enumerate(loader_train), total=len(loader_train)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids, attention_mask, token_type_ids)
        student_logits = student_model(input_ids, attention_mask, token_type_ids)

        student_loss = ce_loss_criterion(student_logits, labels)
        student_preds = F.softmax(student_logits, dim=1)

        distillation_loss = kl_loss_criterion(input=F.log_softmax(student_logits/args.temperature, dim=1),
                                   target=F.softmax(teacher_logits/args.temperature, dim=1),
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
                for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_valid):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        logits = student_model(input_ids, attention_mask, token_type_ids)
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
        test_data = load_data(args.test_data_path)
        train_data = train_data_temp[:int(len(train_data_temp) * args.train_valid_rate)]
        valid_data = train_data_temp[int(len(train_data_temp) * args.train_valid_rate):]
    else:
        train_data = load_data(args.train_data_path)
        valid_data = load_data(args.valid_data_path)
        test_data = load_data(args.test_data_path)

    loader_train = dataset_loader(MyDataset(train_data), args)
    loader_valid = dataset_loader(MyDataset(valid_data), args)
    loader_test = dataset_loader(MyDataset(test_data), args)

    teacher_model = BertModel.from_pretrained(args.teacher_checkpoint).to(device)
    student_model = BertModel.from_pretrained(args.student_checkpoint).to(device)

    if args.freeze_bert:
        # 不训练,不需要计算梯度
        for param in teacher_model.parameters():
            param.requires_grad_(False)

    teacher = TeacherModel(teacher_model, args).to(device)
    optimizer = AdamW(teacher.parameters(), lr=args.learning_rate)
    print(teacher)

    student = StudentModel(student_model, args).to(device)
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
        avg_loss, _, avg_val_loss, val_accuracy = distill(i, teacher, student, args, loader_train, loader_valid)
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
