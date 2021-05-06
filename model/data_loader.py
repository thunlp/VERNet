import os
import torch
import numpy as np
import json
import re
from torch.autograd import Variable

INCORRECT_ID =0
CORRECT_ID = 1
SUB_ID = 2
PAD_ID = 3

def convert_label(string):
    lab_list = string.strip().split()
    new_labs = list()
    for lab in lab_list:
        if lab == "c":
            new_labs.append(CORRECT_ID)
        elif lab == "i":
            new_labs.append(INCORRECT_ID)
        else:
            raise ("Wrong Label")
    return new_labs

def reform_label(tokens, labels, tokenizer, max_seq_length):
    new_tokens = list()
    new_labels = list()
    assert len(tokens) == len(labels)
    for step, token in enumerate(tokens[:-1]):
        split_token = tokenizer.tokenize(token)
        if len(split_token) > 0:
            new_tokens.extend(split_token)
            new_labels.extend([labels[step]] + [SUB_ID] * (len(split_token) - 1))
    new_tokens = new_tokens[:max_seq_length] + [tokens[-1]]
    new_labels = new_labels[:max_seq_length] + [labels[-1]]
    if len(new_tokens) != len(new_labels):
        print (new_tokens)
        print (new_labels)
        print (tokens)
        print (labels)
    assert len(new_tokens) == len(new_labels)
    return new_tokens, new_labels

def tok2int_sent(example, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    src_tokens = example[0]
    src_labels = example[1]
    hyp_tokens = example[2]
    hyp_labels = example[3]

    src_tokens, src_labels = reform_label(src_tokens, src_labels, tokenizer, max_seq_length)
    hyp_tokens, hyp_labels = reform_label(hyp_tokens, hyp_labels, tokenizer, max_seq_length)

    tokens = src_tokens
    tokens = ["[CLS]"] + tokens
    labels = src_labels
    labels = [PAD_ID] + labels
    input_seg = [0] * len(tokens)

    tokens = tokens + hyp_tokens
    labels = labels + hyp_labels
    input_seg = input_seg + [1] * len(hyp_tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)


    max_len = max_seq_length * 2 + 3
    labels += [PAD_ID] * (max_len - len(input_ids))
    padding = [0] * (max_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    input_seg += padding

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(input_seg) == max_len
    assert len(labels) == max_len
    return input_ids, input_mask, input_seg, labels

def tok2int_list(data, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    inps = list()
    msks = list()
    segs = list()
    labs = list()
    for examples in data:
        inp = list()
        msk = list()
        seg = list()
        lab = list()
        for example in examples:
            input_ids, input_mask, input_seg, labels = tok2int_sent(example, tokenizer, max_seq_length)
            inp.append(input_ids)
            msk.append(input_mask)
            seg.append(input_seg)
            lab.append(labels)
        inps.append(inp)
        msks.append(msk)
        segs.append(seg)
        labs.append(lab)
    return inps, msks, segs, labs




class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, args, test=False, cuda=True, batch_size=64, src_flag=True, hyp_flag=True):
        self.cuda = cuda
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.data_path = data_path
        self.test = test
        self.src_flag = src_flag
        self.hyp_flag = hyp_flag
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0

    def read_file(self, data_path):
        data_list = list()
        with open(data_path) as fin:
            for line in fin:
                examples = list()
                line = line.strip()
                data = json.loads(line)
                src_token = data["src"].strip().split()[:-1] + ["[SEP]"]
                src_label = [PAD_ID] * len(src_token)
                if self.src_flag:
                    src_label = convert_label(data["src_lab"])
                assert len(src_token) == len(src_label)
                if "hyp_lab" in data:
                    assert len(data["hyp"]) == len(data["hyp_lab"])
                for i in range(len(data["hyp"])):
                    hyp_token = data["hyp"][i].strip().split()[:-1] + ["[SEP]"]
                    example = [src_token, src_label, hyp_token]
                    hyp_label = [PAD_ID] * len(hyp_token)
                    if "hyp_lab" in data and self.hyp_flag:
                        hyp_label = convert_label(data["hyp_lab"][i])
                    assert len(hyp_token) == len(hyp_label)
                    example.append(hyp_label)
                    examples.append(example)
                data_list.append(examples)
        return data_list


    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            inp, msk, seg, score = tok2int_list(examples, self.tokenizer, self.max_len)

            inp_tensor = Variable(
                torch.LongTensor(inp))
            msk_tensor = Variable(
                torch.LongTensor(msk))
            seg_tensor = Variable(
                torch.LongTensor(seg))
            score_tensor = Variable(
                torch.LongTensor(score))

            if self.cuda:
                inp_tensor = inp_tensor.cuda()
                msk_tensor = msk_tensor.cuda()
                seg_tensor = seg_tensor.cuda()
                score_tensor = score_tensor.cuda()

            self.step += 1
            return inp_tensor, msk_tensor, seg_tensor, score_tensor

        else:
            self.step = 0
            if not self.test:
                self.shuffle()
            raise StopIteration()


