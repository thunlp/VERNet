import random, os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from transformers import *

from models import inference_model
from data_loader import DataLoader
from torch.nn import NLLLoss
import logging
import json

logger = logging.getLogger(__name__)


def tok2int_sent(example, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    src_tokens = example[0]
    hyp_tokens = example[1]

    src_tokens = tokenizer.tokenize(src_tokens)
    src_tokens = src_tokens[:max_seq_length]
    hyp_tokens = tokenizer.tokenize(hyp_tokens)
    hyp_tokens = hyp_tokens[:max_seq_length]

    tokens = ["[CLS]"] + src_tokens + ["[SEP]"]
    input_seg = [0] * len(tokens)
    input_label = [0] * len(tokens)

    tokens = tokens + hyp_tokens  + ["[SEP]"]
    for token in hyp_tokens:
        if "##" in token:
            input_label.append(0)
        else:
            input_label.append(1)
    input_label.append(1)
    input_seg = input_seg + [1] * (len(hyp_tokens) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)


    max_len = max_seq_length * 2 + 3
    padding = [0] * (max_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    input_seg += padding
    input_label += padding

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(input_seg) == max_len
    assert len(input_label) == max_len
    return input_ids, input_mask, input_seg, input_label

def tok2int_list(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    inps = list()
    msks = list()
    segs = list()
    labs = list()
    for example in examples:
        input_ids, input_mask, input_seg, input_lab = tok2int_sent(example, tokenizer, max_seq_length)
        inps.append(input_ids)
        msks.append(input_mask)
        segs.append(input_seg)
        labs.append(input_lab)
    return inps, msks, segs, labs




def eval_model(model, tokenizer, args):
    examples = list()
    with open(args.test_path + ".src") as fsrc, open(args.test_path + ".hyp") as fhyp:
        lines = zip(fsrc, fhyp)
        for line in lines:
            examples.append([line[0].strip(), line[1].strip()])
    total_step = int(np.ceil(len(examples) * 1.0 / (args.batch_size * args.evi_num)))
    model.eval()
    predicts = list()
    with torch.no_grad():
        for step  in range(total_step):
            example = examples[step * args.evi_num * args.batch_size: (step + 1) * args.evi_num * args.batch_size]
            inp_tensor, msk_tensor, seg_tensor, lab_tensor = tok2int_list(example, tokenizer, args.max_len)
            inp_tensor = Variable(
                torch.LongTensor(inp_tensor))
            msk_tensor = Variable(
                torch.LongTensor(msk_tensor))
            seg_tensor = Variable(
                torch.LongTensor(seg_tensor))
            lab_tensor = Variable(
                torch.LongTensor(lab_tensor))
            inp_tensor = inp_tensor.cuda()
            msk_tensor = msk_tensor.cuda()
            seg_tensor = seg_tensor.cuda()
            lab_tensor = lab_tensor.cuda()
            prob = model(inp_tensor, msk_tensor, seg_tensor, score_flag = False)
            prob = prob.view(-1, args.max_len * 2 + 3, 4)
            prob = prob[:, :, :2]
            prob = F.softmax(prob, -1)
            prob = prob[:, :, 1].squeeze(-1)
            prob = torch.sum(prob * lab_tensor.float(), 1) / torch.sum(lab_tensor.float(), 1)
            prob = prob.tolist()
            predicts.extend(prob)

    with open(args.out_path, "w")  as fout:
        for predict in predicts:
            fout.write(str(predict) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument('--out_path', help='output path')
    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--evi_num", default=5, type=int)
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()
    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logger.info('Start training!')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain)

    logger.info('initializing estimator model')
    bert_model = AutoModel.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    logger.info('Start eval!')
    eval_model(model, tokenizer, args)