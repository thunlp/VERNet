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


def eval_result(predicts, labels):
    main_label = 0
    main_correct_count = 0
    correct_sum = 0
    main_predicted_count = 0
    main_total_count = 0
    assert len(predicts) == len(labels)
    for i in range(len(predicts)):
        if labels[i] <= 1:
            predicted_label = predicts[i]
            gold_label = labels[i]
            if gold_label == predicted_label:
                correct_sum += 1
            if predicted_label == main_label:
                main_predicted_count += 1
            if gold_label == main_label:
                main_total_count += 1
            if predicted_label == gold_label and gold_label == main_label:
                main_correct_count += 1
    p = (float(main_correct_count) / float(main_predicted_count)) if (main_predicted_count > 0) else 0.0
    r = (float(main_correct_count) / float(main_total_count)) if (main_total_count > 0) else 0.0
    f = (2.0 * p * r / (p + r)) if (p + r > 0.0) else 0.0
    f05 = ((1.0 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (p + r > 0.0) else 0.0
    return {"p":p, "r":r, "f":f, "f05":f05}


def eval_model(model, validset_reader, args):
    model.eval()
    predicts = list()
    labels = list()
    with torch.no_grad():
        for inp_tensor, msk_tensor, seg_tensor, score_tensor  in validset_reader:
            prob = model(inp_tensor, msk_tensor, seg_tensor)
            predict = torch.max(prob, -1)[1].type_as(score_tensor)
            predict = predict.view([-1, args.evi_num, args.max_len * 2 + 3])
            score_tensor = score_tensor.view([-1, args.evi_num, args.max_len * 2 + 3])
            score_tensor = score_tensor[:, 0]
            predict = predict[:, 0, :]
            predict = predict.contiguous().view(-1).tolist()
            score = score_tensor.contiguous().view(-1).tolist()
            predicts.extend(predict)
            labels.extend(score)
        results = eval_result(predicts, labels)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', help='train path')
    parser.add_argument("--batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--evi_num", default=5, type=int,
                        help="evidence number")
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
    logger.info("loading training set")
    validset_reader = DataLoader(args.test_path, tokenizer, args, batch_size=args.batch_size, hyp_flag=False, test=True)

    logger.info('initializing estimator model')
    bert_model = AutoModel.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model = model.cuda()
    logger.info('Start eval!')
    predict_dict = eval_model(model, validset_reader, args)
    print (predict_dict)