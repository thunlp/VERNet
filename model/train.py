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
import torch.nn as nn

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



def eval_model(model, validset_reader):
    model.eval()
    predicts = list()
    labels = list()
    with torch.no_grad():
        for step, (inp_tensor, msk_tensor, seg_tensor, score_tensor) in enumerate(validset_reader):
            prob = model(inp_tensor, msk_tensor, seg_tensor)
            predict = torch.max(prob, -1)[1].type_as(score_tensor)
            predict = predict.view(-1).tolist()
            score = score_tensor.view(-1).tolist()
            predicts.extend(predict)
            labels.extend(score)
            #if step > 500:
            #    break
        results = eval_result(predicts, labels)
    return results




def train_model(model, ori_model, args, trainset_reader, validset_reader):
    save_path = args.outdir + '/model'
    best_acc = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_step / args.gradient_accumulation_steps * args.num_train_epochs)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )
    #optimizer = optim.Adam(model.parameters(), args.learning_rate)
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        optimizer.zero_grad()
        for inp_tensor, msk_tensor, seg_tensor, score_tensor in trainset_reader:
            model.train()
            score = model(inp_tensor, msk_tensor, seg_tensor)
            log_score = torch.log(score).view(-1, 4)
            score_tensor = score_tensor.view(-1)
            loss = F.nll_loss(log_score, score_tensor,  ignore_index=3)
            running_loss += loss.item()
            if args.gradient_accumulation_steps != 0:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            if global_step % (args.eval_step * args.gradient_accumulation_steps) == 0:
                logger.info('Start eval!')
                result_dict = eval_model(model, validset_reader)
                f05 = result_dict["f05"]
                logger.info('Dev p: {0}, r: {1}, f: {2}, f05: {3}'.format(result_dict["p"], result_dict["r"], result_dict["f"], result_dict["f05"]))
                if f05 >= best_acc:
                    best_acc = f05
                    torch.save({'epoch': epoch,
                                'model': ori_model.state_dict()}, save_path + ".best.pt")
                    logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--valid_batch_size", default=4, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--max_len", default=120, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=500, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--evi_num", default=5, type=int,
                        help="evidence number")
    parser.add_argument('--bert_pretrain', required=True)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain)

    logger.info('Start training!')
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, tokenizer, args, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    validset_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size, test=True)
    logger.info('initializing estimator model')
    bert_model = AutoModel.from_pretrained(args.bert_pretrain)
    bert_model = bert_model.cuda()
    ori_model = inference_model(bert_model, args)
    model = nn.DataParallel(ori_model)
    model = model.cuda()
    train_model(model, ori_model, args, trainset_reader, validset_reader)