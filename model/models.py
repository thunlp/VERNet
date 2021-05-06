import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.pred_model = bert_model
        self.proj_hidden = nn.Linear(self.bert_hidden_dim * 3, 4)
        self.evi_num = args.evi_num
        self.model_name = args.bert_pretrain
        self.max_len = args.max_len * 2 + 3
        self.proj_select = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.proj_select_score = nn.Linear(self.bert_hidden_dim * 3, 1)
        self.proj_attention = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)


    def attention_layer(self, inputs_hidden, select_score, mask_text, i):
        inputs_hidden1 = inputs_hidden[:, i, :, :]
        mask_text1 = mask_text[:, i, :, :]
        inputs_hidden1 = inputs_hidden1.unsqueeze(1).repeat(1, self.evi_num, 1, 1)
        mask_text1 = mask_text1.unsqueeze(1).repeat(1, self.evi_num, 1, 1)

        inputs_hidden1 = inputs_hidden1.view(-1, self.max_len, self.bert_hidden_dim)
        mask_text1 = mask_text1.view(-1, self.max_len, 1)
        inputs_hidden2 = inputs_hidden.view(-1, self.max_len, self.bert_hidden_dim)
        mask_text2 = mask_text.view(-1, self.max_len, 1)

        match_matrix = torch.bmm(inputs_hidden1, torch.transpose(inputs_hidden2, 1, 2))
        mask_matrix = torch.bmm(mask_text1, torch.transpose(mask_text2, 1, 2))
        match_matrix = match_matrix.masked_fill_((1 - mask_matrix).bool(), -1e4)

        attention = F.softmax(match_matrix, dim=2)
        attention = attention.view(-1, self.max_len, self.max_len)
        inputs_hidden2 = torch.bmm(attention, inputs_hidden2)
        inputs_hidden2 = inputs_hidden2.view(-1, self.evi_num, self.max_len, self.bert_hidden_dim)

        select_score = F.softmax(select_score, 1)
        select_score = select_score.view([-1, self.evi_num, 1, 1])
        verificcation_representation = torch.sum(inputs_hidden2 * select_score, 1)
        return verificcation_representation


    def select_layer(self, inputs_hidden, src_mask, hyp_mask):
        src_mask = src_mask.view(-1, self.max_len, 1)
        hyp_mask = hyp_mask.view(-1, self.max_len, 1)
        inputs_hidden2 = inputs_hidden.view(-1, self.max_len, self.bert_hidden_dim)
        inputs_hidden1 = self.proj_attention(inputs_hidden2)
        match_matrix = torch.bmm(inputs_hidden1, torch.transpose(inputs_hidden2, 1, 2))
        mask_matrix = torch.bmm(src_mask, torch.transpose(hyp_mask, 1, 2))
        match_matrix = match_matrix.masked_fill_((1 - mask_matrix).bool(), -1e4)
        match_matrix = match_matrix.view(-1, self.max_len, self.max_len)
        src_attention = torch.sum(F.softmax(match_matrix, dim=1) * mask_matrix, 2) / torch.sum(hyp_mask.view(-1, 1, self.max_len), 2)
        hyp_attention = torch.sum(F.softmax(match_matrix, dim=2) * mask_matrix, 1) / torch.sum(src_mask.view(-1, self.max_len, 1), 1)
        src_attention = src_attention.view(-1, self.max_len, 1)
        hyp_attention = hyp_attention.view(-1, self.max_len, 1)
        src_representation = torch.sum(src_attention * inputs_hidden2, 1)
        hyp_representation = torch.sum(hyp_attention * inputs_hidden2, 1)
        select_score = self.proj_select_score(torch.cat([src_representation, hyp_representation, src_representation * hyp_representation], -1))
        select_score = select_score.view([-1, self.evi_num])

        return select_score






    def forward(self, inp_tensor, msk_tensor, seg_tensor, score_flag=True):
        inp_tensor = inp_tensor.view(-1, self.max_len)
        msk_tensor = msk_tensor.view(-1, self.max_len)
        seg_tensor = seg_tensor.view(-1, self.max_len)
        if "bert-" in self.model_name.lower():
            outputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        elif "electra" in self.model_name.lower():
            outputs = self.pred_model(inp_tensor, msk_tensor)
        else:
            raise ("Not implement!")
        inputs_hidden = outputs[0]
        mask_text = msk_tensor.float()
        mask_text[:, 0] = 0.0
        mask_src = (1 - seg_tensor.float()) * mask_text
        mask_hyp = seg_tensor.float() * mask_text

        select_score = self.select_layer(inputs_hidden, mask_src, mask_hyp)
        verification_hidden = list()
        inputs_hidden = inputs_hidden.view(-1, self.evi_num, self.max_len, self.bert_hidden_dim)
        mask_text = mask_text.view(-1, self.evi_num, self.max_len, 1)
        for i in range(self.evi_num):
            #select_score[:, i] = -1e4
            outputs = self.attention_layer(inputs_hidden, select_score, mask_text, i)
            outputs = outputs.view(-1, 1, self.max_len, self.bert_hidden_dim)
            verification_hidden.append(outputs)
        verification_hidden = torch.cat(verification_hidden, 1)
        inputs_hidden = torch.cat([inputs_hidden, verification_hidden, inputs_hidden * verification_hidden], -1)
        score = self.proj_hidden(inputs_hidden)
        if score_flag:
            score = F.softmax(score, dim=-1)
        return score












