import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst


def load_wiki_word_list(f_path):
    vocab = []
    with open(f_path,"r")as f:
        line = f.readline()
        while line:
            vocab.append(line.strip().split()[0])
            line = f.readline()
    return vocab


class attr_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): # this must be rewriten
        return len(self.data['input_ids'])
    def __getitem__(self, idx): # this must be rewriten
        self.temp_batch = {}
        for k,v in self.data.items():
            self.temp_batch[k] = v[idx]
        return self.temp_batch

class Distribution_Concate_v3(nn.Module):
    def __init__(self, reduction='batchmean', device='cuda'):
        super(Distribution_Concate_v3, self).__init__()
        self.reduction = reduction
        self.device = device

    def forward(self, net_logits, mask_nums):
        loss = []
        self.attr_num = len(net_logits)
        for k in mask_nums:
            attr_logits = []
            attr_probs = []
            for i in range(self.attr_num):
                net_logits[i][k] = torch.sum(net_logits[i][k], dim=0)
                attr_logits.append(net_logits[i][k])
                attr_probs.append(F.softmax(attr_logits[i], dim=1))
                if i==0:
                    sum_probs = attr_probs[i]
                else:
                    sum_probs = torch.add(sum_probs, attr_probs[i])
            total_m = (1/self.attr_num) * sum_probs
            loss_k = 0.0
            for i in range(self.attr_num):
                loss_k += F.kl_div(torch.log_softmax(attr_logits[i], dim=1), total_m, reduction=self.reduction)
            if self.reduction == "none":
                loss_k = torch.sum(loss_k, dim=1)
            loss_k = (1/self.attr_num) * loss_k

            loss.append(loss_k)
        
        # concate distribution through JSD:
        for i in range(len(loss)):
            if i==0:
                final_loss = loss[i]
            else:
                final_loss += loss[i]
        
        return (final_loss)


class Distribution_Concate_v4(nn.Module):
    def __init__(self, reduction='batchmean', device='cuda'):
        super(Distribution_Concate_v4, self).__init__()
        self.reduction = reduction
        self.device = device

    def forward(self, net_logits, mask_nums):
        loss = []
        self.attr_num = len(net_logits)
        for k in mask_nums:
            attr_logits = []
            attr_probs = []
            for i in range(self.attr_num):
                probs = F.softmax(net_logits[i][k], dim=1)
                attr_probs.append(torch.cumprod(probs, dim = 0)[-1])

                if i==0:
                    sum_probs = attr_probs[i]
                else:
                    sum_probs = torch.add(sum_probs, attr_probs[i])
            total_m = (1/self.attr_num) * sum_probs
            loss_k = 0.0
            for i in range(self.attr_num):
                loss_k += F.kl_div(torch.log(attr_probs[i]), total_m, reduction=self.reduction)
            if self.reduction == "none":
                loss_k = torch.sum(loss_k, dim=1)
            loss_k = (1/self.attr_num) * loss_k

            loss.append(loss_k)
        
        # concate distribution through JSD:
        for i in range(len(loss)):
            if i==0:
                final_loss = loss[i]
            else:
                final_loss += loss[i]
        
        return (final_loss)
    

class JSD(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSD, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_logits, net_2_logits):
        total_m = 0.5 * (net_1_logits + net_2_logits)
        loss = 0.0
        loss += F.kl_div(torch.log(net_1_logits), total_m, reduction=self.reduction)
        loss += F.kl_div(torch.log(net_2_logits), total_m, reduction=self.reduction)
     
        return (0.5 * loss) 

class JSD_Train(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(JSD_Train, self).__init__()
        self.reduction = reduction

    def forward(self, net_1_logits, net_2_logits):
        net_1_probs = F.softmax(net_1_logits, dim=1)
        net_2_probs = F.softmax(net_2_logits, dim=1)
        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(torch.log_softmax(net_1_logits, dim=1), total_m, reduction=self.reduction)
        loss += F.kl_div(torch.log_softmax(net_2_logits, dim=1), total_m, reduction=self.reduction)
     
        return (0.5 * loss) 
    
def clean_vocab(vocab):
    new_vocab = []
    for v in vocab:
        if (v[0] not in ['#','[','.','0','1','2','3','4','5','6','7','8','9']) and len(v)>1:
            new_vocab.append(v)
    return new_vocab


def clean_word_list2(tar1_words_, tar2_words_, tokenizer):
    tar1_words = []
    tar2_words = []
    for i in range(len(tar1_words_)):
        if tokenizer.convert_tokens_to_ids(tar1_words_[i])!=tokenizer.unk_token_id and tokenizer.convert_tokens_to_ids(tar2_words_[i])!=tokenizer.unk_token_id:
            tar1_words.append(tar1_words_[i])
            tar2_words.append(tar2_words_[i])
    return tar1_words, tar2_words

def clean_word_list(vocabs, tokenizer):
    vocab_list = []
    for i in range(len(vocabs)):
        if tokenizer(vocabs[i])['input_ids'][1]!=tokenizer.unk_token_id:
            vocab_list.append(vocabs[i])
    return vocab_list

def clean_word_list_v2(vocabs,tokenizer):
    vocab_list = []
    for i in range(len(vocabs)):
        if tokenizer.convert_tokens_to_ids(vocabs[i])!=tokenizer.unk_token_id:
            vocab_list.append(vocabs[i])
    return vocab_list

def clean_multi_word_list(vocabs, max_token_num, tokenizer):
    vocab_list = []
    for i in range(len(vocabs)):
        p_wordslist = tokenizer(vocabs[i], add_special_tokens=False).input_ids
        count = 0
        for j in range(len(p_wordslist)):
            if p_wordslist[j] != tokenizer.unk_token_id:
                count += 1
        if count==len(p_wordslist) and len(p_wordslist) <= max_token_num:
            vocab_list.append(vocabs[i])
    return vocab_list

def clean_multi_word_list2(vocabs_1, vocabs_2, max_token_num, tokenizer):
    vocab_list_1 = []
    vocab_list_2 = []
    for i in range(len(vocabs_1)):
        p_wordslist_1 = tokenizer(vocabs_1[i], add_special_tokens=False).input_ids
        p_wordslist_2 = tokenizer(vocabs_2[i], add_special_tokens=False).input_ids
        count_1 = 0
        count_2 = 0
        for j in range(len(p_wordslist_1)):
            if p_wordslist_1[j] != tokenizer.unk_token_id:
                count_1 += 1
        for j in range(len(p_wordslist_2)):
            if p_wordslist_2[j] != tokenizer.unk_token_id:
                count_2 += 1
        if count_1==len(p_wordslist_1) and len(p_wordslist_1) <= max_token_num:
            if count_2==len(p_wordslist_2) and len(p_wordslist_2) <= max_token_num:
                vocab_list_1.append(vocabs_1[i])
                vocab_list_2.append(vocabs_2[i])
    return vocab_list_1, vocab_list_2

def count_mask_nums(word_list, tokenizer):
    mask_nums = []
    for phrase in word_list:
        phrase_tokenslist = tokenizer(phrase, add_special_tokens=False).input_ids
        length = len(phrase_tokenslist)
        if length not in mask_nums:
            mask_nums.append(length)
    return sorted(mask_nums)

def tokens_to_ids(word_list, tokenizer):
    '''
    return a dict, with key is num of masked token
    '''
    tokens_dict = {1:[], 2:[], 3:[], 4:[], 5:[]}
    for phrase in word_list:
        phrase_tokenslist = tokenizer(phrase, add_special_tokens=False).input_ids
        tokens_dict[len(phrase_tokenslist)].append((phrase_tokenslist))
    for k,v in tokens_dict.items():
        tokens_dict[k] = np.array(v)
    return tokens_dict
