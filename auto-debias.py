import argparse
import torch
import os
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_scheduler
from transformers import BertTokenizer,BertForPreTraining 
from transformers import RobertaTokenizer,RobertaForMaskedLM,RobertaModel
from transformers import AlbertTokenizer, AlbertForPreTraining
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForMaskedLM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser()

parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender'],
    help="Choose from ['gender']",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type",
    default="bert",
    type=str,
    help="choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--data_path",
    default="data/",
    type=str,
    help="data path to put the taget/attribute word list",
)


parser.add_argument(
    "--prompts_file",
    default="",
    type=str,
    help="the name of the file that stores the prompts, by default it is under the data_path",
)

parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="batch size in auto-debias fine-tuning",
)

parser.add_argument(
    "--lr",
    default=1e-5,
    type=float,
    help="learning rate in auto-debias fine-tuning",
)

parser.add_argument(
    "--epochs",
    default=5,
    type=int,
    help="number of epochs in auto-debias fine-tuning",
)

parser.add_argument(
    "--finetuning_vocab_file",
    default=None,
    type=str,
    help="vocabulary to be fine-tuned in auto-debias fine-tuning, if None, tune the whole vocabulary.",
)

parser.add_argument(
    "--tune_pooling_layer",
    default=False,
    type=bool,
    help="whether to tune the pooling layer with the auxiliary loss",
)

def get_tokenized_prompt_v3(prompts, tar_words, tokenizer):
    attr_num = len(tar_words)
    length = len(tar_words[0])
    tar_tokenized = {}
    temp_tar_sen = []
    for i in range(len(prompts)):
        for j in range(length):
            for m in range(attr_num): # m attr words = m channels
                temp_tar_sen.append(tar_words[m][j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")

    temp_tar_tokenized = tokenizer(temp_tar_sen, padding=True, truncation=True, return_tensors="np")
    temp_tar_mask_index = np.where(temp_tar_tokenized['input_ids']==tokenizer.mask_token_id)[-1]
    temp_tar_tokenized['tar_mask_index'] = temp_tar_mask_index

    #reshape channels:
    for k,v in temp_tar_tokenized.items():
        tar_tokenized[k] = []
        for s in range(0, len(v), attr_num):
            tar_tokenized[k].append(v[s:s+attr_num])
        tar_tokenized[k] = np.array(tar_tokenized[k])
    assert tar_tokenized['input_ids'].shape[0] == tar_tokenized['tar_mask_index'].shape[0]
    assert tar_tokenized['input_ids'].shape[1] == tar_tokenized['tar_mask_index'].shape[1]
    
    print("length of tar_tokenized:", len(tar_tokenized))
    print("shape of tar_tokenized['input_ids']:{}".format(tar_tokenized['input_ids'].shape))
    print("shape of tar_tokenized['tar_mask_index']:{}".format(tar_tokenized['tar_mask_index'].shape))
    return tar_tokenized, attr_num


def get_tokenized_prompt_v4(prompts, tar_words, mask_nums, tokenizer):
    attr_num = len(tar_words) # channel num: m attr words = m channels
    attr_length = len(tar_words[0])
    masked_sent_num = len(mask_nums) # masked sent num
    tar_tokenized = {}
    temp_tar_sen = []
    for i in range(len(prompts)):
        for j in range(attr_length):
            for m in range(attr_num): # m attr words = m channels
                for mask_num in mask_nums:
                    temp_tar_sen.append(tar_words[m][j] + " " + prompts[i] + " " + tokenizer.mask_token * mask_num + ".")

    temp_tar_tokenized = tokenizer(temp_tar_sen, padding=True, truncation=True, return_tensors="np")
    temp_tar_mask_index = np.where(temp_tar_tokenized['input_ids']==tokenizer.mask_token_id)[1]

    # reshape mask index:
    tar_mask_list = []
    span = sum(mask_nums)
    for start in range(0, len(temp_tar_mask_index), span):
        temp = temp_tar_mask_index[start:start+span]
        s = 0
        single_prompt_masks = []
        for num in mask_nums:
            not_mask_num = max(mask_nums) - num
            single_sent_masks = np.concatenate((np.array([0]*not_mask_num), temp[s:s+num]), axis=0)
            single_prompt_masks.append(single_sent_masks)
            s += num
        tar_mask_list += single_prompt_masks
    temp_tar_tokenized['tar_mask_index'] = np.array(tar_mask_list)
    
    # reshape channels: m attr words = m channels
    for k,v in temp_tar_tokenized.items():
        tar_tokenized[k] = []
        for s in range(0, len(v), attr_num*masked_sent_num):
            single_sent_data = v[s:s+attr_num*masked_sent_num]
            reshaped_channels = []
            for i in range(0, len(single_sent_data), masked_sent_num):
                reshaped_channels.append(single_sent_data[i:i+masked_sent_num])
            tar_tokenized[k].append(reshaped_channels)
        tar_tokenized[k] = np.array(tar_tokenized[k])
    
    # check dimension:
    assert tar_tokenized['input_ids'].shape[0] == tar_tokenized['tar_mask_index'].shape[0] # data num
    assert tar_tokenized['input_ids'].shape[0] == attr_length * len(prompts) # data num
    assert tar_tokenized['input_ids'].shape[1] == tar_tokenized['tar_mask_index'].shape[1] # channel num
    assert tar_tokenized['input_ids'].shape[1] == attr_num # channel num
    assert tar_tokenized['input_ids'].shape[2] == tar_tokenized['tar_mask_index'].shape[2] # masked sent num
    assert tar_tokenized['input_ids'].shape[2] == masked_sent_num # masked sent num
    assert tar_tokenized['tar_mask_index'].shape[3] == max(mask_nums) # mask num
    
    print("length of tar_tokenized:", len(tar_tokenized))
    print("shape of tar_tokenized['input_ids']:{}".format(tar_tokenized['input_ids'].shape))
    print("shape of tar_tokenized['tar_mask_index']:{}".format(tar_tokenized['tar_mask_index'].shape))
    return tar_tokenized, attr_num


def send_to_cuda(tar1_tokenized,tar2_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    return tar1_tokenized,tar2_tokenized


if __name__ == "__main__":
    args = parser.parse_args()
    print("args.lr = ", args.lr)

    if args.model_type == 'bert':
        args.model_name_or_path = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForPreTraining.from_pretrained(args.model_name_or_path)

    elif args.model_type == 'albert':
        args.model_name_or_path = "albert-base-v2"
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForPreTraining.from_pretrained(args.model_name_or_path)

    elif args.model_type == 'distilbert':
        args.model_name_or_path = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name_or_path)
        model = DistilBertForMaskedLM.from_pretrained(args.model_name_or_path)

    else:
        raise NotImplementedError("not implemented!")
    
    model.train()
    model.to(device)
    dc_layer = Distribution_Concate_v4(device=device)
    jsd_model = JSD_Train()

    searched_prompts = load_word_list(args.data_path+args.prompts_file)

    male_words_ = load_word_list(args.data_path+"male.txt")
    female_words_ = load_word_list(args.data_path+"female.txt")
    print("length of attribute words before clean:{} and {}".format(len(male_words_), len(female_words_)))
    tar1_words, tar2_words = clean_multi_word_list2(male_words_, female_words_, 4, tokenizer) #remove the OOV words
    print("length of attribute words after clean:{} and {}".format(len(tar1_words), len(tar2_words)))

    if args.finetuning_vocab_file:
        finetuning_vocab_ = load_word_list(args.data_path+args.finetuning_vocab_file)
        print("length of stereotype words before clean:{}".format(len(finetuning_vocab_)))

        if args.debias_type == 'gender-CHN' or args.debias_type == 'race-CHN':
            finetuning_vocab_ = clean_CHN_word_list(finetuning_vocab_, tokenizer) #stereotype words
            print("length of stereotype words after clean:{}".format(len(finetuning_vocab_)))
            ster_words = CHN_tokens_to_ids(finetuning_vocab_, tokenizer)
        
        else:
            finetuning_vocab_ = clean_multi_word_list(finetuning_vocab_, 3, tokenizer) #stereotype words
            print("length of stereotype words after clean:{}".format(len(finetuning_vocab_)))
            ster_words = tokens_to_ids(finetuning_vocab_, tokenizer)
        
        mask_nums = count_mask_nums(finetuning_vocab_, tokenizer)
        print("mask_nums:", mask_nums)
        span = len(mask_nums)
        tar_tokenized, attr_num = get_tokenized_prompt_v4(searched_prompts, [tar1_words, tar2_words], mask_nums, tokenizer)
        args.batch_size = len(mask_nums) * args.batch_size
    else:
        tar_tokenized, attr_num = get_tokenized_prompt_v3(searched_prompts, [tar1_words, tar2_words], tokenizer)
    
    print("batch_size:", args.batch_size)
    tar_sen = attr_dataset(tar_tokenized)
    dataloader = torch.utils.data.DataLoader(dataset=tar_sen,
                                             batch_size=args.batch_size, 
                                             shuffle=True,
                                             drop_last=False)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * len(dataloader)

    lr_scheduler = get_scheduler(
        name="constant", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps) #use "linear" or "constant"
    
    progress_bar = tqdm.tqdm(range(num_training_steps))

    for epoch in range(args.epochs):
        print("------------epoch{}------------".format(epoch))
        for idx, batch in  enumerate(dataloader):
            logits = []
            tar_embedding = []
            for attri_index in range(attr_num):
                input_ids_i = batch['input_ids'][:, attri_index].to(device)
                data_num, sent_num, token_num = input_ids_i.size()
                input_ids_i = input_ids_i.view(data_num * sent_num, token_num) # reshape

                attention_mask_i = batch['attention_mask'][:, attri_index].to(device)
                attention_mask_i = attention_mask_i.view(data_num * sent_num, token_num) # reshape

                if args.model_type == 'bert' or args.model_type == 'albert':
                    token_type_ids_i = batch['token_type_ids'][:, attri_index].to(device)
                    token_type_ids_i = token_type_ids_i.view(data_num * sent_num, token_num) # reshape

                tar_mask_index_i = batch['tar_mask_index'][:, attri_index].to(device)
                data_num, sent_num, token_num = tar_mask_index_i.size()
                tar_mask_index_i = tar_mask_index_i.view(data_num * sent_num, token_num) # reshape

                if args.model_type == 'bert' or args.model_type == 'albert':
                    predictions = model(input_ids=input_ids_i, 
                                       attention_mask=attention_mask_i, 
                                       token_type_ids=token_type_ids_i)
                else:
                    predictions = model(input_ids=input_ids_i, 
                                       attention_mask=attention_mask_i)
                
                if args.finetuning_vocab_file:
                    # reshape logits:
                    logits_matrix = {}
                    for i in range(len(mask_nums)):
                        signle_type_matrix = []
                        for mask_idx in range(-mask_nums[i],0,1):
                            index = tar_mask_index_i[i::span, mask_idx].unsqueeze(1).long()
                            true_mask = torch.ones(len(index), 1).to(device)
                            if args.model_type == 'bert' or args.model_type == 'albert':
                                mask = torch.zeros(len(index), predictions.prediction_logits.shape[1]).to(device)
                                mask = mask.scatter(1, index, true_mask).bool()
                                temp = predictions.prediction_logits[i::span][mask][:,ster_words[mask_nums[i]][:,mask_idx]]
                            else:
                                mask = torch.zeros(len(index), predictions.logits.shape[1]).to(device)
                                mask = mask.scatter(1, index, true_mask).bool()
                                temp = predictions.logits[i::span][mask][:,ster_words[mask_nums[i]][:,mask_idx]]
                            signle_type_matrix.append(temp)
                        
                        signle_type_tensor = torch.empty((len(signle_type_matrix), signle_type_matrix[0].shape[0], signle_type_matrix[0].shape[1]), dtype=torch.float32).to(device)
                        for idx in range(len(signle_type_matrix)):
                            signle_type_tensor[idx] = signle_type_matrix[idx]
                        logits_matrix[mask_nums[i]] = signle_type_tensor
                    logits.append(logits_matrix)
                else:
                    if args.model_type == 'bert' or args.model_type == 'albert':
                        tar_predictions_logits = predictions.prediction_logits[:, tar_mask_index_i]
                    else:
                        tar_predictions_logits = predictions.logits[:, tar_mask_index_i]
                    logits.append(tar_predictions_logits)

                if args.tune_pooling_layer:
                    if args.model_type == 'bert':
                        embedding = model.bert(input_ids=input_ids_i, 
                                               attention_mask=attention_mask_i, 
                                               token_type_ids=token_type_ids_i).pooler_output
                        tar_embedding.append(embedding)
                    elif args.model_type == 'distilbert':
                        embedding = model.distilbert(input_ids=input_ids_i, 
                                                     attention_mask=attention_mask_i).last_hidden_state[:,0]
                        tar_embedding.append(embedding)
                    elif args.model_type == 'albert':
                        embedding = model.albert(input_ids=input_ids_i, 
                                                 attention_mask=attention_mask_i, 
                                                 token_type_ids=token_type_ids_i).pooler_output
                        tar_embedding.append(embedding) 
            
            if args.finetuning_vocab_file:
                jsd_loss = dc_layer(logits, mask_nums)
            else:
                jsd_loss = jsd_model(logits[0], logits[1])
            if args.tune_pooling_layer:
                embed_dist = 1-F.cosine_similarity(tar_embedding[0], tar_embedding[1],dim=1)
                embed_dist = torch.mean(embed_dist)
                print("cosine_similarity:", embed_dist)
                loss = jsd_loss+0.2*embed_dist
            else:
                loss = jsd_loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print('jsd loss {}'.format(jsd_loss))
            progress_bar.update(1)
        model.save_pretrained('model/debiased_model_{}_{}_{}'.format(args.model_name_or_path, args.debias_type, epoch+1))
        tokenizer.save_pretrained('model/debiased_model_{}_{}_{}'.format(args.model_name_or_path, args.debias_type, epoch+1))
