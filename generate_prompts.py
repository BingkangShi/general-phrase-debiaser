import time
import tqdm
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from transformers import BertTokenizer,BertForMaskedLM
from transformers import RobertaTokenizer,RobertaForMaskedLM
from transformers import AlbertTokenizer,AlbertForMaskedLM
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForMaskedLM
from multiprocessing import Pool

global_dict = {}

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
    help="Choose from ['bert','albert','distilbert']",
)

parser.add_argument(
    "--vocab_file",
    default='data/wiki_words_5000.txt',
    type=str,
    help="Path to the file that stores the vocabulary of the prompts",
)

parser.add_argument(
    "--batch_size",
    default=1000,
    type=int,
    help="batch size of the data fed into the model",
)

parser.add_argument(
    "--PL",
    default=5,
    type=int,
    help="maximun length of the generated prompts",
)

parser.add_argument(
    "--K",
    default=100,
    type=int,
    help="top K prompts to be selected in the beam search",
)

parser.add_argument(
    "--from_which_PL",
    default=1,
    type=int,
    help="start from which PL",
)

parser.add_argument(
    "--temp_prompt_file",
    default='data/prompts_bert-base_gender',
    type=str,
    help="Path to the prompt file that stores the prompts of the last iter",
)


def get_tokenized_ith_prompt_v3(prompts, tar1_word, tar2_word, mask_nums, tokenizer_):
    tar1_sen_i = []
    tar2_sen_i = []
    for i in range(len(prompts)):
        temp1 = tar1_word + ' ' + prompts[i]
        temp2 = tar2_word + ' ' + prompts[i]
        for mask_num in mask_nums:
            mask = (' ' + tokenizer_.mask_token)*mask_num
            sent1 = temp1 + mask
            sent2 = temp2 + mask
            tar1_sen_i.append(sent1)
            tar2_sen_i.append(sent2)
    tar_sen_i = [tar1_sen_i, tar2_sen_i]
    return tar_sen_i


def attribute_tokenizer_v3(tar_sen_i, copied_mask_nums, tokenizer_):
    word_list_length = len(tar_sen_i[0])
    real_word_list_length = word_list_length//len(copied_mask_nums)
    word_list_num = len(tar_sen_i)

    # Concate tar_sen_i:
    temp = []
    for j in range(word_list_num):
        temp.extend(tar_sen_i[j])
    tar_sen_i = temp.copy()
    del temp

    # Encode:
    tar_sen_i = tokenizer_(tar_sen_i, padding=True, truncation=True, return_tensors="pt")

    # Mask reshape:
    tar_mask_array = np.where(tar_sen_i['input_ids'].numpy()==tokenizer_.mask_token_id)[-1]
    
    # reshape mask index:
    tar_mask_list = []
    span = sum(copied_mask_nums)

    for start in range(0, len(tar_mask_array), span):
        temp = tar_mask_array[start:start+span]
        s = 0
        single_prompt_masks = []
        for num in copied_mask_nums:
            not_mask_num = max(copied_mask_nums) - num
            single_sent_masks = np.concatenate((np.array([0]*not_mask_num), temp[s:s+num]), axis=0)
            single_prompt_masks.append(single_sent_masks)
            s += num
        tar_mask_list += single_prompt_masks
    tar_mask_array = np.array(tar_mask_list)
    tar_mask_array = torch.from_numpy(tar_mask_array)

    # Resize "tar_sen_i" and "tar_mask_array":
    tar_sen_i['input_ids'] = tar_sen_i['input_ids'].unsqueeze(1)
    tar_sen_i['attention_mask'] = tar_sen_i['attention_mask'].unsqueeze(1)
    if args.model_type == 'bert' or args.model_type == 'albert':
        tar_sen_i['token_type_ids'] = tar_sen_i['token_type_ids'].unsqueeze(1)
    tar_mask_array = tar_mask_array.unsqueeze(1)

    temp1 = tar_sen_i['input_ids'].numpy()
    temp2 = tar_sen_i['attention_mask'].numpy()
    if args.model_type == 'bert' or args.model_type == 'albert':
        temp3 = tar_sen_i['token_type_ids'].numpy()
    tar_mask_array = tar_mask_array.numpy()

    for j in range(0, word_list_num):
        if j == 0:
            input_ids_i = temp1[:word_list_length].copy()
            attention_mask_i = temp2[:word_list_length].copy()
            if args.model_type == 'bert' or args.model_type == 'albert':
                token_type_ids_i = temp3[:word_list_length].copy()
            tar_mask_index_i = tar_mask_array[:word_list_length].copy()
        else:
            input_ids_i = np.concatenate((input_ids_i, temp1[j*word_list_length:(j+1)*word_list_length]), axis=1)
            attention_mask_i = np.concatenate((attention_mask_i, temp2[j*word_list_length:(j+1)*word_list_length]), axis=1)
            if args.model_type == 'bert' or args.model_type == 'albert':
                token_type_ids_i = np.concatenate((token_type_ids_i, temp3[j*word_list_length:(j+1)*word_list_length]), axis=1)
            tar_mask_index_i = np.concatenate((tar_mask_index_i, tar_mask_array[j*word_list_length:(j+1)*word_list_length]), axis=1)

    assert tar_mask_index_i.shape[0]==input_ids_i.shape[0]

    tar_sen_i['input_ids'] = torch.from_numpy(input_ids_i)
    tar_sen_i['attention_mask'] = torch.from_numpy(attention_mask_i)
    if args.model_type == 'bert' or args.model_type == 'albert':
        tar_sen_i['token_type_ids'] = torch.from_numpy(token_type_ids_i)
    tar_sen_i['tar_mask_index'] = torch.from_numpy(tar_mask_index_i)

    return tar_sen_i


def get_prompt_and_attribute_tokenize(copied_prompts, copied_mask_nums, tar1_word, tar2_word, tar1_word_idx, model_type):
    if model_type == 'bert' or model_type == 'bert-CHN' or model_type == 'roberta-CHN' or model_type == 'albert-CHN':
        tokenizer_ = BertTokenizer.from_pretrained(args.model_name_or_path)
    elif model_type == 'albert':
        tokenizer_ = AlbertTokenizer.from_pretrained(args.model_name_or_path)
    elif model_type == 'distilbert':
        tokenizer_ = DistilBertTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")
    
    tar_sen = get_tokenized_ith_prompt_v3(copied_prompts, tar1_word, tar2_word, copied_mask_nums, tokenizer_)
    tar_sen = attribute_tokenizer_v3(tar_sen, copied_mask_nums, tokenizer_)

    return [tar1_word_idx, tar_sen]


# callback function, but relative ordering of writes between processes is not guaranteed:
def setcallback_dict(msg):
    global global_dict
    global_dict[msg[0]] = msg[1]


def tokenize_prompt_data(prompts, mask_nums, tar1_words, tar2_words, model_type, num_processors=30, offset=0):
    print('-----------------Mutiprocess Start------------------')
    
    num_processors = len(tar1_words)
    print('num of processors:{}\n'.format(num_processors))
    p = Pool(num_processors)

    print('Start search with multiprocess...')
    start_multi = time.time()
    for i in range(num_processors):
        p.apply_async(get_prompt_and_attribute_tokenize, args=(prompts.copy(), 
                                                               mask_nums.copy(), 
                                                               tar1_words[i], 
                                                               tar2_words[i], 
                                                               i+offset, 
                                                               model_type,), callback=setcallback_dict)
    p.close()
    p.join()

    end_multi = time.time()
    print('Mutiprocess Time Cost:{}s\n'.format(end_multi-start_multi))
    print('-----------------Mutiprocess Over------------------')


def get_JSD(tar_sen_i, model, ster_words, mask_nums):
    jsd_list=[]

    # DataLoader:
    dataloader = torch.utils.data.DataLoader(dataset=tar_sen_i,
                                             batch_size=args.batch_size*len(mask_nums),
                                             shuffle=False,
                                             drop_last=False)
    span = len(mask_nums)
    for idx, batch in  enumerate(dataloader):
        input_ids_i = batch['input_ids'].to(device)
        attention_mask_i = batch['attention_mask'].to(device)
        if args.model_type == 'bert' or args.model_type == 'albert':
            token_type_ids_i = batch['token_type_ids'].to(device)
        tar_mask_index_i = batch['tar_mask_index'].to(device)

        with torch.no_grad():
            predictions_logits = []
            for attri_index in range(input_ids_i.size(1)):
                if args.model_type == 'bert' or args.model_type == 'albert':
                    predictions = model(input_ids=input_ids_i[:, attri_index], 
                                        attention_mask=attention_mask_i[:, attri_index], 
                                        token_type_ids=token_type_ids_i[:, attri_index])
                else:
                    predictions = model(input_ids=input_ids_i[:, attri_index], 
                                        attention_mask=attention_mask_i[:, attri_index])
                
                logits_matrix = {}
                for i in range(len(mask_nums)):
                    signle_type_matrix = []
                    for mask_idx in range(-mask_nums[i],0,1):
                        index = tar_mask_index_i[i::span, attri_index, mask_idx].unsqueeze(1).long()
                        true_mask = torch.ones(len(index), 1).to(device)
                        mask = torch.zeros(len(index), predictions.logits.shape[1]).to(device)
                        mask = mask.scatter(1, index, true_mask).bool()
                        temp = predictions.logits[i::span][mask][:,ster_words[mask_nums[i]][:,mask_idx]]
                        signle_type_matrix.append(temp)
                    
                    signle_type_tensor = torch.empty((len(signle_type_matrix), 
                                                      signle_type_matrix[0].shape[0], 
                                                      signle_type_matrix[0].shape[1]), dtype=torch.float32).to(device)
                    
                    for idx in range(len(signle_type_matrix)):
                        signle_type_tensor[idx] = signle_type_matrix[idx]
                    
                    logits_matrix[mask_nums[i]] = signle_type_tensor
                
                predictions_logits.append(logits_matrix)
            
            jsd = dc_layer(predictions_logits, mask_nums)
            jsd_np = jsd.detach().cpu().numpy()
            jsd_list += list(jsd_np)
    
    torch.cuda.empty_cache()

    return jsd_list


def get_prompt_jsd(tar1_words, tar2_words, prompts, mask_nums, model, ster_words):
    jsd_word_list = []
    assert len(tar1_words)==len(tar2_words)

    global global_dict
    
    num = len(tar1_words)//2
    multi_p_time = len(tar1_words)//num
    for idx in range(multi_p_time): # avoid multiprocessing bug!!!
        tokenize_prompt_data(prompts, mask_nums, tar1_words[idx*num:(idx+1)*num], tar2_words[idx*num:(idx+1)*num], args.model_type, offset=idx*num)

        for i in tqdm.tqdm(range(idx*num, idx*num+len(tar1_words[idx*num:(idx+1)*num]))):
            tar_sen_i = attr_dataset(global_dict[i])
            jsd_list = get_JSD(tar_sen_i, model, ster_words, mask_nums)
            jsd_word_list.append(jsd_list)
            print("got the jsd for the word",tar1_words[i])
    
        global_dict = {} # clean global parameters

    jsd_word_list = np.array(jsd_word_list)

    print("jsd for every prompt, every word has shape",jsd_word_list.shape)
    assert jsd_word_list.shape == (len(tar1_words),len(prompts))

    return np.mean(jsd_word_list, axis=0)


if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.model_type == 'bert':
        args.model_name_or_path = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForMaskedLM.from_pretrained(args.model_name_or_path)

    elif args.model_type == 'albert':
        args.model_name_or_path = "albert-base-v2"
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForMaskedLM.from_pretrained(args.model_name_or_path)
    
    elif args.model_type == 'distilbert':
        args.model_name_or_path = "distilbert-base-uncased"
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name_or_path)
        model = DistilBertForMaskedLM.from_pretrained(args.model_name_or_path)
    
    else:
        raise NotImplementedError("not implemented!")
    
    model.eval()
    model.to(device)
    
    dc_layer = Distribution_Concate_v3(reduction='none', device=device)
    jsd_model = JSD(reduction='none')

    male_words=['fathers','actor','prince','men','gentlemen','sir','brother','his','king','husband','dad','males','sir','him','boyfriend','he','hero', 'kings','brothers','son','sons','himself','gentleman','his','father','male','man','grandpa','boy','grandfather']
    female_words=['mothers','actress','princess','women','ladies','madam','sister','her','queen','wife','mom','females','miss','her','girlfriend', 'she','heroine','queens','sisters','daughter','daughters','herself','lady','hers','mother','female','woman','grandma','girl','grandmother']

    ster_words_ = load_word_list("data/stereotype_of_{}_filtered.txt".format(args.model_type))
    #ster_words_ = load_word_list("data/stereotype_raw.txt")
    print("length of stereotype words before clean:{}".format(len(ster_words_)))
    ster_words_ = clean_multi_word_list(ster_words_, 3, tokenizer) # stereotype words
    print("length of stereotype words after clean:{}".format(len(ster_words_)))
    ster_words = tokens_to_ids(ster_words_, tokenizer)

    vocab = load_wiki_word_list(args.vocab_file)
    print("length of vocab before clean:{}".format(len(vocab)))
    vocab = clean_word_list_v2(vocab, tokenizer) # vocabulary in prompts
    print("length of vocab after clean:{}".format(len(vocab)))
    
    current_prompts = vocab # vocabulary in prompts
    prompts_cache_original = []
    prompts_cache = []
    jsd_topk = []
    jsd_selected = []
    mask_nums = count_mask_nums(ster_words_, tokenizer)
    print("mask_nums:", mask_nums)
    
    f=open('data/prompts_{}_{}'.format(args.model_type, args.debias_type),'w')

    if args.from_which_PL > 1:
        args.PL = args.PL - args.from_which_PL + 1
        top_k_prompts = load_word_list(args.temp_prompt_file)
        top_k_prompts = top_k_prompts[len(top_k_prompts)-100:len(top_k_prompts)]
        new_prompts = []
        for tkp in top_k_prompts:
            for v in vocab:
                new_prompts.append(tkp+v)
        current_prompts = new_prompts
        print(len(current_prompts))

    for m in range(args.PL):
        current_prompts_jsd = get_prompt_jsd(male_words, female_words, current_prompts, mask_nums, model, ster_words)
        
        sorted_jsd = np.sort(current_prompts_jsd)[::-1]
        sorted_prompts = np.array(current_prompts)[np.argsort(current_prompts_jsd)[::-1]]
        jsd_topk.append(np.mean(sorted_jsd[:args.K])) # get total jsd(top k) in this search loop 
        
        # filter:
        selected = []
        print(sorted_prompts[0])
        print("len of sorted_jsd:", len(sorted_jsd))
        print("len of sorted_prompts:", len(sorted_prompts))

        if len(tokenizer(sorted_prompts[0], add_special_tokens=False).input_ids) >= 1:
            top_k_prompts = sorted_prompts[:args.K].tolist()
            prompts_cache_original += sorted_prompts[:2*args.K].tolist()
            jsd_selected.append(jsd_topk[m]) # get total jsd(actual select) in this search loop 

        else:
            top_k_prompts = []
            num_prompts = 0
            for p_i in range(len(sorted_prompts)):

                # split prompt into list form:
                p = sorted_prompts[p_i]
                p_wordslist = p.split()
                p_length = len(p_wordslist)
                
                # test if part of prompt in prompt cache:
                if p_wordslist[-1] not in prompts_cache:
                    top_k_prompts.append(p)
                    selected.append(p_i) # get index for count total jsd in this search loop 
                    num_prompts += 1

                # break loop condition:
                if num_prompts >= 100:
                    break
            print("len of sorted_jsd:", len(sorted_jsd[selected]))
            jsd_selected.append(np.mean(sorted_jsd[selected]))
        
        print(top_k_prompts)

        # out put mean jsd score:
        print("jsd_topk:", jsd_topk)
        print("jsd_selected:", jsd_selected)
        
        for p in top_k_prompts:
            f.write(p)
            f.write("\n")

        new_prompts = []
        for tkp in top_k_prompts:
            for v in vocab:
                new_prompts.append(tkp+" "+v)
        
        prompts_cache = prompts_cache + top_k_prompts
        print("prompts_cache:", prompts_cache)
        print("len of prompts_cache:", len(prompts_cache))

        current_prompts = new_prompts
        del current_prompts_jsd, top_k_prompts

        torch.cuda.empty_cache()

        print("search space size:",len(current_prompts))
    f.close()

    f=open('data/2prompts_{}_{}'.format(args.model_type, args.debias_type),'w')
    for p in prompts_cache_original:
        f.write(p)
        f.write("\n")
    f.close()
    print("length of prompts_cache_original:", prompts_cache_original)
    