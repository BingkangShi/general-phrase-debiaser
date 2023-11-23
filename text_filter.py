import numpy as np
import tqdm
import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, RobertaModel, AlbertModel, BertConfig, AlbertConfig
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader

DATA_DIR = "./wordlists/"
MAX_SEQ_LENGTH = 128
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    print("Loading {}...".format(sent_file))
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
    return all_data

def save_dict_to_json(D, output_eval_file):
    with open(output_eval_file, 'w') as f:
        json.dump(D, f)

def parse_args():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--model_path",
                        type=str,
                        default="bert-base-uncased",
                        help="Path of the model to be evaluated")
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--results_dir", type=str,
                        help="directory for storing results")
    parser.add_argument("--subject", type=str,
                        help="total subject word")
    parser.add_argument("--subject_w", type=str,
                        help="which subject word")
    parser.add_argument("--token_maxlength", type=int, default=3, 
                        help="maxlength of tokens in a phrase")
    parser.add_argument("--topK", type=int, default=20, 
                        help="top K words")
    parser.add_argument("--SEAT_format_num", type=int, default=14, 
                        help="num of SEAT sentences format")
    
    args = parser.parse_args()

    if (args.results_dir == None):
        args.results_dir = DATA_DIR

    args.max_seq_length = 128
    args.batch_size = 512

    return args


def load_word_list(f_path):
    lst = []
    with open(f_path,'r') as f:
        line = f.readline()
        while line:
            lst.append(line.strip())
            line = f.readline()
    return lst


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


def filter_repeat_word(wordlist):
    final_wordlist = []
    for i in range(len(wordlist)):
        if wordlist[i] not in final_wordlist:
            final_wordlist.append(wordlist[i])
    return final_wordlist


def SEAT_format(wordlist):
    choise_an = "n "
    text_templates = [["This is a ", "."], 
                      ["That is a  ", "."], 
                      ["There is a ", "."], 
                      ["Here is a ", "."], 
                      ["The ", " is here."], 
                      ["The ", " is there."], 
                      ["A ", " is a thing."], 
                      ["It is a ", "."], 
                      ["These are ", "s."], 
                      ["Those are ", "s."], 
                      ["They are ", "s."], 
                      ["The ", "s are here."], 
                      ["The ", "s are there."], 
                      ["", "s are things."]]
    
    sentlist = []
    for phrase in wordlist:
        temp = []
        for i in range(len(text_templates)):
            if phrase[0] in ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"] and i in [0, 1, 2, 3, 6, 7]:
                sent = text_templates[i][0][:-1] + choise_an + phrase + text_templates[i][1]
            else:
                sent = text_templates[i][0] + phrase + text_templates[i][1]
            temp.append(sent)
        sentlist.append(temp)
    return sentlist


class eval_dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): # this must be rewriten
        return len(self.data['input_ids'])
    def __getitem__(self, idx): # this must be rewriten
        self.temp_batch = {}
        for k,v in self.data.items():
            self.temp_batch[k] = v[idx]
        return self.temp_batch


class BertEncoder(object):
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(device)

    def encode(self, input_ids, token_type_ids=None, attention_mask=None):
        embeddings = self.model(input_ids, 
                                token_type_ids=token_type_ids, 
                                attention_mask=attention_mask)
        embeddings = torch.nn.functional.normalize(embeddings.pooler_output, p=2, dim=-1) # this operation won't change dimension num
        return embeddings

class DistilBertEncoder(object):
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(device)

    def encode(self, input_ids, attention_mask=None,):
        embeddings = self.model(input_ids, attention_mask=attention_mask)
        embeddings = embeddings.last_hidden_state[:,0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

def get_tokenizer_encoder(args, device=None):
    '''Return BERT tokenizer and encoder based on args. Used in eval_bias.py.'''
    
    if args.model_type == 'bert':
        args.model_path = "bert-base-uncased"
        print("get tokenizer from {}".format(args.model_path))
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        config = BertConfig.from_pretrained(args.model_path, num_labels=2, hidden_dropout_prob=0.3)
        model = BertModel.from_pretrained(args.model_path, config=config)
        bert_encoder = BertEncoder(model, device)

    elif args.model_type == 'albert':
        args.model_path = "albert-base-v2"
        print("get tokenizer from {}".format(args.model_path))
        tokenizer = AlbertTokenizer.from_pretrained(args.model_path)
        config = AlbertConfig.from_pretrained(args.model_path, num_labels=2, hidden_dropout_prob=0)
        model = AlbertModel.from_pretrained(args.model_path, config=config)
        bert_encoder = BertEncoder(model, device)
    
    elif args.model_type == 'distilbert':
        args.model_path = "distilbert-base-uncased"
        print("get tokenizer from {}".format(args.model_path))
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_path)
        config = DistilBertConfig.from_pretrained(args.model_path, num_labels=2, hidden_dropout_prob=0)
        model = DistilBertModel.from_pretrained(args.model_path, config=config)
        bert_encoder = DistilBertEncoder(model, device)

    return tokenizer, bert_encoder


def get_cls_embedding(args, sentlist, tokenizer, model):
    '''get sentence embeddings through MLM.'''
    """
    subject (str): subject word
    sentlist (2 dim list):  a list of sentences which consist of related words under subject word
    """
    words_num = len(sentlist)
    s_num = len(sentlist[0])
    sent_array = []
    for i in range(len(sentlist)):
        sent_array += sentlist[i]

    sent_tokenized = tokenizer(sent_array, padding=True, truncation=True, return_tensors="pt")

    sent_tokenized = eval_dataset(sent_tokenized)
    dataloader = DataLoader(dataset=sent_tokenized, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)
    
    num_training_steps = len(dataloader)

    progress_bar = tqdm.tqdm(range(num_training_steps))
    all_embeddings = []
    for idx, batch in  enumerate(dataloader):
        input_ids_i = batch['input_ids'].to(DEVICE)
        attention_mask_i = batch['attention_mask'].to(DEVICE)
        if args.model_type == 'bert' or args.model_type == 'albert':
            token_type_ids_i = batch['token_type_ids'].to(DEVICE)
            embeddings = model.encode(input_ids=input_ids_i, 
                                      attention_mask=attention_mask_i, 
                                      token_type_ids=token_type_ids_i)
        else:
            embeddings = model.encode(input_ids=input_ids_i, 
                                      attention_mask=attention_mask_i)
        embeddings = embeddings.cpu().detach().numpy()
        all_embeddings.append(embeddings)
        progress_bar.update(1)
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    for i in range(len(all_embeddings)):
        all_embeddings[i] /= np.linalg.norm(all_embeddings[i])
    
    reshapeed_emb = []
    for i in range(0, len(all_embeddings), args.SEAT_format_num):
        reshapeed_emb.append(all_embeddings[i:i+args.SEAT_format_num])
    return reshapeed_emb


def text_filter(args):
    # read file and construct list:
    filename = "wordlist-{}.txt".format(args.subject)
    sent_file = os.path.join(DATA_DIR, filename)
    wordlist = load_word_list(sent_file)

    print("length of wordlist before clean:{}".format(len(wordlist)))

    tokenizer, model = get_tokenizer_encoder(args, DEVICE)
    print("tokenizer: {}".format(tokenizer!=None))

    wordlist = clean_multi_word_list(wordlist, args.token_maxlength, tokenizer)
    print("length of wordlist after clean:{}".format(len(wordlist)))
    subject_w_pos = wordlist.index(args.subject_w)

    # construct sentences list with SEAT format:
    sentlist = SEAT_format(wordlist)

    # get embedding from output of model:
    reshapeed_emb = get_cls_embedding(args, sentlist, tokenizer, model)

    # compute distances between raw words and word cluster:
    # 1.compute mean value of 14 embeddings:
    reshapeed_emb = np.array(reshapeed_emb)
    emb_mean = []
    for i in range(len(reshapeed_emb)):
        emb_mean.append(np.mean(reshapeed_emb[i], axis=0))
    emb_mean = np.array(emb_mean)

    # 2.compute distances in Euclidean space between raw words in word list and word cluster:
    # (represented by cosine similarity)
    cluster_center = emb_mean[subject_w_pos]
    cos_sim_unsorted = []
    for i in range(1, len(emb_mean)):
        cos_sim = cluster_center.dot(emb_mean[i]) / (np.linalg.norm(cluster_center) * np.linalg.norm(emb_mean[i]))
        cos_sim_unsorted.append(cos_sim)
        
    # 3.sort cosine similarities of raw words in word list:
    cos_sim_sorted = sorted(np.array(cos_sim_unsorted), reverse=True)
    sorted_index = np.array(cos_sim_unsorted).argsort()
    
    # 4.output subject word and selected topK words:
    print("selected topK words:", args.subject_w, np.array(wordlist)[sorted_index[-1:-args.topK-1:-1]])

    # 5.store topK words:
    result = []
    write_file = 'selected_top{}_{}.txt'.format(args.topK, args.subject_w)
    write_file = os.path.join(DATA_DIR, args.model_type, write_file)
    f = open(write_file,'w')
    for phrase in np.array(wordlist)[sorted_index[-1:-args.topK-1:-1]]:
        result.append(phrase)
        f.write(phrase)
        f.write("\n")
    f.close()
    return result


if __name__ == '__main__':
    args = parse_args()

    subject_dict = {
        "math":["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"], 
        "art":["art", "poetry", "dance", "literature", "novel", "symphony", "drama", "sculpture", "Shakespeare"], 
        "science":["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"]
    }
    
    num_steps = 0
    for v in subject_dict.values():
        num_steps += len(v)
    filter_progress_bar = tqdm.tqdm(range(num_steps))

    filter_result = []
    for k,v in subject_dict.items():
        args.subject = k
        for subject_word in v:
            args.subject_w = subject_word
            result = text_filter(args)
            filter_result.append(result)
            filter_progress_bar.update(1)
    
    write_file = 'stereotype_of_{}.txt'.format(args.model_type)
    write_file = os.path.join(DATA_DIR, args.model_type, write_file)
    f = open(write_file,'w')
    for result in filter_result:
        for phrase in result:
            f.write(phrase)
            f.write("\n")
    f.close()

    final = []
    for result in filter_result:
        final += result
            
    final = filter_repeat_word(final)
    
    write_file = 'stereotype_of_{}_filtered.txt'.format(args.model_type)
    write_file = os.path.join(DATA_DIR, args.model_type, write_file)
    f = open(write_file,'w')
    for w in final:
        f.write(w)
        f.write("\n")
    f.close()

