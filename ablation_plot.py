import numpy as np
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
#from sklearn import manifold
import matplotlib.pyplot as plt

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

class GPTEncoder(object):
    def __init__(self, model, device):
        self.device = device
        self.model = model
        self.model.eval()
        self.model.to(device)

    def encode(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        embeddings = torch.nn.functional.normalize(outputs.last_hidden_state[:, -1], p=2, dim=-1) # this operation won't change dimension num
        return embeddings

def get_tokenizer_encoder(model_type, model_path, device=None):
    '''Return BERT tokenizer and encoder based on args. Used in eval_bias.py.'''
    
    if model_type == 'bert':
        if model_path=="original": model_path = "bert-base-uncased"
        print("get tokenizer from {}".format(model_path))
        tokenizer = BertTokenizer.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path, num_labels=2, hidden_dropout_prob=0.3)
        model = BertModel.from_pretrained(model_path, config=config)
        bert_encoder = BertEncoder(model, device)

    elif model_type == 'albert':
        if model_path=="original": model_path = "albert-base-v2"
        print("get tokenizer from {}".format(model_path))
        tokenizer = AlbertTokenizer.from_pretrained(model_path)
        config = AlbertConfig.from_pretrained(model_path, num_labels=2, hidden_dropout_prob=0)
        model = AlbertModel.from_pretrained(model_path, config=config)
        bert_encoder = BertEncoder(model, device)
    
    elif model_type == 'distilbert':
        if model_path=="original": model_path = "distilbert-base-uncased"
        print("get tokenizer from {}".format(model_path))
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        config = DistilBertConfig.from_pretrained(model_path, num_labels=2, hidden_dropout_prob=0)
        model = DistilBertModel.from_pretrained(model_path, config=config)
        bert_encoder = DistilBertEncoder(model, device)

    return tokenizer, bert_encoder


def get_cls_embedding(args, sentlist, tokenizer, model):
    '''get sentence embeddings through MLM.'''
    """
    subject (str): subject word
    sentlist (2 dim list):  a list of sentences which consist of related words under subject word
    """
    sent_array = []
    for i in range(len(sentlist)):
        sent_array += sentlist[i]

    sent_tokenized = tokenizer(sent_array, padding=True, truncation=True, return_tensors="pt")

    sent_tokenized = eval_dataset(sent_tokenized)
    dataloader = DataLoader(dataset=sent_tokenized, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)

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

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    for i in range(len(all_embeddings)):
        all_embeddings[i] /= np.linalg.norm(all_embeddings[i])
    
    reshapeed_emb = []
    for i in range(0, len(all_embeddings), args.SEAT_format_num):
        reshapeed_emb.append(all_embeddings[i:i+args.SEAT_format_num])
    return reshapeed_emb


def compute_similarity(args, model_type, model_path):
    male_words=['fathers','actor','prince','men','gentlemen',
                'sir','brother','his','king','husband',
                'dad','males','sir','him','boyfriend',
                'he','hero','kings','brothers','son',
                'sons','himself','gentleman','his','father',
                'male','man','grandpa','boy','grandfather']

    female_words=['mothers','actress','princess','women','ladies',
                  'madam','sister','her','queen','wife',
                  'mom','females','miss','her','girlfriend',
                  'she','heroine','queens','sisters','daughter',
                  'daughters','herself','lady','hers','mother',
                  'female','woman','grandma','girl','grandmother']
    
    subject_dict = {
        "math":["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"], 
        "art":["art", "poetry", "dance", "literature", "novel", "symphony", "drama", "sculpture", "Shakespeare"], 
        "science":["science", "technology", "physics", "chemistry", "Einstein", "NASA", "experiment", "astronomy"]
    }

    subject_list = []
    for k,v in subject_dict.items():
        subject_list += v
    
    subject_list = load_word_list("data/stereotype_of_{}_filtered.txt".format(args.model_type))

    print("length of subject_list before clean:{}".format(len(subject_list)))

    tokenizer, model = get_tokenizer_encoder(model_type, model_path, DEVICE)
    print("tokenizer: {}".format(tokenizer!=None))

    subject_list = clean_multi_word_list(subject_list, args.token_maxlength, tokenizer)
    print("length of wordlist after clean:{}".format(len(subject_list)))

    # construct sentences list with SEAT format:
    sentlist = SEAT_format(subject_list)
    m_sentlist = SEAT_format(male_words)
    f_sentlist = SEAT_format(female_words)
    
    # get embedding from output of model:
    reshapeed_emb = get_cls_embedding(args, sentlist, tokenizer, model)
    m_reshapeed_emb = get_cls_embedding(args, m_sentlist, tokenizer, model)
    f_reshapeed_emb = get_cls_embedding(args, f_sentlist, tokenizer, model)

    # compute mean embedding of every word:
    reshapeed_emb = np.array(reshapeed_emb)
    m_reshapeed_emb = np.array(m_reshapeed_emb)
    f_reshapeed_emb = np.array(f_reshapeed_emb)

    emb_mean = []
    m_emb_mean = []
    f_emb_mean = []

    for i in range(len(reshapeed_emb)):
        emb_mean.append(np.mean(reshapeed_emb[i], axis=0))
    emb_mean = np.array(emb_mean)

    for i in range(len(m_reshapeed_emb)):
        m_emb_mean.append(np.mean(m_reshapeed_emb[i], axis=0))
    m_emb_mean = np.array(m_emb_mean)
    m_embedding = np.mean(m_emb_mean, axis=0) # to get only 1 embedding

    s_m = []
    for i in range(1, len(m_emb_mean)):
        cos_sim = m_embedding.dot(m_emb_mean[i]) / (np.linalg.norm(m_embedding) * np.linalg.norm(m_emb_mean[i]))
        s_m.append(cos_sim)
    s_m_AVG = np.mean(s_m, axis=0)

    for i in range(len(f_reshapeed_emb)):
        f_emb_mean.append(np.mean(f_reshapeed_emb[i], axis=0))
    f_emb_mean = np.array(f_emb_mean)
    f_embedding = np.mean(f_emb_mean, axis=0) # to get only 1 embedding
    
    s_f = []
    for i in range(1, len(f_emb_mean)):
        cos_sim = f_embedding.dot(f_emb_mean[i]) / (np.linalg.norm(f_embedding) * np.linalg.norm(f_emb_mean[i]))
        s_f.append(cos_sim)
    s_f_AVG = np.mean(s_f, axis=0)

    # compute distances in Euclidean space between raw words in word list and male/female word:
    # (represented by cosine similarity)

    cos_sim_m = []
    for i in range(1, len(emb_mean)):
        cos_sim = m_embedding.dot(emb_mean[i]) / (np.linalg.norm(m_embedding) * np.linalg.norm(emb_mean[i]))
        cos_sim_m.append(cos_sim)
    
    cos_sim_f = []
    for i in range(1, len(emb_mean)):
        cos_sim = f_embedding.dot(emb_mean[i]) / (np.linalg.norm(f_embedding) * np.linalg.norm(emb_mean[i]))
        cos_sim_f.append(cos_sim)
    
    cos_sim_m = np.array(cos_sim_m) / s_m_AVG
    cos_sim_f = np.array(cos_sim_f) / s_f_AVG

    return [cos_sim_m, cos_sim_f]


if __name__ == '__main__':
    args = parse_args()
    figure = plt.figure()

    img_1 = figure.add_subplot(1, 3, 1)
    args.model_type = 'bert'
    args.model_path = 'debiased_model_bert-base-uncased_gender'
    original_x_y = compute_similarity(args, args.model_type, "original")
    debiased_x_y = compute_similarity(args, args.model_type, args.model_path)

    img_1.plot([0.935, 1.005], [0.935, 1.005], alpha=1, color='k', linewidth=1)
    img_1.scatter(original_x_y[0], original_x_y[1], marker='x', s=40, color='darkorange', alpha=0.8, label='original')
    img_1.scatter(debiased_x_y[0], debiased_x_y[1], marker='x', s=40, color='dodgerblue', alpha=0.8, label='debiased')
    img_1.set_xlim((0.93, 1.01))
    img_1.set_ylim((0.93, 1.01))
    img_1.set_xticks(np.arange(0.94, 1.00, 0.02))
    img_1.set_yticks(np.arange(0.94, 1.00, 0.02))
    img_1.tick_params(labelsize=15)
    img_1.set_xlabel('Masculine', fontsize=15)
    img_1.set_ylabel('Feminine', fontsize=15)
    img_1.legend(loc='best', fontsize=15)
    img_1.set_title('BERT', fontsize=15)

    img_2 = figure.add_subplot(1, 3, 2)
    args.model_type = 'albert'
    args.model_path = 'debiased_model_albert-base-v2_gender_4'
    original_x_y = compute_similarity(args, args.model_type, "original")
    debiased_x_y = compute_similarity(args, args.model_type, args.model_path)

    img_2.plot([0.915, 1.005], [0.915, 1.005], alpha=1, color='k', linewidth=1)
    img_2.scatter(original_x_y[0], original_x_y[1], marker='x', s=40, color='darkorange', alpha=0.8, label='original')
    img_2.scatter(debiased_x_y[0], debiased_x_y[1], marker='x', s=40, color='dodgerblue', alpha=0.8, label='debiased')
    img_2.set_xlim((0.91, 1.01))
    img_2.set_ylim((0.91, 1.01))
    img_2.set_xticks(np.arange(0.92, 1.01, 0.02))
    img_2.set_yticks(np.arange(0.92, 1.01, 0.02))
    img_2.tick_params(labelsize=15)
    img_2.set_xlabel('Masculine', fontsize=15)
    img_2.set_ylabel('Feminine', fontsize=15)
    img_2.legend(loc='best', fontsize=15)
    img_2.set_title('ALBERT', fontsize=15)

    img_3 = figure.add_subplot(1, 3, 3)
    args.model_type = 'distilbert'
    args.model_path = 'debiased_model_distilbert-base-uncased_gender_5'
    original_x_y = compute_similarity(args, args.model_type, "original")
    debiased_x_y = compute_similarity(args, args.model_type, args.model_path)

    img_3.plot([0.935, 1.005], [0.935, 1.005], alpha=1, color='k', linewidth=1)
    img_3.scatter(original_x_y[0], original_x_y[1], marker='x', s=40, color='darkorange', alpha=0.8, label='original')
    img_3.scatter(debiased_x_y[0], debiased_x_y[1], marker='x', s=40, color='dodgerblue', alpha=0.8, label='debiased')
    img_3.set_xlim((0.93, 1.01))
    img_3.set_ylim((0.93, 1.01))
    img_3.set_xticks(np.arange(0.94, 1.00, 0.02))
    img_3.set_yticks(np.arange(0.94, 1.00, 0.02))
    img_3.tick_params(labelsize=15)
    img_3.set_xlabel('Masculine', fontsize=15)
    img_3.set_ylabel('Feminine', fontsize=15)
    img_3.legend(loc='best', fontsize=15)
    img_3.set_title('DistilBERT', fontsize=15)

    plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=0.3, hspace=None)
    plt.show()
