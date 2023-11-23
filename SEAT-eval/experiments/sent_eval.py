from __future__ import absolute_import, division, print_function
import numpy as np
import json
import os
import logging
import argparse
import torch

import weat
from seat_utils import get_encodings, compute_gender_dir, get_tokenizer_encoder
from experiments.def_sent_utils import get_def_pairs

logger = logging.getLogger(__name__)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', evel = logging.INFO)

DATA_DIR = "../gender_tests/"
MAX_SEQ_LENGTH = 128
DEVICE = torch.device("cuda") if torch.cuda.is_available() else None


def load_json(sent_file):
    ''' Load from json. We expect a certain format later, so do some post processing '''
    logger.info("Loading %s..." % sent_file)
    all_data = json.load(open(sent_file, 'r'))
    data = {}
    for k, v in all_data.items():
        examples = v["examples"]
        data[k] = examples
    return all_data  # data

def save_dict_to_json(D, output_eval_file):
    with open(output_eval_file, 'w') as f:
        json.dump(D, f)

def parse_args():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        type=str,
                        default="bert-base-uncased",
                        help="Path of the model to be evaluated")
    parser.add_argument("--debias",
                        action='store_true',
                        help="Whether to debias.")
    parser.add_argument("--equalize",
                        action='store_true',
                        help="Whether to equalize.")
    parser.add_argument("--def_pairs_name", default="all", type=str,
                        help="Name of definitional sentence pairs.")
    parser.add_argument("--model", "-m", type=str, default="dummy")
    parser.add_argument("--model_type", type=str, default="bert")
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--results_dir", type=str,
                        help="directory for storing results")
    parser.add_argument("--encode_only", action='store_true')
    parser.add_argument("--num_dimension", "-k", type=int, default=1,
                        help="dimensionality of bias subspace")
    
    args = parser.parse_args()

    if (args.output_name == None):
        args.output_name = args.def_pairs_name if args.debias else "biased"
    print("outputname: {}".format(args.output_name))
    if (args.results_dir == None):
        args.results_dir = os.path.join("results", args.model)
    args.do_lower_case = True
    #args.cache_dir = "local_TRANSFORMERS_CACHE"
    args.local_rank = -1
    args.max_seq_length = 128
    args.eval_batch_size = 32 #8
    args.n_samples = 100000
    args.parametric = True
    args.tune_bert = False
    args.normalize = True

    # word embeddings
    args.word_model = 'fasttext-wiki-news-subwords-300'
    wedata_path = 'my_debiaswe/data'
    args.wedata_path = wedata_path
    args.definitional_filename = os.path.join(wedata_path, 'definitional_pairs.json')
    args.equalize_filename = os.path.join(wedata_path, 'equalize_pairs.json')
    args.gendered_words_filename = os.path.join(wedata_path, 'gender_specific_complete.json')

    return args


def evaluate(args, def_pairs, word_level=False):
    '''Evaluate bias level with given definitional sentence pairs.'''
    results_path = os.path.join(args.results_dir, args.output_name)
    if (not os.path.exists(args.results_dir)): os.makedirs(args.results_dir)

    results = []
    all_tests_dict = dict()
    abs_esizes = []

    tokenizer, encoder = get_tokenizer_encoder(args, DEVICE)
    print("tokenizer: {}".format(tokenizer==None))

    gender_subspace = None
    if (args.debias):
        gender_subspace = compute_gender_dir(DEVICE, tokenizer, encoder, def_pairs, 
            args.max_seq_length, k=args.num_dimension, load=True, task=args.model, word_level=word_level, keepdims=True)
        logger.info("Computed (gender) bias direction")

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    specific_set = set(gender_specific_words)

    for test_id in ['6', '6b', '7', '7b', '8', '8b']:
        filename = "sent-weat{}.jsonl".format(test_id)
        sent_file = os.path.join(DATA_DIR, filename)
        data = load_json(sent_file)

        encs = get_encodings(args, data, tokenizer, encoder, gender_subspace, 
            DEVICE, word_level=word_level, specific_set=specific_set)
        '''
        encs: targ1, targ2, attr1, attr2
                 -> category
                 -> encs
                 	-> (id1, sent1_emb), (id2, sent2_emb), ...
        '''

        esize, pval = weat.run_test(encs, n_samples=args.n_samples, parametric=args.parametric)
        abs_esizes.append(abs(esize))

        result = "{}: esize={} pval={}".format(filename, esize, pval)
        print(filename, result)
        results.append(result)
        test_results = {"esize": esize, "pval": pval}
        
        all_tests_dict[filename] = test_results
    avg_absesize = np.mean(np.array(abs_esizes))
    print("Averge of Absolute esize: {}".format(avg_absesize))
    all_tests_dict['avg_absesize'] = avg_absesize
	
    # print and save results
    for result in results: logger.info(result)
    save_dict_to_json(all_tests_dict, results_path)

    return


if __name__ == '__main__':
    '''
    Evaluate bias level using definitional sentences specified in args.
    '''
    args = parse_args()
    def_pairs = get_def_pairs(args.def_pairs_name)
    evaluate(args, def_pairs)
