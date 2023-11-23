# coding=utf-8
import logging
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertModel, RobertaModel, AlbertModel, BertConfig, AlbertConfig
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import (DataLoader, TensorDataset)
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from transformers import BertModel, BertConfig
from transformers import BertTokenizer
from my_debiaswe import my_we

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DualInputFeatures(object):
	"""A single set of dual features of data."""

	def __init__(self, input_ids_a, input_ids_b, mask_a, mask_b, segments_a, segments_b):
		self.input_ids_a = input_ids_a
		self.input_ids_b = input_ids_b
		self.mask_a = mask_a
		self.mask_b = mask_b
		self.segments_a = segments_a
		self.segments_b = segments_b


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, tokens, input_ids, input_mask, segment_ids, label_id):
		self.tokens = tokens
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


class BertEncoder(object):
	def __init__(self, model, device):
		self.device = device
		self.bert = model

	def encode(self, input_ids, token_type_ids=None, attention_mask=None):
		self.bert.eval()
		embeddings = self.bert(input_ids, token_type_ids=token_type_ids, 
			attention_mask=attention_mask)
		
		embeddings = torch.nn.functional.normalize(embeddings.pooler_output, p=2, dim=-1) # this operation won't change dimension num
		return embeddings

class DistilBertEncoder(object):
	def __init__(self, model, device):
		self.device = device
		self.bert = model

	def encode(self, input_ids, attention_mask=None,):
		self.bert.eval()
		embeddings = self.bert(input_ids, attention_mask=attention_mask)
		embeddings = embeddings.last_hidden_state[:,0]
		embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1) # this operation won't change dimension num
		return embeddings
    
def extract_embeddings(args, encoder, tokenizer, examples, max_seq_length, device, 
        label_list, output_mode):
    '''Encode examples into BERT embeddings in batches.'''
    features = convert_examples_to_dualfeatures(
        examples, label_list, max_seq_length, tokenizer, output_mode)
    all_inputs_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
    all_mask_a = torch.tensor([f.mask_a for f in features], dtype=torch.long)
    all_segments_a = torch.tensor([f.segments_a for f in features], dtype=torch.long)

    data = TensorDataset(all_inputs_a, all_mask_a, all_segments_a)
    dataloader = DataLoader(data, batch_size=32, shuffle=False)
    all_embeddings = []
    for step, batch in enumerate(tqdm(dataloader)):
        inputs_a, mask_a, segments_a = batch
        if (device != None):
            inputs_a = inputs_a.to(device)
            mask_a = mask_a.to(device)
            segments_a = segments_a.to(device)
        if args.model_type == 'distilbert':
            embeddings = encoder.encode(input_ids=inputs_a, attention_mask=mask_a)
        else:
            embeddings = encoder.encode(input_ids=inputs_a, token_type_ids=segments_a, attention_mask=mask_a)
        embeddings = embeddings.cpu().detach().numpy()
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def extract_embeddings_pair(bert_encoder, tokenizer, examples, max_seq_length, device, 
		load, task, label_list, output_mode, norm, word_level=False):
	'''Encode paired examples into BERT embeddings in batches.
	   Used in the computation of gender bias direction.
	   Save computed embeddings under saved_embs/.
	'''
	emb_loc_a = 'saved_embs/num%d_a_%s.pkl' % (len(examples), task)
	emb_loc_b = 'saved_embs/num%d_b_%s.pkl' % (len(examples), task)
	if os.path.isfile(emb_loc_a) and os.path.isfile(emb_loc_b) and load:
		with open(emb_loc_a, 'rb') as f:
			all_embeddings_a = pickle.load(f)
		with open(emb_loc_b, 'rb') as f:
			all_embeddings_b = pickle.load(f)
		print ('preprocessed embeddings loaded from:', emb_loc_a, emb_loc_b)
	else:
		features = convert_examples_to_dualfeatures(
			examples, label_list, max_seq_length, tokenizer, output_mode)
		all_inputs_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
		all_mask_a = torch.tensor([f.mask_a for f in features], dtype=torch.long)
		all_segments_a = torch.tensor([f.segments_a for f in features], dtype=torch.long)
		all_inputs_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
		all_mask_b = torch.tensor([f.mask_b for f in features], dtype=torch.long)
		all_segments_b = torch.tensor([f.segments_b for f in features], dtype=torch.long)

		data = TensorDataset(all_inputs_a, all_inputs_b, all_mask_a, all_mask_b, all_segments_a, all_segments_b)
		dataloader = DataLoader(data, batch_size=32, shuffle=False)
		all_embeddings_a = []
		all_embeddings_b = []
		for step, batch in enumerate(tqdm(dataloader)):
			inputs_a, inputs_b, mask_a, mask_b, segments_a, segments_b = batch
			if (device != None):
				inputs_a = inputs_a.to(device)
				mask_a = mask_a.to(device)
				segments_a = segments_a.to(device)
				inputs_b = inputs_b.to(device)
				mask_b = mask_b.to(device)
				segments_b = segments_b.to(device)
			embeddings_a = bert_encoder.encode(input_ids=inputs_a, token_type_ids=segments_a, attention_mask=mask_a)
			embeddings_b = bert_encoder.encode(input_ids=inputs_b, token_type_ids=segments_b, attention_mask=mask_b)
			embeddings_a /= torch.norm(embeddings_a, dim=-1, keepdim=True)
			embeddings_b /= torch.norm(embeddings_b, dim=-1, keepdim=True)
			if not torch.isnan(embeddings_a).any() and not torch.isnan(embeddings_b).any():
				embeddings_a = embeddings_a.cpu().detach().numpy()
				embeddings_b = embeddings_b.cpu().detach().numpy()
				all_embeddings_a.append(embeddings_a)
				all_embeddings_b.append(embeddings_b)

		all_embeddings_a = np.concatenate(all_embeddings_a, axis=0)
		all_embeddings_b = np.concatenate(all_embeddings_b, axis=0)

		with open(emb_loc_a, 'wb') as f:
			pickle.dump(all_embeddings_a, f)
		with open(emb_loc_b, 'wb') as f:
			pickle.dump(all_embeddings_b, f)

		print ('preprocessed embeddings saved to:', emb_loc_a, emb_loc_b)

	means = (all_embeddings_a + all_embeddings_b) / 2.0
	all_embeddings_a -= means
	all_embeddings_b -= means
	all_embeddings = np.concatenate([all_embeddings_a, all_embeddings_b], axis=0)

	return all_embeddings


def doPCA(matrix, num_components=10):
    pca = PCA(n_components=num_components, svd_solver="auto")
    pca.fit(matrix) # Produce different results each time...
    return pca


def get_def_examples(def_pairs):
    '''Construct definitional examples from definitional pairs.'''
    def_examples = []
    for group_id in def_pairs:
        def_group = def_pairs[group_id]
        f_sents = def_group['f']
        m_sents = def_group['m']
        for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
            def_examples.append(InputExample(guid='{}-{}'.format(group_id, sent_id), 
                text_a=sent_a, text_b=sent_b, label=None))
    return def_examples


def compute_gender_dir(device, tokenizer, bert_encoder, def_pairs, max_seq_length, k, load, task, word_level=False, keepdims=True):
    '''Compute gender bias direction from definitional sentence pairs.'''
    def_examples = get_def_examples(def_pairs) # 1D list where 2i and 2i+1 are a pair

    all_embeddings = extract_embeddings_pair(bert_encoder, tokenizer, def_examples, max_seq_length, device, load, task, 
        label_list=None, output_mode=None, norm=True, word_level=word_level)
    gender_dir = doPCA(all_embeddings).components_[:k]
    if (not keepdims):
        gender_dir = np.mean(gender_dir, axis=0)
    logger.info("gender direction={} {} {}".format(gender_dir.shape,
            type(gender_dir), gender_dir[:10]))
    return gender_dir


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
	"""Loads a data file into a list of input features."""
	'''
	output_mode: classification or regression
	'''	
	if (label_list != None):
		label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids:   0   0  0    0    0     0       0 0    1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids:   0    0   0   0   0   0   0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambiguously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert(len(input_ids) == max_seq_length)
		assert(len(input_mask) == max_seq_length)
		assert(len(segment_ids) == max_seq_length)

		if (label_list != None):
			if output_mode == "classification":
				label_id = label_map[example.label]
			elif output_mode == "regression":
				label_id = float(example.label)
			else:
				raise KeyError(output_mode)
		else:
			label_id = None

		features.append(
				InputFeatures(tokens=tokens,
							  input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_id=label_id))
	return features


def convert_examples_to_dualfeatures(examples, label_list, max_seq_length, tokenizer, output_mode):
	"""Loads a data file into a list of dual input features."""
	'''
	output_mode: classification or regression
	'''	
	features = []
	for (ex_index, example) in enumerate(tqdm(examples)):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)
		# truncate length
		if len(tokens_a) > max_seq_length - 2:
			tokens_a = tokens_a[:(max_seq_length - 2)]

		tokens_a = ["[CLS]"] + tokens_a + ["[SEP]"]
		segments_a = [0] * len(tokens_a)
		input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
		mask_a = [1] * len(input_ids_a)
		padding_a = [0] * (max_seq_length - len(input_ids_a))
		input_ids_a += padding_a
		mask_a += padding_a
		segments_a += padding_a
		assert(len(input_ids_a) == max_seq_length)
		assert(len(mask_a) == max_seq_length)
		assert(len(segments_a) == max_seq_length)

		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			if len(tokens_b) > max_seq_length - 2:
				tokens_b = tokens_b[:(max_seq_length - 2)]

			tokens_b = ["[CLS]"] + tokens_b + ["[SEP]"]
			segments_b = [0] * len(tokens_b)
			input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
			mask_b = [1] * len(input_ids_b)
			padding_b = [0] * (max_seq_length - len(input_ids_b))
			input_ids_b += padding_b
			mask_b += padding_b
			segments_b += padding_b
			assert(len(input_ids_b) == max_seq_length)
			assert(len(mask_b) == max_seq_length)
			assert(len(segments_b) == max_seq_length)
		else:
			input_ids_b = None
			mask_b = None
			segments_b = None

		features.append(
				DualInputFeatures(input_ids_a=input_ids_a,
						     	  input_ids_b=input_ids_b,
								  mask_a=mask_a,
								  mask_b=mask_b,
								  segments_a=segments_a,
								  segments_b=segments_b))
	return features


def get_tokenizer_encoder(args, device=None):
    '''Return BERT tokenizer and encoder based on args. Used in eval_bias.py.'''
    print("get tokenizer from {}".format(args.model_path))

    import os
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    
    model_weights_path = args.model_path
    if args.model_type == 'bert':
        #args.model_path = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
        config = BertConfig.from_pretrained(args.model_path, num_labels=2, hidden_dropout_prob=0.3)
        model = BertModel.from_pretrained(args.model_path, config=config)
        
    elif args.model_type == 'albert':
        #args.model_path = "albert-base-v2"
        tokenizer = AlbertTokenizer.from_pretrained(args.model_path)
        config = AlbertConfig.from_pretrained(args.model_path, num_labels=2, hidden_dropout_prob=0)
        model = AlbertModel.from_pretrained(args.model_path, config=config)
	
    elif args.model_type == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_path)
        config = DistilBertConfig.from_pretrained(args.model_path, num_labels=2, hidden_dropout_prob=0)
        model = DistilBertModel.from_pretrained(args.model_path, config=config)
        bert_encoder = DistilBertEncoder(model, device)
    
    if (device != None): 
        model.to(device)
        if args.model_type == 'distilbert':
            encoder = DistilBertEncoder(model, device)
        else:
            encoder = BertEncoder(model, device)
    return tokenizer, encoder

def get_encodings(args, encs, tokenizer, encoder, gender_space, device, 
		word_level=False, specific_set=None):
	'''Extract BERT embeddings from encodings dictionary.
	   Perform the debiasing step if debias is specified in args.
	'''
	if (word_level): assert(specific_set != None)

	logger.info("Get encodings")
	logger.info("Debias={}".format(args.debias))

	examples_dict = dict()
	for key in ['targ1', 'targ2', 'attr1', 'attr2']:
		texts = encs[key]['examples'] #a list of text
		category = encs[key]['category'].lower()
		examples = []
		encs[key]['text_ids'] = dict()
		for i, text in enumerate(texts):
			examples.append(InputExample(guid='{}'.format(i), text_a=text, text_b=None, label=None))
			encs[key]['text_ids'][i] = text
		examples_dict[key] = examples
		all_embeddings = extract_embeddings(args, encoder, tokenizer, examples, args.max_seq_length, device, 
					label_list=None, output_mode=None)

		logger.info("Debias category {}".format(category))

		emb_dict = {}
		for index, emb in enumerate(all_embeddings):
			emb /= np.linalg.norm(emb)
			if (args.debias and not category in {'male','female'}): # don't debias gender definitional sentences
				emb = my_we.dropspace(emb, gender_space)
			emb /= np.linalg.norm(emb) # Normalization actually doesn't affect e_size
			emb_dict[index] = emb

		encs[key]['encs'] = emb_dict
	return encs

