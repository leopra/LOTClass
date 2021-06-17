from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from nltk.corpus import stopwords
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from model import LOTClassModel
import warnings
warnings.filterwarnings("ignore")
import string
import spacy

from util import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import math

class LOTClassTrainer(object):

    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        self.dataset_dir = args.dataset_dir
        self.dist_port = args.dist_port
        self.num_cpus = min(10, cpu_count() - 1) if cpu_count() > 1 else 1
        self.world_size = args.gpus #number gpus
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.accum_steps = args.accum_steps
        eff_batch_size = self.train_batch_size * self.world_size * self.accum_steps
        assert abs(eff_batch_size - 128) < 10, f"Make sure the effective training batch size is around 128, current: {eff_batch_size}"
        print(f"Effective training batch size: {eff_batch_size}")
        self.pretrained_lm = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True) #tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.read_label_names(args.dataset_dir, args.label_names_file, 'ext_label_names.txt' ) #CHANGE added file with extended dictionary
        self.num_class = len(self.label_name_dict)
        self.model = LOTClassModel.from_pretrained(self.pretrained_lm,
                                                   output_attentions=False,
                                                   output_hidden_states=False,
                                                   num_labels=self.num_class)
        self.read_data(args.dataset_dir, args.train_file, args.test_file, args.test_label_file)
        self.with_test_label = True if args.test_label_file is not None else False
        self.temp_dir = f'tmp_{self.dist_port}'
        self.mcp_loss = nn.CrossEntropyLoss()
        self.st_loss = nn.KLDivLoss(reduction='batchmean')
        self.update_interval = args.update_interval
        self.early_stop = args.early_stop
        self.spacyWord2Idx = {}
        self.spacyIdx2Word = {}
        self.label_name_dict_spacy = {}

    def computeLemmSpacy(self, docs, spacy_text_file):
        loader_file = os.path.join(self.dataset_dir, spacy_text_file)

        # TODO could add check to skip calculation if file is saved

        nlp = spacy.load("en_core_web_sm")
        lemmDocs = []
        max_len = 0
        for doc in nlp.pipe(docs, disable=["tok2vec","parser"]):
            # Do something with the doc here
            try:
                k = [n.lemma_ for n in doc]
                lemmDocs.append(k)
                if len(k) > max_len:
                    max_len = len(k)
            except:
                lemmDocs.append(['[VUOTA]'])

        vectorizer = CountVectorizer(analyzer=lambda x: x)
        vectorizer.fit_transform(lemmDocs)  # sparse matrix with columns corresponding to words
        words = {w:i for i,w in enumerate(vectorizer.get_feature_names())}
        encodedText = np.full((len(lemmDocs),max_len),-1, dtype=int)

        for j,doc in enumerate(lemmDocs):
            for i, tok in enumerate(doc):
                encodedText[j,i] = words[tok]
        torch.save(encodedText, loader_file)

        self.spacyWord2Idx = words
        self.spacyIdx2Word = {x:i for i,x in words.items()}

        encoded_dict = {i:[] for i in self.label_name_dict.keys()}
        for k,values in self.label_name_dict.items():
            for v in values:
                try:
                    encoded_dict[k].append(self.spacyWord2Idx[v])
                except Exception as e:
                    print('exception', e, v)
                    continue

        self.label_name_dict_spacy = encoded_dict

        torch.save([self.spacyWord2Idx, self.spacyIdx2Word, self.label_name_dict_spacy], os.path.join(self.dataset_dir, 'spacy_data.pt'))
        return encodedText

    def generate_pseudo_labels(self, df, labels, label_term_dict):
        def argmax_label(count_dict):
            print(count_dict)
            maxi = 0
            max_label = None
            for l in count_dict:
                count = 0
                for t in count_dict[l]:
                    count += count_dict[l][t]
                if count > maxi:
                    maxi = count
                    max_label = l
            return max_label

        y = []
        X = []
        for index, tokens in enumerate(df):
            words = tokens
            count_dict = {}
            flag = 0
            for l in labels:
                seed_words = set()
                for w in label_term_dict[l]:
                    seed_words.add(w)
                int_labels = list(set(words).intersection(seed_words))
                if len(int_labels) == 0:
                    continue
                for word in words:
                    if word in int_labels:
                        flag = 1
                        try:
                            temp = count_dict[l]
                        except:
                            count_dict[l] = {}
                        try:
                            count_dict[l][word] += 1
                        except:
                            count_dict[l][word] = 1
            if flag:
                lbl = argmax_label(count_dict)
                if not lbl:
                    y.append(-1)
                else:
                    y.append(lbl)
                    X.append(tokens)
            else:
                y.append(-1)
        if (np.array(y) != -1).any():
            print('found:', [(x,y) for x,y in list(zip(X,y)) if y != -1])
        return X, y

    # set up distributed training
    def set_up_dist(self, rank):
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{self.dist_port}',
            world_size=self.world_size,
            rank=rank
        )
        # create local model
        model = self.model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        return model

    # get document truncation statistics with the defined max length
    def corpus_trunc_stats(self, docs):
        doc_len = []
        for doc in docs:
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            doc_len.append(len(input_ids))
        print(f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
        trunc_frac = np.sum(np.array(doc_len) > self.max_len) / len(doc_len)
        print(f"Truncated fraction of all documents: {trunc_frac}")

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False, label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)

        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]

            # TODO check if this works
            spacy_encode = self.computeLemmSpacy(docs, 'spacy_lemm.txt')
            tensor_spacy = torch.tensor(spacy_encode)

            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")
            if label_file is not None:
                print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
                truth = open(os.path.join(dataset_dir, label_file))
                labels = [int(label.strip()) for label in truth.readlines()]
                labels = torch.tensor(labels)
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels, "tensor_spacy": tensor_spacy}
            else:
                data = {"input_ids": input_ids, "attention_masks": attention_masks, "tensor_spacy": tensor_spacy}
            torch.save(data, loader_file)

        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                print(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]
                print("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name, "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                print(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data
    
    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1 # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1: # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i+1].startswith("##"):
                word = ''.join(wordpcs)
                #CHANGE label2class to mapping
                if word in self.mappingWord2Index:
                    label_idx[idx] = self.mappingWord2Index[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True, max_length=self.max_len, 
                                                            padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, test_file, test_label_file):

        self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, None, "train.pt",
                                                                  find_label_name=True, label_name_loader_name="label_name_data.pt")
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt")

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file, extended_dict):
        ext_dict = os.path.join(self.dataset_dir, extended_dict)
        if os.path.exists(ext_dict):
            print(f"Loading extended category labels {ext_dict}")
            label_name_file = open(ext_dict)
        else:
            label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower() for word in category_words.strip().split()] for i, category_words in enumerate(label_names)}

        self.mappingWordIndexClass = {}
        self.mappingWord2Index = {}
        wordCounter = 0
        for i, words in self.label_name_dict.items():
            for word in words:
                self.mappingWord2Index[word] = wordCounter
                self.mappingWordIndexClass[wordCounter] = i
                wordCounter += 1


        #dictionary added to train multiple dictionaries for the same class
        self.indextoken2class = {i: [word.lower() for word in category_words.strip().split()] for i, category_words in enumerate(label_names)}
        print(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            #map word to class {'politics':2, ...}
            for word in self.label_name_dict[class_idx]:
                assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # create dataset loader
    def make_dataloader(self, rank, data_dict, batch_size):
        if "labels" in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["labels"])

        elif "tensor_spacy" in data_dict and "labels" not in data_dict:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"], data_dict["tensor_spacy"])
        else:
            dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])

        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
        return dataset_loader

    # filter out stop words and words in multiple categories
    def filter_keywords(self, category_vocab_size=100):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.category_words_freq.items():
            sorted_dict = {k:v for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:category_vocab_size]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.category_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.category_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = stopwords.words('english')
        for i, word_list in self.category_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
                if word in self.label_name_dict[self.mappingWordIndexClass[i]]:
                    continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word_id in repeat_words:
                    delete_idx.append(j)
            self.category_vocab[i] = np.delete(self.category_vocab[i], delete_idx)

    # construct category vocabulary (distributed function)
    def category_vocabulary_dist(self, rank, top_pred_num=50, loader_name="category_vocab.pt"):
        print("RANK: ", rank)
        model = self.set_up_dist(rank)
        #eval switches off some layers that behave differently betweeen traning and predictiong
        model.eval()
        label_name_dataset_loader = self.make_dataloader(rank, self.label_name_data, self.eval_batch_size)
        #CHANGE num class to num words
        category_words_freq = {i: defaultdict(float) for i in range(len(self.mappingWordIndexClass))}
        wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader) if rank == 0 else label_name_dataset_loader
        try:
            for batch in wrap_label_name_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    label_pos = batch[2].to(rank)
                    match_idx = label_pos >= 0
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None, 
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)
                    label_idx = label_pos[match_idx]
                    for i, word_list in enumerate(sorted_res):
                        for j, word_id in enumerate(word_list):
                            category_words_freq[label_idx[i].item()][word_id.item()] += 1
            save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
            torch.save(category_words_freq, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # construct category vocabulary
    def category_vocabulary(self, top_pred_num=50, category_vocab_size=100, loader_name="category_vocab.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading category vocabulary from {loader_file}")
            self.category_vocab = torch.load(loader_file)
        else:
            print("Contructing category vocabulary.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.category_vocabulary_dist, nprocs=self.world_size, args=(top_pred_num, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            self.category_words_freq = {i: defaultdict(float) for i in range(len(self.mappingWordIndexClass))}
            for i in range(len(self.mappingWordIndexClass)):
                for category_words_freq in gather_res:
                    for word_id, freq in category_words_freq[i].items():
                        self.category_words_freq[i][word_id] += freq
            self.filter_keywords(category_vocab_size)
            torch.save(self.category_vocab, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        for i, category_vocab in self.category_vocab.items():
            print(f"Class {i} category vocabulary: {[self.inv_vocab[w] for w in category_vocab]}\n")

    # prepare self supervision for masked category prediction (distributed function)
    def prepare_mcp_dist(self, rank, top_pred_num=50, match_threshold=20, loader_name="mcp_train.pt", strictThreshClass = []):
        if len(self.label_name_dict_spacy.keys()) == 0:
            self.spacyWord2Idx, self.spacyIdx2Word, self.label_name_dict_spacy = torch.load(os.path.join(self.dataset_dir, 'spacy_data.pt'))
        model = self.set_up_dist(rank)
        model.eval()
        train_dataset_loader = self.make_dataloader(rank, self.train_data, self.eval_batch_size)
        all_input_ids = []
        all_mask_label = []
        all_input_mask = []
        category_doc_num = defaultdict(int)
        wrap_train_dataset_loader = tqdm(train_dataset_loader) if rank == 0 else train_dataset_loader
        try:
            for batch in wrap_train_dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    predictions = model(input_ids,
                                        pred_mode="mlm",
                                        token_type_ids=None,
                                        attention_mask=input_mask)
                    _, sorted_res = torch.topk(predictions, top_pred_num, dim=-1)
                    for i, category_vocab in self.category_vocab.items():
                        k = 1
                        if i in strictThreshClass:
                            k = 2
                        match_idx = torch.zeros_like(sorted_res).bool()
                        for word_id in category_vocab:
                            match_idx = (sorted_res == word_id) | match_idx #TODO what is going on here
                        match_count = torch.sum(match_idx.int(), dim=-1)
                        valid_idx = (match_count > match_threshold) & (input_mask > 0)

                        #TODO added check for words counts
                        spacy_lemm = batch[2]
                        #print(self.label_name_dict_spacy)
                        X, y_cls = self.generate_pseudo_labels(spacy_lemm, self.label_name_dict_spacy.keys(), self.label_name_dict_spacy)
                        # TODO put this back valid_idx = (match_count > len(category_vocab) * match_threshold * k / top_pred_num) & (input_mask > 0)
                        valid_doc = torch.sum(valid_idx, dim=-1) > 0

                        if valid_doc.any():
                            mask_label = -1 * torch.ones_like(input_ids)
                            mask_label[valid_idx] = self.mappingWordIndexClass[i] #TODO probably add here conversion word to class

                            #print(mask_label[:, 0])

                            mask_label[:,0] = torch.tensor(y_cls)
                            all_input_ids.append(input_ids[valid_doc].cpu())
                            all_mask_label.append(mask_label[valid_doc].cpu())
                            all_input_mask.append(input_mask[valid_doc].cpu())
                            category_doc_num[i] += valid_doc.int().sum().item()
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_mask_label = torch.cat(all_mask_label, dim=0)
            all_input_mask = torch.cat(all_input_mask, dim=0)
            save_dict = {
                "all_input_ids": all_input_ids,
                "all_mask_label": all_mask_label,
                "all_input_mask": all_input_mask,
                "category_doc_num": category_doc_num,
            }
            save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
            torch.save(save_dict, save_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)

    # prepare self supervision for masked category prediction
    def prepare_mcp(self, top_pred_num=50, match_threshold=20, loader_name="mcp_train.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading masked category prediction data from {loader_file}")
            self.mcp_data = torch.load(loader_file)
        else:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            print("Preparing self supervision for masked category prediction.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.prepare_mcp_dist, nprocs=self.world_size, args=(top_pred_num, match_threshold, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            all_input_ids = torch.cat([res["all_input_ids"] for res in gather_res], dim=0)
            all_mask_label = torch.cat([res["all_mask_label"] for res in gather_res], dim=0)
            all_input_mask = torch.cat([res["all_input_mask"] for res in gather_res], dim=0)
            category_doc_num = {i: 0 for i in range(self.num_class)}
            for i in category_doc_num:
                for res in gather_res:
                    if i in res["category_doc_num"]:
                        category_doc_num[i] += res["category_doc_num"][i]
            print(f"Number of documents with category indicative terms found for each category is: {category_doc_num}")
            self.mcp_data = {"input_ids": all_input_ids, "attention_masks": all_input_mask, "labels": all_mask_label}
            torch.save(self.mcp_data, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            for i in category_doc_num:
                assert category_doc_num[i] > 10, f"Too few ({category_doc_num[i]}) documents with category indicative terms found for category {i}; " \
                       "try to add more unlabeled documents to the training corpus (recommend) or reduce `--match_threshold` (not recommend)"
        print(f"There are totally {len(self.mcp_data['input_ids'])} documents with category indicative terms.")

    # masked category prediction (distributed function)
    def mcp_dist(self, rank, epochs=5, loader_name="mcp_model.pt"):
        model = self.set_up_dist(rank)
        mcp_dataset_loader = self.make_dataloader(rank, self.mcp_data, self.train_batch_size)
        total_steps = len(mcp_dataset_loader) * epochs / self.accum_steps
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        try:
            for i in range(epochs):
                model.train()
                total_train_loss = 0
                if rank == 0:
                    print(f"Epoch {i+1}:")
                wrap_mcp_dataset_loader = tqdm(mcp_dataset_loader) if rank == 0 else mcp_dataset_loader
                model.zero_grad()
                for j, batch in enumerate(wrap_mcp_dataset_loader):
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    labels = batch[2].to(rank)
                    mask_pos = labels >= 0
                    labels = labels[mask_pos]
                    # mask out category indicative words
                    #TODO consider masking randomly
                    input_ids[mask_pos] = self.mask_id
                    logits = model(input_ids, 
                                   pred_mode="classification",
                                   token_type_ids=None, 
                                   attention_mask=input_mask)
                    logits = logits[mask_pos]
                    loss = self.mcp_loss(logits.view(-1, self.num_class), labels.view(-1)) / self.accum_steps
                    total_train_loss += loss.item()
                    loss.backward()
                    if (j+1) % self.accum_steps == 0:
                        # Clip the norm of the gradients to 1.0.
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                avg_train_loss = torch.tensor([total_train_loss / len(mcp_dataset_loader) * self.accum_steps]).to(rank)
                gather_list = [torch.ones_like(avg_train_loss) for _ in range(self.world_size)]
                dist.all_gather(gather_list, avg_train_loss)
                avg_train_loss = torch.tensor(gather_list)
                if rank == 0:
                    print(f"Average training loss: {avg_train_loss.mean().item()}")
            if rank == 0:
                loader_file = os.path.join(self.dataset_dir, loader_name)
                torch.save(model.module.state_dict(), loader_file)
        except RuntimeError as err:
            self.cuda_mem_error(err, "train", rank)

    # masked category prediction
    def mcp(self, top_pred_num=50, match_threshold=20, epochs=5, loader_name="mcp_model.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"\nLoading model trained via masked category prediction from {loader_file}")
        else:
            self.prepare_mcp(top_pred_num, match_threshold)
            print(f"\nTraining model via masked category prediction.")
            mp.spawn(self.mcp_dist, nprocs=self.world_size, args=(epochs, loader_name))
        self.model.load_state_dict(torch.load(loader_file))

    # prepare self training data and target distribution
    def prepare_self_train_data(self, rank, model, idx):
        target_num = min(self.world_size * self.train_batch_size * self.update_interval * self.accum_steps, len(self.train_data["input_ids"]))
        if idx + target_num >= len(self.train_data["input_ids"]):
            select_idx = torch.cat((torch.arange(idx, len(self.train_data["input_ids"])),
                                    torch.arange(idx + target_num - len(self.train_data["input_ids"]))))
        else:
            select_idx = torch.arange(idx, idx + target_num)
        assert len(select_idx) == target_num
        idx = (idx + len(select_idx)) % len(self.train_data["input_ids"])
        select_dataset = {"input_ids": self.train_data["input_ids"][select_idx],
                          "attention_masks": self.train_data["attention_masks"][select_idx]}
        dataset_loader = self.make_dataloader(rank, select_dataset, self.eval_batch_size)
        input_ids, input_mask, preds = self.inference(model, dataset_loader, rank, return_type="data")
        gather_input_ids = [torch.ones_like(input_ids) for _ in range(self.world_size)]
        gather_input_mask = [torch.ones_like(input_mask) for _ in range(self.world_size)]
        gather_preds = [torch.ones_like(preds) for _ in range(self.world_size)]
        dist.all_gather(gather_input_ids, input_ids)
        dist.all_gather(gather_input_mask, input_mask)
        dist.all_gather(gather_preds, preds)
        input_ids = torch.cat(gather_input_ids, dim=0).cpu()
        input_mask = torch.cat(gather_input_mask, dim=0).cpu()
        all_preds = torch.cat(gather_preds, dim=0).cpu()
        weight = all_preds**2 / torch.sum(all_preds, dim=0)
        target_dist = (weight.t() / torch.sum(weight, dim=1)).t()
        all_target_pred = target_dist.argmax(dim=-1)
        agree = (all_preds.argmax(dim=-1) == all_target_pred).int().sum().item() / len(all_target_pred)
        self_train_dict = {"input_ids": input_ids, "attention_masks": input_mask, "labels": target_dist}
        return self_train_dict, idx, agree

    # train a model on batches of data with target labels
    def self_train_batches(self, rank, model, self_train_loader, optimizer, scheduler, test_dataset_loader):
        model.train()
        total_train_loss = 0
        wrap_train_dataset_loader = tqdm(self_train_loader) if rank == 0 else self_train_loader
        model.zero_grad()
        try:
            for j, batch in enumerate(wrap_train_dataset_loader):
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                target_dist = batch[2].to(rank)
                logits = model(input_ids,
                               pred_mode="classification",
                               token_type_ids=None,
                               attention_mask=input_mask)
                logits = logits[:, 0, :]
                preds = nn.LogSoftmax(dim=-1)(logits)
                loss = self.st_loss(preds.view(-1, self.num_class), target_dist.view(-1, self.num_class)) / self.accum_steps
                total_train_loss += loss.item()
                loss.backward()
                if (j+1) % self.accum_steps == 0:
                    # Clip the norm of the gradients to 1.0.
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
            if self.with_test_label:
                acc = self.inference(model, test_dataset_loader, rank, return_type="acc")
                gather_acc = [torch.ones_like(acc) for _ in range(self.world_size)]
                dist.all_gather(gather_acc, acc)
                acc = torch.tensor(gather_acc).mean().item()
            avg_train_loss = torch.tensor([total_train_loss / len(wrap_train_dataset_loader) * self.accum_steps]).to(rank)
            gather_list = [torch.ones_like(avg_train_loss) for _ in range(self.world_size)]
            dist.all_gather(gather_list, avg_train_loss)
            avg_train_loss = torch.tensor(gather_list)
            if rank == 0:
                print(f"lr: {optimizer.param_groups[0]['lr']:.4g}")
                print(f"Average training loss: {avg_train_loss.mean().item()}")
                if self.with_test_label:
                    print(f"Test acc: {acc}")
        except RuntimeError as err:
            self.cuda_mem_error(err, "train", rank)

    # self training (distributed function)
    def self_train_dist(self, rank, epochs, loader_name="final_model.pt"):
        model = self.set_up_dist(rank)
        test_dataset_loader = self.make_dataloader(rank, self.test_data, self.eval_batch_size) if self.with_test_label else None
        total_steps = int(len(self.train_data["input_ids"]) * epochs / (self.world_size * self.train_batch_size * self.accum_steps))
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
        idx = 0
        if self.early_stop:
            agree_count = 0
        for i in range(int(total_steps / self.update_interval)):
            self_train_dict, idx, agree = self.prepare_self_train_data(rank, model, idx)
            # early stop if current prediction agrees with target distribution for 3 consecutive updates
            if self.early_stop:
                if 1 - agree < 1e-3:
                    agree_count += 1
                else:
                    agree_count = 0
                if agree_count >= 3:
                    break
            self_train_dataset_loader = self.make_dataloader(rank, self_train_dict, self.train_batch_size)
            self.self_train_batches(rank, model, self_train_dataset_loader, optimizer, scheduler, test_dataset_loader)
        if rank == 0:
            loader_file = os.path.join(self.dataset_dir, loader_name)
            print(f"Saving final model to {loader_file}")
            torch.save(model.module.state_dict(), loader_file)

    # self training
    def self_train(self, epochs, loader_name="final_model.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"\nFinal model {loader_file} found, skip self-training")
        else:
            rand_idx = torch.randperm(len(self.train_data["input_ids"]))
            self.train_data = {"input_ids": self.train_data["input_ids"][rand_idx],
                               "attention_masks": self.train_data["attention_masks"][rand_idx]}
            print(f"\nStart self-training.")
            mp.spawn(self.self_train_dist, nprocs=self.world_size, args=(epochs, loader_name))
    
    # use a model to do inference on a dataloader
    def inference(self, model, dataset_loader, rank, return_type):
        if return_type == "data":
            all_input_ids = []
            all_input_mask = []
            all_preds = []
        elif return_type == "acc":
            pred_labels = []
            truth_labels = []
        elif return_type == "pred":
            pred_labels = []
        model.eval()
        try:
            for batch in dataset_loader:
                with torch.no_grad():
                    input_ids = batch[0].to(rank)
                    input_mask = batch[1].to(rank)
                    logits = model(input_ids,
                                   pred_mode="classification",
                                   token_type_ids=None,
                                   attention_mask=input_mask)
                    logits = logits[:, 0, :]
                    if return_type == "data":
                        all_input_ids.append(input_ids)
                        all_input_mask.append(input_mask)
                        all_preds.append(nn.Softmax(dim=-1)(logits))
                    elif return_type == "acc":
                        labels = batch[2]
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
                        truth_labels.append(labels)
                    elif return_type == "pred":
                        pred_labels.append(torch.argmax(logits, dim=-1).cpu())
            if return_type == "data":
                all_input_ids = torch.cat(all_input_ids, dim=0)
                all_input_mask = torch.cat(all_input_mask, dim=0)
                all_preds = torch.cat(all_preds, dim=0)
                return all_input_ids, all_input_mask, all_preds
            elif return_type == "acc":
                pred_labels = torch.cat(pred_labels, dim=0)
                truth_labels = torch.cat(truth_labels, dim=0)
                samples = len(truth_labels)
                acc = (pred_labels == truth_labels).float().sum() / samples
                return acc.to(rank)
            elif return_type == "pred":
                pred_labels = torch.cat(pred_labels, dim=0)
                return pred_labels
        except RuntimeError as err:
            self.cuda_mem_error(err, "eval", rank)
    
    # use trained model to make predictions on the test set
    def write_results(self, loader_name="final_model.pt", out_file="out.txt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"\nLoading final model from {loader_file}")
        self.model.load_state_dict(torch.load(loader_file))
        self.model.to(0)
        test_set = TensorDataset(self.test_data["input_ids"], self.test_data["attention_masks"])
        test_dataset_loader = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=self.eval_batch_size)
        pred_labels = self.inference(self.model, test_dataset_loader, 0, return_type="pred")
        out_file = os.path.join(self.dataset_dir, out_file)
        print(f"Writing prediction results to {out_file}")
        f_out = open(out_file, 'w')
        for label in pred_labels:
            f_out.write(str(label.item()) + '\n')



    def get_rank_matrix(self, docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index, term_count, word_to_index, doc_freq_thresh):
        E_LT = np.zeros((label_count, term_count))
        components = {}


        for l in label_docs_dict:
            components[l] = {}
            docs = label_docs_dict[l]
            #TODO ADDED FIX
            if len(docs) != 0:
                docfreq_local = calculate_doc_freq(docs)
                vect = CountVectorizer(tokenizer=lambda x: x.split())
                X = vect.fit_transform(docs)
                rel_freq = X.sum(axis=0) / len(docs)
                rel_freq = np.asarray(rel_freq).reshape(-1)
                names = vect.get_feature_names()

                for i, name in enumerate(names):
                    try:
                        if docfreq_local[name] < doc_freq_thresh:
                            continue
                    except:
                        continue
                    E_LT[l][word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[
                        name] * np.tanh(rel_freq[i])
                    components[l][name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                                           "idf": inv_docfreq[name],
                                           "rel_freq": np.tanh(rel_freq[i]),
                                           "rank": E_LT[l][word_to_index[name]]}
        return E_LT, components

    def expand(self, E_LT, index_to_label, index_to_word, it, label_count, old_label_term_dict, label_docs_dict, n1):

        word_map = {}
        zero_docs_labels = set()
        stopwords_vocab = stopwords.words('english')
        for l in range(label_count):
            N=20
            count=0
            if not np.any(E_LT):
                continue
            elif len(label_docs_dict[l]) == 0:
                zero_docs_labels.add(l)
            else:
                n = 100 #min(n1 * (it), int(math.log(len(label_docs_dict[l]), 1.5)))
                inds_popular = E_LT[l].argsort()[::-1][:n]
                for num, word_ind in enumerate(inds_popular):

                    word = index_to_word[word_ind]
                    #if the word is not in the Bert vocabualry i can just skip
                    if word not in self.vocab:
                        continue
                    if word in stopwords_vocab:
                        continue
                    if word in string.punctuation:
                        continue
                    if any(char.isdigit() for char in word):
                        continue
                    #if i found 20 good words i can quit
                    if count == N:
                        break
                    try:
                        temp = word_map[word]
                        if E_LT[l][word_ind] > temp[1]:
                            word_map[word] = (l, E_LT[l][word_ind])
                    except:
                        word_map[word] = (l, E_LT[l][word_ind])
                    count += 1

        new_label_term_dict = defaultdict(set)
        for word in word_map:
            label, val = word_map[word]
            new_label_term_dict[label].add(word)
        for l in zero_docs_labels:
            new_label_term_dict[l] = old_label_term_dict[l]


        return new_label_term_dict

    def expansion(self, loader_name="train.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print('NO LOADER FOUND')
            return

        ### VARIABLES INTEGRATION
        label_count = self.num_class
        #TODO HARDCODED
        index_to_label = {0: 'politics', 1: 'sports', 2:'business', 3: 'technology'}
        label_term_dict = {0: ['politics'], 1: ['sports'], 2:['business'], 3: ['technology']}
        label_to_index = dict([(i,x) for (x,i) in index_to_label.items()])

        pred_label_file = os.path.join(self.dataset_dir, "pred_labels_train.pt")

        #### PREDICTION AND EXPANSION
        if os.path.exists(pred_label_file):
            pred_labels = torch.load(pred_label_file)
        else:
            loader_file = os.path.join(self.dataset_dir, "mcp_model.pt")
            assert os.path.exists(loader_file)
            print(f"\nLoading final model from {loader_file}, seed expansion")
            self.model.load_state_dict(torch.load(loader_file))
            self.model.to(0)
            train_set = TensorDataset(self.train_data["input_ids"], self.train_data["attention_masks"])
            train_dataset_loader = DataLoader(train_set, sampler=SequentialSampler(train_set), batch_size=self.eval_batch_size)
            pred_labels = self.inference(self.model, train_dataset_loader, 0, return_type="pred").numpy()
            torch.save(pred_labels, pred_label_file)

        df = data['input_ids'].numpy()
        df = [self.tokenizer.decode(doc).replace('[PAD]','').strip() for doc in df] #TODO remove special tokens [mask] [CLS] [SEP] from output

        from nltk.tokenize import RegexpTokenizer
        tokenizerPunct = RegexpTokenizer(r'\w+')

        df = [' '.join(tokenizerPunct.tokenize(sent)) for sent in df]

        #FOR TESTING use random prediction as the 120k preds take a lot of time
        #import random
        #pred_labels = np.array([random.sample([0,1,2,3],1)[0] for x in range(len(df))])

        label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

        print([len(x) for y,x in label_docs_dict.items()])
        word_vec = preprocess(df)
        word_to_index, index_to_word = create_word_index_maps(word_vec)

        docfreq = calculate_df_doc_freq(df)
        inv_docfreq = calculate_inv_doc_freq(df, docfreq)
        #TODO remove punctuation connected to tokens

        term_count = len(word_to_index)
        E_LT, components = self.get_rank_matrix(docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index,
                                                term_count, word_to_index, doc_freq_thresh=5)

        label_term_dict = self.expand(E_LT, index_to_label, index_to_word, 1, label_count, label_term_dict, label_docs_dict, n1=5)

        print('Expansion: ', label_term_dict)

        # save the extended one in 'ext_label_names.txt'
        with open(os.path.join(self.dataset_dir, 'ext_label_names.txt'), "w+") as f:
            #I have to be sure there are no duplicates
            labelss = [x[1] for x in index_to_label.items()]
            num_seed_to_add=1
            for l, seeds in sorted(label_term_dict.items(), key=lambda x: x[0]):
                f.write(index_to_label[l])
                cc = 0
                for w in seeds:
                    #exit if the correct number of seed is added
                    if cc == num_seed_to_add:
                        break
                    if w not in labelss:
                        f.write(' ' + w)
                        cc +=1

                f.write('\n')
            f.close()

    # print error message based on CUDA memory error
    def cuda_mem_error(self, err, mode, rank):
        if rank == 0:
            print(err)
            if "CUDA out of memory" in str(err):
                if mode == "eval":
                    print(f"Your GPUs can't hold the current batch size for evaluation, try to reduce `--eval_batch_size`, current: {self.eval_batch_size}")
                else:
                    print(f"Your GPUs can't hold the current batch size for training, try to reduce `--train_batch_size`, current: {self.train_batch_size}")
        sys.exit(1)
