import os
import ast
import json
import random
import time
import unidecode
import torch
import torch.utils.data
import pickle
import glob
import jsonlines
import numpy as np
from tasks.data_utils import InputExample
from tqdm import tqdm
from utils import print_rank_0
from data_utils.corpora import punctuation_standardization
from data_utils.lazy_loader import exists_lazy, LazyWriter, LazyLoader
from .pvp import PVPS
from bs4 import BeautifulSoup as bs

class DataProcessor:
    def __init__(self, data_dir, tokenizer, lazy_seq2seq_loader=False, **kwargs):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.lazy_seq2seq_loader = lazy_seq2seq_loader

    def _yield_examples(self, split):
        raise NotImplementedError

    def create_examples(self, split):
        print_rank_0(f"Creating {split} dataset from {self.data_dir}")
        if not self.lazy_seq2seq_loader:
            example_list = []
            for idx, example in enumerate(self._yield_examples(split)):
                if (idx + 1) % 20000 == 0:
                    print_rank_0(f"Complete {idx + 1} examples")
                example_list.append(example)
        else:
            if (not exists_lazy(self.data_dir,
                                data_type=split) and torch.distributed.get_rank() == 0):
                example_writer = LazyWriter(self.data_dir, data_type=split, is_array=False)
                for idx, example in enumerate(self._yield_examples(split)):
                    if (idx + 1) % 20000 == 0:
                        print_rank_0(f"Complete {idx + 1} examples")
                    example_writer.write(example)
            else:
                while not os.path.exists(LazyWriter.get_len_path(self.data_dir, data_type=split)):
                    time.sleep(1)
            example_list = LazyLoader(self.data_dir, data_type=split, map_fn=InputExample.from_json_string,
                                      mem_map=True, is_array=False)
        print_rank_0(f"Creating {len(example_list)} examples for {split}")
        return example_list


def blanklm_detokenize(string, is_target=False):
    string = string.replace("_UNK", "[UNK]")
    string = string.replace("<blank>", "[MASK]")
    return string


class AlltoEnMultitaskProcessor(DataProcessor):

    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir 
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args

    def detokenize(self, string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-",
                     "[": "-LSB-", "]": "-RSB-",
                     "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", "\"")
        string = string.replace("``", "\"")
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string
    
    def create_examples(self, split):

        example_list = []
        #adding Scisummnet dataset, first 800 to train, the rest for evaluation
        sci_count = 0
        total_count = 0
        pids = os.listdir(os.path.join(self.data_dir, "scisummnet/top1000_complete"))
        if split == "train":
            pids = pids[:800]
        else:
            pids = pids[800:]

        for pid in pids:               
            sci_count += 1
            total_count += 1
            targetfile = os.path.join(self.data_dir, "scisummnet/top1000_complete", pid, "summary", f"{pid}.gold.txt")
            sourcefile = os.path.join(self.data_dir, "scisummnet/top1000_complete", pid, "Documents_xml", f"{pid}.xml")
                
            source = ""
            infile = open(sourcefile,"r")
            contents = infile.read()
            soup = bs(contents,'xml')
            texts = soup.find_all('SECTION')
            for section in texts:
                #print(section.get('title'))
                #txt = txt + '\n' + section.get('title') + '\n'
                temp = section.find_all('S')
                for data in temp:
                    line = data.get_text()
                    line = line.strip()
                    line = punctuation_standardization(line)
                    line = self.detokenize(line)
                    source = source + line
                
            target = ""
            infile = open(targetfile, 'r')
            for line in infile:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line)
                
                target = target + line
                
                y_ref = self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target).tokenization)
            if sci_count % 100 == 0:
                print_rank_0(f"sci document summary = {target}")

            example = InputExample(guid=total_count, text_a=source, text_b=target, meta={'tgt_lang':'en', 'ref':y_ref, 'task':'sum'})
            example_list.append(example)           
                        
        if split == "train":
            srcs = ["cs-en/train.cs_en_src_documents.txt", "de-en/train.de_en_src_documents.txt", "fr-en/train.fr_en_src_documents.txt"]
            en_tgt = ["cs-en/train.cs_en_tgt_summaries_prep.txt", "de-en/train.de_en_tgt_summaries_prep.txt", "fr-en/train.fr_en_tgt_summaries_prep.txt"]
        elif split == "dev" or split == "test":
            srcs = ["cs-en/val.cs_en_src_documents.txt", "de-en/val.de_en_src_documents.txt", "fr-en/val.fr_en_src_documents.txt"]
            en_tgt = ["cs-en/val.cs_en_tgt_summaries_prep.txt", "de-en/val.de_en_tgt_summaries_prep.txt", "fr-en/val.fr_en_tgt_summaries_prep.txt"]
        else:
            raise NotImplementedError(split)
        
        # Processing XWikis

        for i in range(len(srcs)):
            if i == 0:
                lang = 'cs'
            elif i == 1:
                lang = 'de'
            else:
                lang = 'fr'
            
            print_rank_0(f"now processing {lang}")
            lang_count = 0 

            with open(os.path.join(self.data_dir, "XWikis-prepa", srcs[i])) as source, open(os.path.join(self.data_dir, "XWikis-prepa", en_tgt[i])) as target: 
                for x, y in zip(source, target):
                    x = x.strip()
                    y = y.strip()
                    lang_count += 1
                    total_count += 1
                    y = punctuation_standardization(y)
                    y = self.detokenize(y)
                    y_ref = self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(y).tokenization)
                    example = InputExample(guid=total_count, text_a=x, text_b=y, meta={'tgt_lang':'en', 'ref':y_ref, 'task':'sum'})
                    example_list.append(example)
                    
                    if lang_count == 20000 and split == 'train':
                        print_rank_0(y)
                        break
                    elif lang_count == 200 and split in ["dev", "test"]:
                        print_rank_0(y)
                        break
            print_rank_0(f"{lang}-{split} has {lang_count} examples")
        
        #LCSTS Chinese Summary, Extracted from Weibo 
        
        print_rank_0("processing LCSTS for chinese summary")
        lcsts_count = 0
        with jsonlines.open(f"/mnt/yrfs/litianjian/datasets/LCSTS2.0/DATA/{split}.jsonl") as f:
            for data in f.iter():
                total_count += 1
                lcsts_count += 1
                text = data['text']
                en_summary = data['en-summary']
                target = punctuation_standardization(en_summary)
                target = self.detokenize(target)
                #zh_summary = data['zh-summary']
                
                meta_en = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(en_summary).tokenization),"tgt_lang": "en", "task": "sum"}
                #meta_zh = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(zh_summary).tokenization),"tgt_lang": "zh", "task": "sum"}
                example_en = InputExample(guid=total_count, text_a=text, text_b=target, meta=meta_en)
                #example_zh = InputExample(guid=count, text_a=text, text_b=zh_summary, meta=meta_zh)

                example_list.append(example_en)
                if lcsts_count == 10000 and split == 'train':
                    print_rank_0(en_summary)
                    break
                elif lcsts_count == 200 and split in ['dev', 'test']:
                    print_rank_0(en_summary)
                    break
        
        if split == 'train':
            random.shuffle(example_list)
            print_rank_0(f"returning {len(example_list)} for {split}")
            return example_list

   
        # CNNDM JA dataset for evaluation 
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "val"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        dir = "/mnt/yrfs/mengyang/summary_mGLM/mGLM/DATA/cnn_dm_ja"
        cnndm_ja_list = []
        source_texts, target_texts = [], []
        with open(os.path.join(dir, f"{filename}.source"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line)
                source_texts.append(line)
        with open(os.path.join(dir, f"{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line, is_target=True)
                target_texts.append(line)
        assert len(source_texts) == len(target_texts)
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            guid = "%s-%s" % (split, idx)
            meta = {"tgt_lang": 'en', "task": "sum", "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            if idx < 3:
                print_rank_0(
                    (source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
            example_list.append(example)

        
        random.shuffle(example_list)
        print_rank_0(f"returning {len(example_list)} for {split}")
        return example_list

class XWikisProcessor(DataProcessor):
    
    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args

    def detokenize(self, string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-",
                     "[": "-LSB-", "]": "-RSB-",
                     "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", "\"")
        string = string.replace("``", "\"")
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string
    
    def create_examples(self, split):
        
        if split == "train":
            srcs = ["cs-en/train.cs_en_src_documents.txt", "de-en/train.de_en_src_documents.txt", "fr-en/train.fr_en_src_documents.txt"]
            en_tgt = ["cs-en/train.cs_en_tgt_summaries_prep.txt", "de-en/train.de_en_tgt_summaries_prep.txt", "fr-en/train.fr_en_tgt_summaries_prep.txt"]
        elif split == "dev" or split == "test":
            srcs = ["cs-en/val.cs_en_src_documents.txt", "de-en/val.de_en_src_documents.txt", "fr-en/val.fr_en_src_documents.txt"]
            en_tgt = ["cs-en/val.cs_en_tgt_summaries_prep.txt", "de-en/val.de_en_tgt_summaries_prep.txt", "fr-en/val.fr_en_tgt_summaries_prep.txt"]
        else:
            raise NotImplementedError(split)
        
        
        example_list = []
        total_count = 0
        for i in range(len(srcs)):
            if i == 0:
                lang = 'cs'
            elif i == 1:
                lang = 'de'
            else:
                lang = 'fr'
            
            print_rank_0(f"now processing {lang}")
            count = 0 

            with open(os.path.join(self.data_dir, srcs[i])) as source, open(os.path.join(self.data_dir, en_tgt[i])) as target: 
                for x, y in zip(source, target):
                    x = x.strip()
                    y = y.strip()
                    count += 1
                    total_count += 1
                    y_ref = self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(y).tokenization)
                    example = InputExample(guid=total_count, text_a=x, text_b=y, meta={'tgt_lang':'en', 'ref':y_ref, 'task':'sum'})
                    example_list.append(example)
                    
                    if count == 10000 and split == 'train':
                        print_rank_0(y)
                        break
            print_rank_0(f"{lang}-{split} has {count} examples")
        
        return example_list
        
class SciTLDRProcessor(DataProcessor):

    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task 
        self.args = args


    def create_examples(self, split):
        
        if split == "train":
            filename = "train.jsonl"
        elif split == "dev":
            filename = "dev.jsonl"
        elif split == "test":
            filename = "test.jsonl"
        else:
            raise NotImplementedError(split)
        
        count = 0
        example_list = []
        with jsonlines.open(os.path.join(self.data_dir, filename)) as f:
            for data in f.iter():
                count += 1
                source = data['source']
                target = data['target']
                text_a = ""
                text_b = ""
                
                for sentence in source:
                    text_a = text_a + sentence
                               
                text_b = data['target'][-1]
                if count % 100 == 0:
                    print(text_b)
                example = InputExample(guid=count, text_a=text_a, text_b=text_b, meta={'ref':text_b})
                example_list.append(example)
         
        return example_list


class LCSTSProcessor(DataProcessor):

    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args


    def create_examples(self, split):
        
        count = 0
        example_list = []


        import jsonlines
        with jsonlines.open(os.path.join(self.data_dir, f"{split}.jsonl")) as f:
            for data in f.iter():
                text = data['text']
                en_summary = data['en-summary']
                zh_summary = data['summary']
                
                meta_en = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(en_summary).tokenization),"tgt_lang": "en", "task": "sum"}
                meta_zh = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(zh_summary).tokenization),"tgt_lang": "zh", "task": "sum"}
                count += 1
                example_en = InputExample(guid=count, text_a=text, text_b=en_summary, meta=meta_en)
                example_list.append(example_en)

                if split == 'train':
                    count += 1
                    example_zh = InputExample(guid=count, text_a=text, text_b=zh_summary, meta=meta_zh)
                    example_list.append(example_zh)
                    
                
                if count % 100000 == 0:
                    print_rank_0(en_summary)
        """            
        if split == 'train':
            return example_list[:20000]
        else:
            return example_list[:5]
        """

        return example_list

class NCLSProcessor(DataProcessor):
    
    def __init__(self, data_dir, tokenizer, max_src_length, multitask, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args
        self.multitask = multitask
    

    def create_examples(self, split):

        if split == "train":
            filenames = ["train.txt", "../data_en2en/train.txt"]
        elif split == "dev":
            filenames = ["dev.txt"]
        elif split == "test":
            filenames = ["test.txt"]
        else:
            raise NotImplementedError(split)

        count = 0
        example_list = []
    
        for filename in filenames[:1]:
            if 'en2en' in filename: 
                lang = 'en'
            else:
                lang = 'zh'
            
            with open(os.path.join(self.data_dir, filename), encoding = 'utf-8') as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    line = punctuation_standardization(line)
                    if index % 2 == 0:
                        text_a = line
                    else:
                        text_b = line
                        count += 1

                        if count % 1000 == 0:
                            #print_rank_0(text_a)
                            print_rank_0(text_b)

                        example = InputExample(guid=count, text_a=text_a, text_b=text_b, meta={'tgt_lang':lang, 'ref':text_b, 'task':'sum'})
                        example_list.append(example)
        
        #random.shuffle(example_list)
        #if split == 'train' and len(filenames) == 2: 
        #    example_list = example_list[:len(example_list)//2]
        
        #small sample for debugging
        if split == 'train':
            return example_list[:40000]
        else:
            return example_list[:200]
          
        return example_list


class AlltoZHProcessor(DataProcessor):

    def __init__(self, data_dir, tokenizer, max_src_length, multitask, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args
        self.multitask = multitask
    

    def create_examples(self, split):

        if split == "train":
            filenames = ["train.txt", "../data_en2en/train.txt"]
        elif split == "dev":
            filenames = ["dev.txt"]
        elif split == "test":
            filenames = ["test.txt"]
        else:
            raise NotImplementedError(split)

        count = 0
        example_list = []
        
        if split == 'train' or split == 'dev' or split == 'test':
            f = open("/mnt/yrfs/litianjian/datasets/scisummnet-zh.txt", "r")
            scicnt = 0
            
            for line in f.readlines():
                count += 1
                scicnt += 1
                if split == 'train' and scicnt == 800:
                    break
                elif split == 'dev' or split == 'test':
                    if scicnt <= 800:
                        continue
                line = ast.literal_eval(line)
                pid = line['pid']
                target = line['zh_sum']
                sourcefile = f"/mnt/yrfs/litianjian/datasets/scisummnet/top1000_complete/{pid}/Documents_xml/{pid}.xml"


                source = ""
                infile = open(sourcefile,"r")
                contents = infile.read()
                soup = bs(contents,'xml')
                texts = soup.find_all('SECTION')
                for section in texts:
                    #print(section.get('title'))
                    #txt = txt + '\n' + section.get('title') + '\n'
                    temp = section.find_all('S')
                    for data in temp:
                        source = source + data.get_text()
                
                example = InputExample(guid=count, text_a=source, text_b=target, meta={'tgt_lang':'zh', 'ref':target, 'task':'sum'})
                print_rank_0(target)
                example_list.append(example)
            
            if split == 'dev' or split == 'test':
                return example_list



        if self.multitask:
            for filename in filenames[:1]:
                filename = "../both/" + filename
                with open(os.path.join(self.data_dir, filename), encoding='utf-8') as f:
                    for index, line in enumerate(f):
                        line = line.strip()
                        line = punctuation_standardization(line)
                    
                        if index % 3 == 0:
                            content = line
                        elif index % 3 == 1:
                            en_summary = line
                        else:
                            count += 1
                            zh_summary = line
                            if count % 4000 == 0:
                                print_rank_0(f"content = {content}")
                                print_rank_0(f"en_summary = {en_summary}")
                                print_rank_0(f"zh_summary = {zh_summary}")
                            example = InputExample(guid=count, text_a=content, text_b=zh_summary, meta={'tgt_lang':'zh', 'ref':zh_summary, 'task':'sum'})
                            example_list.append(example)
                            if split == "train":
                                
                                example = InputExample(guid=count, text_a=en_summary, text_b=zh_summary, meta={'tgt_lang':'zh', 'ref':zh_summary, 'task':'trans'})
                                example_list.append(example)
                                
                                #example = InputExample(guid=count, text_a=content, text_b=en_summary, meta={'tgt_lang':'en', 'ref':en_summary, 'task':'sum'})
                                #example_list.append(example)
                                 
            random.shuffle(example_list)
            #if split == "train":
            #    end = len(example_list)*2 // 3
            #    example_list = example_list[:end]
            return example_list
        

        for filename in filenames[:1]:
            if 'en2en' in filename: 
                lang = 'en'
            else:
                lang = 'zh'
            
            with open(os.path.join(self.data_dir, filename), encoding = 'utf-8') as f:
                for index, line in enumerate(f):
                    line = line.strip()
                    line = punctuation_standardization(line)
                    if index % 2 == 0:
                        text_a = line
                    else:
                        text_b = line
                        count += 1

                        if count % 1000 == 0:
                            #print_rank_0(text_a)
                            print_rank_0(text_b)

                        example = InputExample(guid=count, text_a=text_a, text_b=text_b, meta={'tgt_lang':lang, 'ref':text_b, 'task':'sum'})
                        example_list.append(example)
        
        #random.shuffle(example_list)
        #if split == 'train' and len(filenames) == 2: 
        #    example_list = example_list[:len(example_list)//2]
        
        #small sample for debugging
        if split == 'train':
            return example_list[:40000]
        else:
            return example_list[:200]
          
        return example_list

class WikiLinguaProcesssor(DataProcessor):
    
    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args
    
    def detokenize(self, string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-",
                     "[": "-LSB-", "]": "-RSB-",
                     "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", "\"")
        string = string.replace("``", "\"")
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string
    
    def create_examples(self, split):
        
        assert split in ["train", "dev", "test"]
        
        # train:dev:test = 8:1:1
        print_rank_0("Processing Wikilingua dataset")
        filenames = glob.glob(f"{self.data_dir}/*.pkl")
        with open(f"{self.data_dir}/english.pkl", "rb") as f:
            english_docs=pickle.load(f)
        
        example_list = []
        count = -1

        for filename in filenames:
            lang = filename.split('/')[-1].split('.')[0]
            print_rank_0(f"now processing language {lang}")
            with open(filename, "rb") as f:
                docs = pickle.load(f)
                for url in docs:
                    for title in docs[url]:
                        count += 1
                        
                        if count % 10 != 0 and split == 'dev':
                            continue
                        
                        if count % 10 != 1 and split == 'test':
                            continue 

                        if split == 'train' and (count % 10 == 0 or count % 10 == 1):
                            continue 

                        article = docs[url][title]
                        content = article['document']
                        #summary_src_lang = article['summary']
                        
                        if lang == 'english':
                            summary_en = article['summary']
                        else:
                            en_url = article['english_url']
                            en_title = article['english_section_name']
                            summary_en = english_docs[en_url][en_title]["summary"]
                        #if count % 1000 == 5:
                        #    print_rank_0(summary_en)
                        example = InputExample(guid=count, text_a=content, text_b=summary_en, meta={'src_lang':lang, 'tgt_lang':'en', 'ref':summary_en, 'task':'sum'})
                        example_list.append(example)
        random.shuffle(example_list)
        return example_list

        print_rank_0("Processing CNNDM dataset")
        # Multitask Finetuning - CNNDM + Wikilingua
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "val"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        source_texts, target_texts = [], []
        with open(os.path.join(self.data_dir, f"../cnn_dm/{filename}.source"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line)
                source_texts.append(line)

        with open(os.path.join(self.data_dir, f"../cnn_dm/{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line, is_target=True)
                target_texts.append(line)
        assert len(source_texts) == len(target_texts)
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            guid = "%s-%s" % (split, idx)
            meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            #if idx < 3:
            #    print_rank_0(
            #        (source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
            #yield example
            example_list.append(example)

        random.shuffle(example_list)
        


        return example_list

class XLSumProcessor(DataProcessor):
    
    def detokenize(self, string, is_target=False):
        return string

    def _yield_examples(self, split): #used for lazy loader 
        
        assert split in ["train", "dev", "test"]
        if split == 'dev':
            split = 'val' 
        #import glob
        filenames = glob.glob(f"{self.data_dir}/*_{split}.jsonl")

        count = 0
        import jsonlines
        for filename in filenames:
            #if split == 'train' and 'english' not in filename:
            #    continue

            with jsonlines.open(filename) as f: #jsonlines uses utf-8 as encoding by default
                for data in f.iter():
                    count += 1
                    if count % 10 != 0 and split == 'val':
                        continue
                    title = data['title']
                    summary = data['summary']
                    text = data['text']
                    
                    title = punctuation_standardization(title)
                    summary = punctuation_standardization(summary)
                    text = punctuation_standardization(text)

                    source_text = title + " " + text
                    target_text = summary 

                    meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization),"lang":filename.split('_')[0]}
                    
                    example = InputExample(guid=count, text_a=source_text, text_b=target_text, meta=meta)
                    if count % 10000 == 0: 
                        print_rank_0(target_text) #for debugging
                    
                    yield example 

class SummaryProcessor(DataProcessor):
    def detokenize(self, string, is_target=False):
        return string

    def _yield_examples(self, split):
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "val"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        source_texts, target_texts = [], []
        with open(os.path.join(self.data_dir, f"{filename}.source"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line)
                source_texts.append(line)
        with open(os.path.join(self.data_dir, f"{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = punctuation_standardization(line)
                line = self.detokenize(line, is_target=True)
                target_texts.append(line)
        assert len(source_texts) == len(target_texts)
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            guid = "%s-%s" % (split, idx)
            meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            if idx < 3:
                print_rank_0(
                    (source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
            yield example

class CNNDMProcessor(SummaryProcessor):
    def detokenize(self, string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-",
                     "[": "-LSB-", "]": "-RSB-",
                     "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", "\"")
        string = string.replace("``", "\"")
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string

class MTGCrossSummaryProcessor(SummaryProcessor):
    def detokenize(self, string, is_target=False):
        _tok_dict = {"(": "-LRB-", ")": "-RRB-",
                     "[": "-LSB-", "]": "-RSB-",
                     "{": "-LCB-", "}": "-RCB-"}
        if not is_target:
            string = string.replace("<S_SEP>", "")
        else:
            string = string.replace("<S_SEP>", "[SEP]")
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        string = string.replace("''", "\"")
        string = string.replace("``", "\"")
        string = string.replace("`", "'")
        string = string.replace(" n't", "n't")
        string = string.replace(" 's", "'s")
        string = string.replace(" 'd", "'d")
        string = string.replace(" 'll", "'ll")
        return string
    def _yield_examples(self, split):
        tgt_lang = "zh"
        train_lang = "en"
        if split == "train":
            source_texts, target_texts = [], []
            with open(os.path.join(self.data_dir, f"train.doc."+train_lang)) as file:
                for line in file:
                    text = self.detokenize(punctuation_standardization(line.strip()))
                    source_texts.append(text)
            with open(os.path.join(self.data_dir, f"train.sum."+tgt_lang)) as file:
                for line in file:
                    text = self.detokenize(punctuation_standardization(line.strip()))
                    target_texts.append(text)
            assert len(source_texts) == len(target_texts)
        elif split == "dev":
            source_texts, target_texts = [], []
            for lang in ['en','de','es','fr','zh']:
                if lang == train_lang:
                    continue
                with open(os.path.join(self.data_dir, "dev.annotation.doc."+lang)) as file:
                    for line in file:
                        text = self.detokenize(punctuation_standardization(line.strip()))
                        source_texts.append(text)
                with open(os.path.join(self.data_dir, "dev.annotation.sum."+tgt_lang)) as file:
                    for line in file:
                        text = self.detokenize(punctuation_standardization(line.strip()))
                        target_texts.append(text)
            assert len(source_texts) == len(target_texts)
        elif split == "test":
            source_texts, target_texts = [], []
            for lang in ['en','de','es','fr','zh']:
                if lang == train_lang:
                    continue
                with open(os.path.join(self.data_dir, "test.annotation.doc."+lang)) as file:
                    for line in file:
                        text = self.detokenize(punctuation_standardization(line.strip()))
                        source_texts.append(text)
                with open(os.path.join(self.data_dir, "test.annotation.sum."+tgt_lang)) as file:
                    for line in file:
                        text = self.detokenize(punctuation_standardization(line.strip()))
                        target_texts.append(text)
            assert len(source_texts) == len(target_texts)
        else:
            raise NotImplementedError(split)
        idxs = [i for i in range(len(source_texts))]
        random.shuffle(idxs)
        random.shuffle(idxs)
        source_texts = [source_texts[i] for i in idxs]
        target_texts = [target_texts[i] for i in idxs]
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            guid = "%s-%s" % (split, idx)
            meta = {"tgt_lang": tgt_lang, "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            if idx < 3:
                print_rank_0((source_text, target_text, meta["ref"]))
            yield example


class GGWProcessor(SummaryProcessor):
    def detokenize(self, string, is_target=False):
        _tok_dict = {"(": "-lrb-", ")": "-rrb-",
                     "[": "-lsb-", "]": "-rsb-",
                     "{": "-lcb-", "}": "-rcb-",
                     '&': '&amp;', '<': '&lt;', '>': '&gt;'}
        string = string.replace('UNK', '[UNK]')
        string = string.replace('<unk>', '[UNK]')
        for key, value in _tok_dict.items():
            string = string.replace(value, key)
        # string = string.replace("''", "\"")
        # string = string.replace("``", "\"")
        # string = string.replace("`", "'")
        # string = string.replace(" n't", "n't")
        # string = string.replace(" 's", "'s")
        # string = string.replace(" 'd", "'d")
        # string = string.replace(" 'll", "'ll")
        return string


class CMRCProcessor(DataProcessor):
    def _yield_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        idx = 0
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
            dataset = json.load(file)
            for article in dataset['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer['text'] for answer in qa["answers"]} if split != 'test' else {"FAKE_ANSWER"}
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer": answer,
                                "question": question,
                                "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(answer).tokenization)}
                            example = InputExample(guid=guid, text_a=context, meta=meta)
                            if idx < 10:
                                print_rank_0(
                                    (context.encode('utf-8'), answer.encode('utf-8'), meta["ref"].encode('utf-8')))
                            yield example
                            idx += 1


class SQuADQGProcessor(DataProcessor):
    def _yield_examples(self, split):
        if split == "train":
            filename = "train.json"
        elif split == "dev":
            filename = "dev.json"
        elif split == "test":
            filename = "test.json"
        else:
            raise NotImplementedError(split)
        idx = 0
        with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
            dataset = json.load(file)
            for paragraphs in dataset:
                for paragraph in paragraphs['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa["question"]
                        answers = {answer["text"] for answer in qa["answers"]}
                        answer_starts = {answer["text"]: answer["answer_start"] for answer in qa["answers"]}
                        for answer in answers:
                            guid = "%s-%s" % (split, idx)
                            meta = {
                                "answer_start": answer_starts[answer],
                                "answer": answer,
                                "question": question,
                                "ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(question).tokenization)}
                            example = InputExample(guid=guid, text_a=context, meta=meta)
                            if idx < 3:
                                print_rank_0(
                                    (context.encode('utf-8'), answer.encode('utf-8'), meta["ref"].encode('utf-8')))
                            yield example
                            idx += 1



class MLQAProcessor(DataProcessor):
    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args

    def create_examples(self, split):
        
        if split == "train": #The Training Set of MLQA is SQuAD English
            filename = "train-v1.1.json" 
            filenames = [filename]
     
        elif split == 'dev':
            filenames = ['../mlqa/MLQA_V1/dev/'+ x for x in os.listdir('/mnt/yrfs/litianjian/mGLM/multi-finetune/xtreme-master/download/mlqa/MLQA_V1/dev')]
        elif split == 'test':
            filenames = ['../mlqa/MLQA_V1/test/'+ x for x in os.listdir('/mnt/yrfs/litianjian/mGLM/multi-finetune/xtreme-master/download/mlqa/MLQA_V1/test')]
        else:
            raise NotImplementedError(split)
        
        print_rank_0(f"Creating MLQA-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        total_qas = 0
        total_na = 0
        
                    
        for filename in filenames:
            with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
                if split == 'dev' or split == 'test':
                    src_lang = filename.split('-')[2]
                else:
                    src_lang = 'en'
                dataset = json.load(file)['data']
                for paragraphs in dataset:
                    for paragraph in paragraphs['paragraphs']:
                        context = paragraph['context']
                        context_tokens = self.tokenizer.EncodeAsIds(context).tokenization
                        token_to_char = None
                        for qa in paragraph['qas']:
                            total_qas += 1
                            question = qa["question"]
                            question_tokens = self.tokenizer.EncodeAsIds(" " + question).tokenization
                            answers = [answer["text"] for answer in qa["answers"]]
                            if len(qa['answers']) == 0:
                                answers = ['N/A']
                            for start in range(0, len(context_tokens), self.max_src_length // 2):
                                length = self.max_src_length - 3 - len(question_tokens)
                                tokens = context_tokens[start:start + length]
                                new_context = self.tokenizer.DecodeIds(tokens)
                                answer = answers[0]
                                answer_tokens_text = self.tokenizer.DecodeIds(
                                    self.tokenizer.EncodeAsIds(answer).tokenization)
                                if answer_tokens_text and answer_tokens_text in new_context:
                                    # new_context = new_context.replace(answer_tokens_text, answer)
                                    pass
                                else:
                                    answer = 'N/A'
                                if self.task == 'mlqa' and answer == 'N/A':
                                    continue
                                guid = "%s-%s" % (split, idx)
                                meta = {
                                    "context": context,
                                    "context_tokens": context_tokens,
                                    "token_to_char": token_to_char,
                                    "answer": answer,
                                    "answers": answers,
                                    "question": question,
                                    "ref": answer,
                                    "language":src_lang
                                }
                                example = InputExample(guid=guid, text_a=new_context, meta=meta, idx=qa['id'])
                                example_list.append(example)
                                idx += 1
                                total_na += (answer == 'N/A')
                                if len(tokens) < length:
                                    break
        

        print_rank_0(f"Creating {len(example_list)} / {total_qas} examples for {split}. {total_na} N/A")
        return example_list

class TyDiQAProcessor(DataProcessor):
    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args

    def create_examples(self, split):
        
        if split == "train": 
            filenames = ['tydiqa-goldp-v1.1-train/' + x for x in os.listdir('/mnt/yrfs/litianjian/mGLM/multi-finetune/xtreme-master/download/tydiqa/tydiqa-goldp-v1.1-train/')]
        elif split == 'dev' or split == 'test':
            filenames = ['tydiqa-goldp-v1.1-dev/' + x for x in os.listdir('/mnt/yrfs/litianjian/mGLM/multi-finetune/xtreme-master/download/tydiqa/tydiqa-goldp-v1.1-dev/')]
        else:
            raise NotImplementedError(split)
        
        print_rank_0(f"Creating TyDiQA-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        total_qas = 0
        total_na = 0
        
                    
        for filename in filenames:
            with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
                if split == "train":
                    src_lang = filename.split('.')[-3]
                    print(src_lang)
                    if src_lang != 'en': #zero-shot: Trained only on English data 
                        continue        
                else:
                    src_lang = filename.split('-')[-1][:-5]
                dataset = json.load(file)['data']
                for paragraphs in dataset:
                    for paragraph in paragraphs['paragraphs']:
                        context = paragraph['context']
                        context_tokens = self.tokenizer.EncodeAsIds(context).tokenization
                        token_to_char = None
                        for qa in paragraph['qas']:
                            total_qas += 1
                            question = qa["question"]
                            question_tokens = self.tokenizer.EncodeAsIds(" " + question).tokenization
                            answers = [answer["text"] for answer in qa["answers"]]
                            if len(qa['answers']) == 0:
                                answers = ['N/A']
                            for start in range(0, len(context_tokens), self.max_src_length // 2):
                                length = self.max_src_length - 3 - len(question_tokens)
                                tokens = context_tokens[start:start + length]
                                new_context = self.tokenizer.DecodeIds(tokens)
                                answer = answers[0]
                                answer_tokens_text = self.tokenizer.DecodeIds(
                                    self.tokenizer.EncodeAsIds(answer).tokenization)
                                if answer_tokens_text and answer_tokens_text in new_context:
                                    # new_context = new_context.replace(answer_tokens_text, answer)
                                    pass
                                else:
                                    answer = 'N/A'
                                if self.task == 'mlqa' and answer == 'N/A':
                                    continue
                                guid = "%s-%s" % (split, idx)
                                meta = {
                                    "context": context,
                                    "context_tokens": context_tokens,
                                    "token_to_char": token_to_char,
                                    "answer": answer,
                                    "answers": answers,
                                    "question": question,
                                    "ref": answer,
                                    "language":src_lang
                                }
                                example = InputExample(guid=guid, text_a=new_context, meta=meta, idx=qa['id'])
                                example_list.append(example)
                                idx += 1
                                total_na += (answer == 'N/A')
                                if len(tokens) < length:
                                    break
        

        print_rank_0(f"Creating {len(example_list)} / {total_qas} examples for {split}. {total_na} N/A")
        return example_list




class SQuADProcessor(DataProcessor):
    def __init__(self, data_dir, tokenizer, max_src_length, args):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_src_length = max_src_length
        self.task = args.task
        self.args = args
        import transformers
        tokenizer_model_type = self.args.tokenizer_model_type
        if tokenizer_model_type == 'roberta':
            tokenizer_model_type = 'roberta-large'
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        #self.transformer_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model_type)

    def create_examples(self, split):

        if split == "train":
            filename = "train-v1.1.json" if self.task == "squad_v1" else "train-v2.0.json"
            filenames = [filename]

        #XQuAD Dataset, validatedon SQuAD(English), testing on XQuAD(Multilingual)
        
        #elif split == "dev": 
        #    filename = "dev-v1.1.json" if self.task == "squad_v1" else "dev-v2.0.json"
        #    filenames = [filename]
        
        elif split == "test" or split == 'dev':
            filenames = ['../xquad/' + x for x in os.listdir("/mnt/yrfs/litianjian/mGLM/multi-finetune/xtreme-master/download/xquad")]
            
        
        # SQuAD Test Set

        # elif split == "test":
        # filename = "dev-v1.1.json" if self.task == "squad_v1" else "dev-v2.0.json"
        # filenames = [filename]

        
        print_rank_0(f"Creating SQuAD-{split} dataset from {self.data_dir}")
        example_list = []
        idx = 0
        total_qas = 0
        total_na = 0
        
                    
        for filename in filenames:
            with open(os.path.join(self.data_dir, filename), encoding='utf-8') as file:
                if split == 'test':
                    print(filename)
                    src_lang = filename.split('.')[-2]
                else:
                    src_lang = 'en'
                dataset = json.load(file)['data']
                for paragraphs in dataset:
                    for paragraph in paragraphs['paragraphs']:
                        context = paragraph['context']
                        context_tokens = self.tokenizer.EncodeAsIds(context).tokenization
                        token_to_char = None
                        """
                        transformer_encode = self.transformer_tokenizer(context,
                                                                        return_offsets_mapping=True,
                                                                        add_special_tokens=False,
                                                                        verbose=False)
                        assert transformer_encode['input_ids'] == context_tokens
                        token_to_char = transformer_encode['offset_mapping']
                        # if self.tokenizer_type == 'BertWordPieceTokenizer':
                        #     token_to_char = generate_token_to_char_map(context_tokens, context, self.tokenizer)
                        # else:
                        #     token_to_char = None
                        """

                    
                        for qa in paragraph['qas']:
                            total_qas += 1
                            question = qa["question"]
                            question_tokens = self.tokenizer.EncodeAsIds(" " + question).tokenization
                            answers = [answer["text"] for answer in qa["answers"]]
                            if len(qa['answers']) == 0:
                                answers = ['N/A']
                            for start in range(0, len(context_tokens), self.max_src_length // 2):
                                length = self.max_src_length - 3 - len(question_tokens)
                                tokens = context_tokens[start:start + length]
                                new_context = self.tokenizer.DecodeIds(tokens)
                                answer = answers[0]
                                answer_tokens_text = self.tokenizer.DecodeIds(
                                    self.tokenizer.EncodeAsIds(answer).tokenization)
                                if answer_tokens_text and answer_tokens_text in new_context:
                                    # new_context = new_context.replace(answer_tokens_text, answer)
                                    pass
                                else:
                                    answer = 'N/A'
                                if self.task == 'squad_v1' and answer == 'N/A':
                                    continue
                                guid = "%s-%s" % (split, idx)
                                meta = {
                                    "context": context,
                                    "context_tokens": context_tokens,
                                    "token_to_char": token_to_char,
                                    "answer": answer,
                                    "answers": answers,
                                    "question": question,
                                    "ref": answer,
                                    "language":src_lang
                                }
                                example = InputExample(guid=guid, text_a=new_context, meta=meta, idx=qa['id'])
                                example_list.append(example)
                                idx += 1
                                total_na += (answer == 'N/A')
                                if len(tokens) < length:
                                    break
        

        print_rank_0(f"Creating {len(example_list)} / {total_qas} examples for {split}. {total_na} N/A")
        return example_list


class XSumProcessor(DataProcessor):
    def _yield_examples(self, split):
        if split == "train":
            key = "train"
        elif split == "dev":
            key = "validation"
        elif split == "test":
            key = "test"
        else:
            raise NotImplementedError(split)
        with open(os.path.join(self.data_dir, "XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json")) as file:
            id_list = json.load(file)
        id_list = id_list[key]
        for i, idx in enumerate(id_list):
            with open(os.path.join(self.data_dir, f"{idx}.summary")) as file:
                key, sentences = None, []
                source_text, target_text = None, None
                for line in file:
                    line = line.strip()
                    if line.startswith("[SN]"):
                        if key is not None:
                            if key == "RESTBODY":
                                source_text = " ".join(sentences)
                            elif key == "FIRST-SENTENCE":
                                target_text = " ".join(sentences)
                        key = line[4:-4]
                        sentences = []
                    elif line:
                        sentences.append(line)
                if key is not None:
                    if key == "RESTBODY":
                        source_text = " ".join(sentences)
                    elif key == "FIRST-SENTENCE":
                        target_text = " ".join(sentences)
                guid = "%s-%s" % (split, i)
                meta = {"ref": self.tokenizer.DecodeIds(self.tokenizer.EncodeAsIds(target_text).tokenization)}
                example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
                if i < 3:
                    print_rank_0(
                        (source_text.encode('utf-8'), target_text.encode('utf-8'), meta["ref"].encode('utf-8')))
                yield example


PROCESSORS = {
    "gigaword": GGWProcessor,
    "cnn_dm": CNNDMProcessor,
    "cnn_dm_original": SummaryProcessor,
    "xsum": XSumProcessor,
    "squad_generation": SQuADQGProcessor,
    "cmrc": CMRCProcessor,
    "xlsum": XLSumProcessor,
    "mtg_crosssum": MTGCrossSummaryProcessor
}


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        self.task, self.data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.dataset_name = split
        if self.task in ["squad", "squad_v1"]:
            self.processor = SQuADProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task  == 'mlqa':
            self.processor = MLQAProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task == 'tydiqa':
            self.processor = TyDiQAProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task == 'wikilingua':
            self.processor = WikiLinguaProcesssor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task == 'scitldr':
            self.processor = SciTLDRProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task == 'xwikis':
            self.processor = XWikisProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task == 'lcsts':
            self.processor = LCSTSProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task == 'ensum':
            self.processor = AlltoEnMultitaskProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        #elif self.task == 'mtg_crosssum':
        #    self.processor = MTGCrossSummaryProcessor(self.data_dir, tokenizer, self.max_src_length, args)
        elif self.task in ['ncls', 'ncls_multitask']:
            if self.task == 'ncls':
                multitask = False
            else:
                multitask = True
            self.processor = AlltoZHProcessor(self.data_dir, tokenizer, self.max_src_length, multitask, args)
        elif self.task in PROCESSORS:
            self.processor = PROCESSORS[self.task](self.data_dir, tokenizer,
                                                   lazy_seq2seq_loader=args.lazy_seq2seq_loader)
        else:
            raise NotImplementedError(self.task)

        example_list = self.processor.create_examples(split)
        self.example_list = example_list
        self.examples = {example.guid: example for example in example_list}

        print_rank_0(f"Return {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        if self.task in ["squad", "squad_v1", 'tydiqa']:
            cls_id = self.tokenizer.get_command('ENC').Id
            mask_id = self.tokenizer.get_command('sMASK').Id
            source_text = example.text_a
            target_text = example.meta["answer"].strip()
            question = example.meta["question"].strip()
            source_tokens = self.tokenizer.EncodeAsIds(" " + source_text.rstrip()).tokenization
            question_tokens = self.tokenizer.EncodeAsIds(" " + question).tokenization
            period_id = self.tokenizer.TokenToId('.')
            max_src_length = self.max_src_length - len(question_tokens) - 3
            if max_src_length <= 0:
                print(question)
            assert max_src_length > 0
            source_tokens = [cls_id] + question_tokens + [mask_id, period_id] + source_tokens[:max_src_length]
        elif self.task in PVPS:
            pvp = PVPS[self.task](self.tokenizer, max_src_length=self.max_src_length,
                                  max_tgt_length=self.max_tgt_length, task_mask=self.args.task_mask)
            mask_id = pvp.mask_id
            source_tokens, target_text = pvp.encode(example)
        else:
            raise NotImplementedError
        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [pad_id] * (self.max_src_length - len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)
        if self.split == 'train':
            target_tokens = self.tokenizer.EncodeAsIds(" " + target_text).tokenization
            target_tokens = target_tokens + [eop_id]
            if len(target_tokens) > self.max_tgt_length:
                target_tokens = target_tokens[:self.max_tgt_length]
            loss_mask = [1] * len(target_tokens)
            if len(target_tokens) < self.max_tgt_length:
                loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
                target_tokens += [pad_id] * (self.max_tgt_length - len(target_tokens))
            tokens = source_tokens + [sop_id] + target_tokens[:-1]
            loss_mask = [0] * len(source_tokens) + loss_mask
            target_ids = [0] * len(source_tokens) + target_tokens
            position_ids += [mask_pos] * len(target_tokens)
            if self.args.no_block_position:
                block_position_ids += [1] * len(target_tokens)
            else:
                block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample


class ExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        task, data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        self.tokenizer = tokenizer
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "valid"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.source"),
                  encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                source_texts.append(line)
        with open(os.path.join(data_dir, f"{filename}.target"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                target_texts.append(line)
        self.examples, self.example_list = {}, []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        source_text, target_text = example.text_a, example.text_b
        mask_token = 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        pad_id = self.tokenizer.get_command('pad').Id

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        masked_tgt = target_text.split("|")
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        if self.split == 'train':
            mask_positions = [i for i, x in enumerate(source_tokens) if x == mask_id]
            assert len(mask_positions) <= len(masked_tgt)
            tokens = source_tokens
            target_ids = [0] * len(source_tokens)
            loss_mask = [0] * len(source_tokens)
            for i, mask_pos in enumerate(mask_positions):
                tgt_text = masked_tgt[i]
                tgt_tokens = self.tokenizer.EncodeAsIds(" " + tgt_text).tokenization
                tokens += [sop_id] + tgt_tokens
                target_ids += tgt_tokens + [eop_id]
                loss_mask += [1] * (len(tgt_tokens) + 1)
                position_ids += [mask_pos] * (len(tgt_tokens) + 1)
                block_position_ids += [i + 1 for i in range(len(tgt_tokens) + 1)]
            tokens = pad_to(tokens, self.max_src_length + self.max_tgt_length, pad_id)
            target_ids = pad_to(target_ids, self.max_src_length + self.max_tgt_length, pad_id)
            loss_mask = pad_to(loss_mask, self.max_src_length + self.max_tgt_length, 0)
            position_ids = pad_to(position_ids, self.max_src_length + self.max_tgt_length, 0)
            block_position_ids = pad_to(block_position_ids, self.max_src_length + self.max_tgt_length, 0)
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            mask_pos = source_tokens.index(mask_id)
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample


class BlankLMDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, tokenizer):
        self.args = args
        task, data_dir = args.task.lower(), args.data_dir
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.split = split
        assert args.tokenizer_type == "BertWordPieceTokenizer"
        self.tokenizer = tokenizer
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "valid"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        print_rank_0(f"Creating {task}-{split} dataset from {data_dir}")
        self.dataset_name = split
        detokenizer = blanklm_detokenize
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.txt"), encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                line = detokenizer(line) if detokenizer else line
                target_texts.append(line)
        if split == 'test':
            with open(os.path.join(data_dir, f"blank/test.maskratio{args.blank_maskratio:.1f}.blank"),
                      encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    line = detokenizer(line) if detokenizer else line
                    source_texts.append(line)
        else:
            source_texts = target_texts
        self.examples, self.example_list = {}, []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            # if idx > 10000:
            #     break
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % (split, idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} {split} examples")
        self.random = random.Random(args.seed)

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        source_text, target_text = example.text_a, example.text_b
        mask_token = 'gMASK' if self.args.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        pad_id = self.tokenizer.get_command('pad').Id
        if self.split in ['train', 'dev']:
            masked_src, masked_tgt = self.mask_text(source_text)
            source_text = masked_src

        def pad_to(text, max_len, pad_id):
            if len(text) > max_len:
                text = text[:max_len]
            else:
                text = text + [pad_id] * (max_len - len(text))
            return text

        source_tokens = self.tokenizer.EncodeAsIds(" " + source_text).tokenization
        source_tokens = pad_to(source_tokens, self.max_src_length, pad_id)
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        if self.split in ['train', 'dev']:
            mask_positions = [i for i, x in enumerate(source_tokens) if x == mask_id]
            assert len(mask_positions) <= len(masked_tgt)
            tokens = source_tokens
            target_ids = [0] * len(source_tokens)
            loss_mask = [0] * len(source_tokens)
            for i, mask_pos in enumerate(mask_positions):
                tgt_text = masked_tgt[i]
                tgt_tokens = self.tokenizer.EncodeAsIds(" " + tgt_text).tokenization
                tokens += [sop_id] + tgt_tokens
                target_ids += tgt_tokens + [eop_id]
                loss_mask += [1] * (len(tgt_tokens) + 1)
                position_ids += [mask_pos] * (len(tgt_tokens) + 1)
                block_position_ids += [i + 1 for i in range(len(tgt_tokens) + 1)]
            max_length = self.max_src_length + int(self.max_src_length * self.args.blank_maskratio)
            tokens = pad_to(tokens, max_length, pad_id)
            target_ids = pad_to(target_ids, max_length, pad_id)
            loss_mask = pad_to(loss_mask, max_length, 0)
            position_ids = pad_to(position_ids, max_length, 0)
            block_position_ids = pad_to(block_position_ids, max_length, 0)
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                      'attention_mask': np.array(sep, dtype=np.int64),
                      'loss_mask': np.array(loss_mask, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        else:
            tokens = source_tokens + [sop_id]
            mask_pos = source_tokens.index(mask_id)
            position_ids = position_ids + [mask_pos]
            block_position_ids = block_position_ids + [1]
            position_ids = [position_ids, block_position_ids]
            sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                      "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample

    def mask_text(self, text):
        tokens = text.split()
        mask_ratio = self.args.blank_maskratio
        n = len(tokens)
        indices = sorted(self.random.sample(range(n), int(n * mask_ratio)))
        masked_src, masked_tgt = "", []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i - 1] + 1:
                masked_tgt.append("")
            masked_tgt[-1] += " " + tokens[idx]
            tokens[idx] = "[MASK]"
        for i, token in enumerate(tokens):
            if i != 0 and token == "[MASK]" and tokens[i - 1] == "[MASK]":
                continue
            masked_src += " " + token
        return masked_src, masked_tgt
