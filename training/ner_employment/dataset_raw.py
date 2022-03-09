from torch.utils.data import Dataset, DataLoader
from nerus import load_nerus
import emoji
import re
import os
import torch
from tqdm import tqdm
import json
import yaml
import random
import time
import pandas as pd
tqdm.pandas()

from torch.utils.data import Dataset, DataLoader
from nerus import load_nerus
import emoji
import re
import os
import torch
from tqdm import tqdm
import json
import yaml
import random
import pandas as pd
tqdm.pandas()

from qrnn.hash_tools import (
    murmurhash,
    simhash_token,
    SimhashToken
)

class NERDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data_len = int(data.shape[0] * (1-test_size))
        if phase == 'train':
            data = data.iloc[:data_len]
        if phase == 'test':
            data = data.iloc[data_len:]
            #data = data.iloc[int(data.shape[0] * (1-0.2/8)):]
                
        self.data = data.reset_index()
        self.transforms = transforms
        self.tag2index = {
            'O': 0,
            'PER': 1,
            'LOC': 2,
            'ORG': 3,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
            
        if add_data is not None:
            self.length = len(self.data)
            self.max_seq_len = min(add_data['max_seq_len'], 512)
            self.vocab = add_data['vocab']
        elif os.path.exists(f'/root/konst/ml-utils/training/ner_employment/hash_{self.phase}_params.json'):
            with open(f'/root/konst/ml-utils/training/ner_employment/hash_{self.phase}_params.json', 'r') as f:
                data = yaml.safe_load(f.read())
            self.length = len(self.data)
            if phase == 'test':
                with open(f'/root/konst/ml-utils/training/ner_employment/hash_train_params.json', 'r') as f:
                    data = yaml.safe_load(f.read())
            
            self.max_seq_len = min(data['max_seq_len'], 512)
            self.vocab = data['vocab']
        else:
            data['spans'] = data.spans.progress_apply(yaml.safe_load)
            print(f'Start vocab building {phase}...')
            self.create_vocab()
            print('Finish')
        
        
    def create_vocab(self):
        distinct_words = set()
        max_seq_len = -1
        for row in tqdm(self.data.itertuples(), total=len(self.data)):            
            text = row.text.lower()
            spans = row.spans
            if self.transforms:
                text = self.transforms(text)
            tokens = text.split()
            
            if len(tokens) > max_seq_len:
                max_seq_len = len(tokens)
                
            [distinct_words.add(token) for token in tokens]
        
        self.length = len(self.data)
        self.max_seq_len = min(max_seq_len, 512)
        distinct_words = [self.pad_tag] + list(distinct_words)
        self.vocab = {w: idx for idx, w in enumerate(distinct_words)}
        
        data = {
            'length': self.length,
            'max_seq_len': max_seq_len,
            'vocab': self.vocab
        }
        with open(f'hash_{self.phase}_params.json', 'w') as f:
            f.write(str(data))
        
    def get_vocab_size(self):
        return len(self.token2idx)
    
    def __len__(self):
        return self.length
    
    def smart_split(self, text, spans):
        begin_idxs = {s['start'] for s in spans if spans}
        counter = 0
        for i, s in enumerate(text):
            if i in begin_idxs:
                span = spans[counter]
                stop = span['stop']
                #text[i: stop] = text[i: stop].replace(' ', self.union_tag)
                tag = text[i: stop].replace(' ', self.union_tag)
                text = text[:i] + tag + text[stop:]
                counter += 1
                
        tokens = text.split()
        tokens = [t.replace(self.union_tag, ' ') for t in tokens]
        
        return tokens
    
    def find_similar(self, search_token):
        print('Finding similar token...')
        print(search_token)
        return random.choice(list(self.vocab.keys()))
        
        search_token_split = search_token.split()
        if len(search_token_split) == 2:
            token1, token2 = search_token_split
            if token1 in self.vocab:
                return token1
            if token2 in self.vocab:
                return token2
            
        for token in self.vocab:
            
            if token in search_token and len():
                pass
                
    
    def __getitem__(self, idx):
        text = self.data.at[idx, 'text']
        spans = self.data.at[idx, 'spans']
        if not isinstance(spans, list):
            spans = yaml.safe_load(spans)
            self.data.at[idx, 'spans'] = spans
        
        if self.transforms:
            text_trans = self.transforms(text)
        
        #print(text)
        #print()
        #print(spans)
        #print()
        tokens = text_trans.split()
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = torch.LongTensor([
            self.vocab[
                token if token in self.vocab 
                else random.choice(list(self.vocab.keys()))
            ] for token in curr_tokens
        ])
        #for token in curr_tokens:
        #    if token not in self.vocab:
        #        print(token)
        
        #for span in spans:
        #    print(text[span['start']: span['stop']].lower(), span['type'])
        #print()
        
        spans_extend = []
        for span in spans:
            text_part = text[span['start']: span['stop']]
            tokens = text_part.split()
            if len(tokens) == 1:
                spans_extend.append(span)
            elif len(tokens) == 2:
                token1, token2 = tokens
                spans_extend.extend([
                    {
                        'start': span['start'],
                        'stop': span['start'] + len(token1),
                        'type': span['type'],
                    },
                    {
                        'start': span['start'] + len(token1) + 1,
                        'stop': span['stop'],
                        'type': span['type'],                        
                    }
                ])
            else:
                continue
                #raise Exception(f'{text_part}, {tokens}')
                        
        #for span in spans_extend:
        #    print(text[span['start']: span['stop']].lower(), span['type'])
        
        #print()
        
        if len(spans_extend) == 0:
            tags = torch.LongTensor([self.tag2index['O']]*len(tensor))
            
            mask = [int(token != self.pad_tag) for token in curr_tokens]
            mask = torch.LongTensor(mask)

            return tensor, mask, tags

        tags = []        
        last_span_idx = 0
        span = spans_extend[last_span_idx]
        for token in curr_tokens:
            #if token != '<pad>':
            #    print(text[span['start']: span['stop']].lower())
            #    print(token)
            if token == text[span['start']: span['stop']].lower():
                tags.append(span['type'])
                last_span_idx += 1
                if len(spans_extend) >= last_span_idx+1:
                    span = spans_extend[last_span_idx]
            else:
                tags.append('O')
        
        """
        for token in curr_tokens:
            found = False
            for span in spans_extend:
                if token == text[span['start']: span['stop']].lower():
                    tags.append(span['type'])
                    found = True
                    break
            if not found:
                tags.append('O')
        """
        #cstlen = len([t for t in tags if t != 'O'])
        #if cstlen != len(spans):
        #    print(f'Fail: {cstlen} != {len(spans)}')
        tags = torch.LongTensor([
            self.tag2index[tag] for tag in tags 
        ])
        mask = [int(token != self.pad_tag) for token in curr_tokens]
        mask = torch.LongTensor(mask)
        
        return tensor, mask, tags
    

class NERProjDataset1(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data_len = int(data.shape[0] * (1-test_size))
        if phase == 'train':
            #data = data.iloc[:data_len]
            data = data.iloc[:int(data_len*0.85)] # 3433/4621 -- error
        if phase == 'test':
            data = data.iloc[data_len: ]
            #data = data.iloc[int(data.shape[0] * (1-0.2/8)):]
                
        self.data = data.reset_index()
        self.transforms = transforms
        self.tag2index = {
            'O': 0,
            'PER': 1,
            'LOC': 2,
            'ORG': 3,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.max_seq_len = 512
        self.hash_func = SimhashToken(128)
        self.hash_table = {
            self.pad_tag: self.hash_func(self.pad_tag)
        }
        
    def __len__(self):
        return len(self.data)
    
    def get_hash(self, token):
        if token in self.hash_table:
            return self.hash_table[token]
        
        token_hash = self.hash_func(token)
        self.hash_table[token] = token_hash
        return token_hash

    def step1(self, idx):
        text = self.data.at[idx, 'text']
        text_trans = self.data.at[idx, 'text_processed']
        spans = self.data.at[idx, 'spans']
        if not isinstance(spans, list):
            spans = yaml.safe_load(spans)
            self.data.at[idx, 'spans'] = spans
        
        #text_trans = text
        if self.transforms:
            text_trans = self.transforms(text)
        
        tokens = text_trans.split()
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = [
            self.get_hash(token) for token in curr_tokens # murmurhash
        ]
        
        spans_extend = []
        for span in spans:
            text_part = text[span['start']: span['stop']]
            tokens = text_part.split()
            if len(tokens) == 1:
                spans_extend.append(span)
            elif len(tokens) == 2:
                token1, token2 = tokens
                spans_extend.extend([
                    {
                        'start': span['start'],
                        'stop': span['start'] + len(token1),
                        'type': span['type'],
                    },
                    {
                        'start': span['start'] + len(token1) + 1,
                        'stop': span['stop'],
                        'type': span['type'],                        
                    },
                ])
        
        if len(spans_extend) == 0:
            tags = [self.tag2index['O']]*len(tensor)
            
            mask = [int(token != self.pad_tag) for token in curr_tokens]
            #mask = torch.LongTensor(mask)

            return tensor, mask, tags

        tags = []
        last_span_idx = 0
        span = spans_extend[last_span_idx]
        for token in curr_tokens:
            if token == text[span['start']: span['stop']].lower():
                tags.append(span['type'])
                last_span_idx += 1
                if len(spans_extend) >= last_span_idx+1:
                    span = spans_extend[last_span_idx]
            else:
                tags.append('O')
        
        tags = [
            self.tag2index[tag] for tag in tags 
        ]
        mask = [int(token != self.pad_tag) for token in curr_tokens]
        #mask = torch.LongTensor(mask)
        
        return tensor, mask, tags
    
    def __getitem__(self, idx):
        try:
            return self.step(idx)
        except:
            return self.step(idx // 2)

class NERProjDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data_len = int(data.shape[0] * (1-test_size))
        if phase == 'train':
            #data = data.iloc[:data_len]
            data = data.iloc[:int(data_len*0.85)] # 3433/4621 -- error
        if phase == 'test':
            data = data.iloc[data_len: ]
            #data = data.iloc[int(data.shape[0] * (1-0.2/8)):]
                
        self.data = data.reset_index()
        self.transforms = transforms
        self.tag2index = {
            'O': 0,
            'PER': 1,
            'LOC': 2,
            'ORG': 3,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.max_seq_len = 512
        self.hash_func = SimhashToken(128)
        self.hash_table = {
            self.pad_tag: self.hash_func(self.pad_tag)
        }
        
    def __len__(self):
        return len(self.data)
    
    def get_hash(self, token):
        if token in self.hash_table:
            return self.hash_table[token]
        
        token_hash = self.hash_func(token)
        self.hash_table[token] = token_hash
        return token_hash

    def step(self, idx):
        text = self.data.at[idx, 'text']
        text_trans = self.data.at[idx, 'text_processed']
        spans = self.data.at[idx, 'spans']
        if not isinstance(spans, list):
            spans = yaml.safe_load(spans)
            self.data.at[idx, 'spans'] = spans
        
        #text_trans = text
        if self.transforms:
            text_trans = self.transforms(text)
        
        tokens = text_trans.split()
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = [
            self.get_hash(token) for token in curr_tokens # murmurhash
        ]
        
        spans_extend = []
        for span in spans:
            text_part = text[span['start']: span['stop']]
            tokens = text_part.split()
            if len(tokens) == 1:
                spans_extend.append(span)
            elif len(tokens) == 2:
                token1, token2 = tokens
                spans_extend.extend([
                    {
                        'start': span['start'],
                        'stop': span['start'] + len(token1),
                        'type': span['type'],
                    },
                    {
                        'start': span['start'] + len(token1) + 1,
                        'stop': span['stop'],
                        'type': span['type'],                        
                    },
                ])
            elif len(tokens) == 3:
                token1, token2, token3 = tokens
                spans_extend.extend([
                    {
                        'start': span['start'],
                        'stop': span['start'] + len(token1),
                        'type': span['type'],
                    },
                    {
                        'start': span['start'] + len(token1) + 1,
                        'stop': span['start'] + len(token1) + 1 + len(token2),
                        'type': span['type'],                        
                    },
                    {
                        'start': span['start'] + len(token1) + 1 + len(token2) + 1,
                        'stop': span['stop'],
                        'type': span['type'],                        
                    },
                ])
                print(spans_extend[-3:])
        
        if len(spans_extend) == 0:
            tags = [self.tag2index['O']]*len(tensor)
            
            mask = [int(token != self.pad_tag) for token in curr_tokens]
            #mask = torch.LongTensor(mask)

            return tensor, mask, tags

        tags = []
        last_span_idx = 0
        span = spans_extend[last_span_idx]
        for token in curr_tokens:
            if token == text[span['start']: span['stop']].lower():
                tags.append(span['type'])
                last_span_idx += 1
                if len(spans_extend) >= last_span_idx+1:
                    span = spans_extend[last_span_idx]
            else:
                tags.append('O')
        
        tags = [
            self.tag2index[tag] for tag in tags 
        ]
        mask = [int(token != self.pad_tag) for token in curr_tokens]
        #mask = torch.LongTensor(mask)
        
        return tensor, mask, tags
    
    def __getitem__(self, idx):
        try:
            return self.step(idx)
        except:
            return self.step(idx // 2)

class NERProjDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data_len = int(data.shape[0] * (1-test_size))
        if phase == 'train':
            #data = data.iloc[:data_len]
            data = data.iloc[:int(data_len*0.85)] # 3433/4621 -- error
        if phase == 'test':
            data = data.iloc[data_len: ]
            #data = data.iloc[int(data.shape[0] * (1-0.2/8)):]
                
        self.data = data.reset_index()
        self.transforms = transforms
        self.tag2index = {
            'O': 0,
            'PER': 1,
            'LOC': 2,
            'ORG': 3,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.max_seq_len = 512
        self.hash_func = SimhashToken(128)
        self.hash_table = {
            self.pad_tag: self.hash_func(self.pad_tag)
        }
        
    def __len__(self):
        return len(self.data)
    
    def get_hash(self, token):
        if token in self.hash_table:
            return self.hash_table[token]
        
        token_hash = self.hash_func(token)
        self.hash_table[token] = token_hash
        return token_hash

    def step(self, idx):
        text = self.data.at[idx, 'text']
        text_trans = self.data.at[idx, 'text_processed']
        spans = self.data.at[idx, 'spans']
        if not isinstance(spans, list):
            spans = yaml.safe_load(spans)
            self.data.at[idx, 'spans'] = spans
        
        #text_trans = text
        if self.transforms:
            text_trans = self.transforms(text)
        
        tokens = text_trans.split()
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = [
            self.get_hash(token) for token in curr_tokens # murmurhash
        ]
        
        spans_extend = []
        for span in spans:
            text_part = text[span['start']: span['stop']]
            tokens = text_part.split()
            if len(tokens) == 1:
                spans_extend.append(span)
            elif len(tokens) == 2:
                token1, token2 = tokens
                spans_extend.extend([
                    {
                        'start': span['start'],
                        'stop': span['start'] + len(token1),
                        'type': span['type'],
                    },
                    {
                        'start': span['start'] + len(token1) + 1,
                        'stop': span['stop'],
                        'type': span['type'],                        
                    },
                ])
            elif len(tokens) == 3:
                token1, token2, token3 = tokens
                spans_extend.extend([
                    {
                        'start': span['start'],
                        'stop': span['start'] + len(token1),
                        'type': span['type'],
                    },
                    {
                        'start': span['start'] + len(token1) + 1,
                        'stop': span['start'] + len(token1) + 1 + len(token2),
                        'type': span['type'],                        
                    },
                    {
                        'start': span['start'] + len(token1) + 1 + len(token2) + 1,
                        'stop': span['stop'],
                        'type': span['type'],                        
                    },
                ])
                print(spans_extend[-3:])
        
        if len(spans_extend) == 0:
            tags = [self.tag2index['O']]*len(tensor)
            
            mask = [int(token != self.pad_tag) for token in curr_tokens]
            #mask = torch.LongTensor(mask)

            return tensor, mask, tags

        tags = []
        last_span_idx = 0
        span = spans_extend[last_span_idx]
        for token in curr_tokens:
            if token == text[span['start']: span['stop']].lower():
                tags.append(span['type'])
                last_span_idx += 1
                if len(spans_extend) >= last_span_idx+1:
                    span = spans_extend[last_span_idx]
            else:
                tags.append('O')
        
        tags = [
            self.tag2index[tag] for tag in tags 
        ]
        mask = [int(token != self.pad_tag) for token in curr_tokens]
        #mask = torch.LongTensor(mask)
        
        return tensor, mask, tags
    
    def __getitem__(self, idx):
        try:
            return self.step(idx)
        except:
            return self.step(idx // 2)
        
class NERDataset1(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data_len = int(data.shape[0] * 0.8)
        if phase == 'train':
            data = data.iloc[:data_len]
        if phase == 'test':
            data = data.iloc[data_len:]
        #data['spans'] = data.spans.progress_apply(yaml.safe_load)
        self.data = data.reset_index()
        
        self.transforms = transforms
        self.tag2index = {
            #'MISC': 0,
            'PER': 0,
            'LOC': 1,
            'ORG': 2,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
            
        if os.path.exists(f'hash_{self.phase}_params.json'):
            with open(f'hash_{self.phase}_params.json', 'r') as f:
                data = yaml.safe_load(f.read())
            self.length = data['length']
            if phase == 'test':
                with open(f'hash_train_params.json', 'r') as f:
                    data = yaml.safe_load(f.read())
            
            self.max_seq_len = min(data['max_seq_len'], 512)
            self.vocab = data['vocab']
        else:
            print(f'Start vocab building {phase}...')
            self.create_vocab()
            print('Finish')
        
        
    def create_vocab(self):
        distinct_words = set()
        max_seq_len = -1
        for row in tqdm(self.data.itertuples(), total=len(self.data)):            
            text = row.text.lower()
            #tokens = text.split()
            #spans = yaml.safe_load(row.spans)
            spans = row.spans
            tokens = self.smart_split(text, spans)
            if self.transforms:
                tokens = [self.transforms(t) for t in tokens]
            
            if len(tokens) > max_seq_len:
                max_seq_len = len(tokens)
                
            [distinct_words.add(token) for token in tokens]
        
        self.length = len(self.data)
        self.max_seq_len = min(max_seq_len, 512)
        distinct_words = [self.pad_tag] + list(distinct_words)
        self.vocab = {w: idx for idx, w in enumerate(distinct_words)}
        
        data = {
            'length': self.length,
            'max_seq_len': max_seq_len,
            'vocab': self.vocab
        }
        with open(f'hash_{self.phase}_params.json', 'w') as f:
            f.write(str(data))
        
    def get_vocab_size(self):
        return len(self.token2idx)
    
    def __len__(self):
        return self.length
    
    def smart_split(self, text, spans):
        begin_idxs = {s['start'] for s in spans if spans}
        counter = 0
        for i, s in enumerate(text):
            if i in begin_idxs:
                span = spans[counter]
                stop = span['stop']
                #text[i: stop] = text[i: stop].replace(' ', self.union_tag)
                tag = text[i: stop].replace(' ', self.union_tag)
                text = text[:i] + tag + text[stop:]
                counter += 1
                
        tokens = text.split()
        tokens = [t.replace(self.union_tag, ' ') for t in tokens]
        
        return tokens
    
    def find_similar(self, search_token):
        print('Finding similar token...')
        print(search_token)
        return random.choice(list(self.vocab.keys()))
        
        search_token_split = search_token.split()
        if len(search_token_split) == 2:
            token1, token2 = search_token_split
            if token1 in self.vocab:
                return token1
            if token2 in self.vocab:
                return token2
            
        for token in self.vocab:
            
            if token in search_token and len():
                pass
                
    
    def __getitem__(self, idx):
        text = self.data.at[idx, 'text'].lower()
        spans = self.data.at[idx, 'spans']
        if not isinstance(spans, list):
            spans = yaml.safe_load(spans)
            self.data.at[idx, 'spans'] = spans
        
        #tokens = text.split()
        tokens = self.smart_split(text, spans)
        if self.transforms:
            tokens = [self.transforms(t) for t in tokens]
        #pad_count = self.max_seq_len - len(tokens)
        #tokens += [self.pad_tag]*(pad_count if pad_count > 0 else 0)
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        #print(len([self.vocab[token] for token in curr_tokens if token in self.vocab]))
        curr_tokens = [
            #self.vocab[token if token in self.vocab else self.find_similar(token)]
            token
            for token in curr_tokens 
            if token in self.vocab
        ]
        pad_count = self.max_seq_len - len(curr_tokens)
        curr_tokens += [self.pad_tag]*(pad_count if pad_count > 0 else 0)
        tensor = torch.LongTensor([
            self.vocab[token] for token in curr_tokens
        ])
        tags = []
        """
        last_span_idx = 0
        span = spans[last_span_idx]
        for token in curr_tokens:
            if token == text[span.start: span.stop]:
                tags.append(span.type)
                last_span_idx += 1
                if len(spans) >= last_span_idx+1:
                    span = spans[last_span_idx]
            else:
                tags.append('MISC')
        """
        for token in curr_tokens:
            found = False
            for span in spans:
                if token == text[span['start']: span['stop']]:
                    tags.append(span['type'])
                    found = True
                    break
            if not found:
                tags.append('MISC')
            
        cstlen = len([t for t in tags if t != 'MISC'])
        #if cstlen != len(spans):
        #    print(f'Fail: {cstlen} != {len(spans)}')
        tags = torch.LongTensor([
            self.tag2index[tag] for tag, token in zip(tags, curr_tokens) 
            #if token in self.vocab
        ])
        
        return tensor, tags
    

class TextProcessing:
    def __init__(self):
        pass

    def lower(self, text):
        return text.lower()
    
    def clean_text(self, text):
        text = text.replace(',', ', ').replace('.', '. ').replace('(', ' (').replace(')', ') ')
        text = re.sub(r"\d+", "NUMB", text)
        text = re.sub('[,."!;?«»]', '', text)
        text = ' '.join(text.split())
        text = text.replace(u'\u200b', '').replace(u'\u200f', '')
        #text = text.replace('ё', 'е')
        return text
        
    def drop_trash(self, text):
        text = re.sub("@[A-Za-zА-Яа-я0-9_]+","", text)
        text = re.sub("#[A-Za-zА-Яа-я0-9_]+","", text)
        text = re.sub(r'http\S+', '', text)
        
        return text
    
    def drop_english_words(self, text):
        text = ' '.join([w for w in text.split() if not re.match(r'[A-Z]+', w, re.I)])

        return text
    
    def drop_emoji(self, text):
        text = emoji.get_emoji_regexp().sub(r'', text)

        return text
    
    def __call__(self, text):
        if text is None:
            return ''
        text = self.lower(text)
        text = self.clean_text(text)
        #text = self.drop_trash(text)
        #text = self.drop_english_words(text)
        text = self.drop_emoji(text)

        return text