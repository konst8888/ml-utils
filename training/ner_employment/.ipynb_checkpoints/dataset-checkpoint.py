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
from collections import Counter
from pymystem3 import Mystem
import pandas as pd
import numpy as np
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

class NERCharDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        """
        data1 = pd.read_csv('/root/konst/data/bio_dataset/bio_ner_dataset.csv', index_col=0)
        data1.columns = ['text', 'spans']
        for col in ['text_processed_lem', 'is_processed', 'text_processed']:
            data1[col] = np.nan
        #print(data1.columns)
        data = data.append(data1)
        """
        data_len = int(data.shape[0] * (1-test_size))
        if phase == 'train':
            data = data.iloc[:data_len]
            #data = data.iloc[:int(data_len*0.2)] # 3433/4621 -- error
        if phase == 'test':
            data = data.iloc[data_len: ]
            #data = data.iloc[int(data.shape[0] * (1-0.2/8)):]
                
        #self.data = data[~data.text.isna()].reset_index()
        self.data = data.dropna().reset_index()
        self.transforms = transforms
        self.tag2index = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-LOC': 3,
            'I-LOC': 4,
            'B-ORG': 5,
            'I-ORG': 6,
        }
        self.tag2index = {
            'O': 0,
            'B-PERSON': 1,
            'I-PERSON': 2,
            'B-GPE': 3,
            'I-GPE': 4,
            'B-ORG': 5,
            'I-ORG': 6,
            'B-EMPLOYMENT': 7,
            'I-EMPLOYMENT': 8,
        }
        self.tag2index = {
            'O': 0,
            'PER': 1,
            'LOC': 2,
            'ORG': 3,
        }
        self.tag2index1 = {
            'O': 0,
            'PERSON': 1,
            'GPE': 2,
            'ORG': 3,
            'EMPLOYMENT': 4,
        }
        self.tag2index = {
            'O': 0,
            'PERSON': 1,
            'GPE': 2,
            'ORG': 3,
            'EMPLOYMENT': 4,
            'AGE': 5,
            'PERSON_FAM': 6,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.text_field = 'bio' # bio
        self.spans_field = 'ner_preds' # ner_preds
        self.params_path = '/root/konst/ml-utils/training/ner_employment/char_{}_params_union.json'
        random.seed(8)
                
        if add_data is not None:
            self.length = len(self.data)
            #self.max_seq_len = min(add_data['max_seq_len'], 512*8)
            self.vocab = add_data['vocab']
        elif os.path.exists(self.params_path.format(self.phase)):
            with open(self.params_path.format(self.phase), 'r', encoding='unicode-escape') as f:
                data = yaml.safe_load(f.read())
            self.length = len(self.data)
            if phase == 'test':
                with open(self.params_path.format('train'), 'r', encoding='unicode-escape') as f:
                    data = yaml.safe_load(f.read())
            
            #self.max_seq_len = min(data['max_seq_len'], 512*8)
            self.vocab = data['vocab']
        else:
            #data['spans'] = data.spans.progress_apply(yaml.safe_load)
            print(f'Start vocab building {phase}...')
            self.create_vocab()
            print('Finish')
            
        #"""
        class_proportion = [(self.tag2index[span['type'][2:]], span['stop'] - span['start']) for spans in self.data[self.spans_field] for span in yaml.safe_load(spans) if span['type'] and span['type'][2:] in self.tag2index]
        class_proportion = pd.DataFrame(class_proportion, columns=['type', 'length'])
        class_proportion = class_proportion.groupby('type').agg({'length': 'sum'}).reset_index()
        class_proportion = class_proportion.sort_values('type')
        self.class_proportion = [x for x in class_proportion.length]
        #"""
        #self.class_proportion = [1 for _ in range(len(self.tag2index) - 1)]
        self.max_seq_len = 512*8 // 4

        
    def create_vocab(self):
        distinct_chars = set()
        max_seq_len = -1
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            text = row[self.text_field] #.lower()
            if self.transforms:
                text = self.transforms(text)
            #tokens = text.split()
            
            if len(text) > max_seq_len:
                max_seq_len = len(text)
            
            [distinct_chars.add(t) for t in text]
        
        self.length = len(self.data)
        self.max_seq_len = min(max_seq_len, 512*8) # // 4
        distinct_chars = [self.pad_tag] + list(distinct_chars)
        #distinct_chars.add('$\n$')
        #distinct_chars.remove('\n')
        self.vocab = {w.encode('unicode-escape').decode('ASCII'): idx for idx, w in enumerate(set(distinct_chars))}
        
        data = {
            'length': self.length,
            'max_seq_len': max_seq_len,
            'vocab': self.vocab
        }
        with open(self.params_path.format(self.phase).split('/')[-1], 'w', encoding='unicode-escape') as f:
            f.write(str(data))
        
    def __len__(self):
        return len(self.data)
    
    def encode_char(self, char):
        char = char.encode('unicode-escape').decode('ASCII')
        if char in self.vocab:
            return char
        prefix = '\\'
        char = prefix + char
        if char in self.vocab:
            return char
        char = prefix + char
        if char in self.vocab:
            return char
        
        #print(char[4:])
        #print(self.vocab)
        return '\\\\U0001f929'
        #return '\\U0001f929'
        
    def remove_parent_cat(self, spans, cat, subcat):
        spans_new = spans[:]
        for i, span in enumerate(spans):
            for span1 in spans:
                if span['type'][2:] == cat:
                    if span1['type'][2:] == subcat:
                        if span['start'] == span1['start']:
                            spans_new.pop(i)
                            
        return spans_new
    
    def __getitem__(self, idx):
        text = self.data.at[idx, self.text_field]
        spans = self.data.at[idx, self.spans_field]
        if not isinstance(spans, list):
            spans = yaml.safe_load(spans)
            spans = [span for span in spans if span['type'][2:] in self.tag2index]
            spans = self.remove_parent_cat(spans, 'PERSON', 'PERSON_FAM')
            self.data.at[idx, self.spans_field] = spans
        
        text = text.replace('\n', ' ') #.lower()
        if self.transforms:
            text_trans = self.transforms(text)
        
        tokens = [t for t in text]
        tokens = [self.encode_char(t) for t in tokens]
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        #print(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = torch.LongTensor([
            self.vocab[token] 
            for token in curr_tokens
        ])
        #spans_extend = []
        spans_extend = spans[:]

        """
        for span in spans:
            text_part = text[span['start']: span['stop']]
            #tokens = self.tokenize(text_part)
            #tokens = [t for t in tokens if t != ' ']
            tokens = text_part.split()
            first_token = tokens[0]
            spans_extend.append({
                'start': span['start'],
                'stop': span['start'] + len(first_token),
                'type': span['type'], # 'B-' + 
            })
            if len(tokens) > 1:
                spans_extend.append({
                    'start': span['start'] + len(first_token) + 1,
                    'stop': span['stop'],
                    'type': span['type'], # 'I-' + 
                })
        """
        #spans_extend.sort(key=lambda x: x['start'])
        
        if len(spans_extend) == 0:
            tags = torch.LongTensor([self.tag2index['O']]*len(tensor))
            
            mask = [int(token != self.pad_tag) for token in curr_tokens]
            mask = torch.LongTensor(mask)

            return tensor, mask, tags, curr_tokens, idx

        tags = []
        for i, token in enumerate(curr_tokens):
            idxs = [j for j, span in enumerate(spans_extend) if span['start'] <= i < span['stop']]
            if len(idxs) > 0:
                span = spans_extend[idxs[0]]
                tags.append(
                    span['type'][2:] 
                    if span['type'][2:] in self.tag2index
                    else 'O'
                )
            else:
                tags.append('O')
        
        tags = torch.LongTensor([
            self.tag2index[tag] for tag in tags 
        ])
        mask = torch.LongTensor([int(token != self.pad_tag) for token in curr_tokens])
        #print([t for t, m in zip(curr_tokens, tags) if m != 0])
        #print(curr_tokens[: 150])
        #print(tags[: 150])
        #print([text[s['start']: s['stop']] for s in spans_extend])
        #import sys
        #sys.exit()
        
        return tensor, mask, tags, curr_tokens, idx


class NERProjDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data_len = int(data.shape[0] * (1-test_size))
        if phase == 'train':
            data = data.iloc[:data_len]
            #data = data.iloc[:int(data_len*0.85)] # 3433/4621 -- error
        if phase == 'test':
            data = data.iloc[data_len: ]
            #data = data.iloc[int(data.shape[0] * (1-0.2/8)):]
                
        self.data = data.dropna().reset_index()
        self.transforms = transforms
        self.tag2index = {
            'O': 0,
            'B-PER': 1,
            'I-PER': 2,
            'B-LOC': 3,
            'I-LOC': 4,
            'B-ORG': 5,
            'I-ORG': 6,
        }
        self.tag2index = {
            'O': 0,
            'B-PERSON': 1,
            'I-PERSON': 2,
            'B-GPE': 3,
            'I-GPE': 4,
            'B-ORG': 5,
            'I-ORG': 6,
            'B-EMPLOYMENT': 7,
            'I-EMPLOYMENT': 8,
        }
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.max_seq_len = 128 # 512
        self.hash_func = SimhashToken(128)
        self.hash_table = {
            self.pad_tag: self.hash_func(self.pad_tag)
        }
        self.m = Mystem()
        self.lem_table = {
            self.pad_tag: self.pad_tag
        }
        random.seed(8)
        
        class_proportion = Counter([self.tag2index[span['type']] for spans in self.data.ner_preds for span in yaml.safe_load(spans) if span['type']])
        class_proportion = dict(class_proportion).items()
        class_proportion = sorted(class_proportion, key=lambda x: x[0])
        self.class_proportion = [x[1] for x in class_proportion]
                
    def __len__(self):
        return len(self.data)
        
    @classmethod
    def tokenize(self, text):
        text = text.lower()
        text = text.replace('-', ' ')
        #text = re.sub(r"\d+", "NUMB", text)
        text = re.sub("@[A-Za-zА-Яа-я0-9_]+","$MENTION$", text)
        text = re.sub(r'http\S+', '$URL$', text)
        em_split_emoji = emoji.get_emoji_regexp().split(text)
        em_split_emoji = [re.split(r"([ |,|.|\n|?|!|(|)|\"|«|»|:|;|•|･|゜ﾟ|/|♡|~|'ﾟ])", token) for token in em_split_emoji]
        em_split_emoji = [token for tokens in em_split_emoji for token in tokens if token]
        em_split_emoji = [t for t in em_split_emoji if t != ' ']
        return em_split_emoji
    
    def get_hash(self, token, lem=False):
        if lem:
            token = self.lemmatize(token)
        
        if token in self.hash_table:
            return self.hash_table[token]
        
        token_hash = self.hash_func(token)
        self.hash_table[token] = token_hash
        return token_hash
    
    def lemmatize(self, token):
        if token in self.lem_table:
            token_lem = self.lem_table[token]
        else:
            token_lem = ''.join(self.m.lemmatize(token)[:-1])
            self.lem_table[token] = token_lem
        
        return token_lem
    
    def __getitem__(self, idx):
        text = self.data.at[idx, 'bio'] # text
        spans = self.data.at[idx, 'ner_preds'] # spans
        if not isinstance(spans, list):
            if pd.isna(spans):
                spans = '[]'
            spans = yaml.safe_load(spans)
            self.data.at[idx, 'ner_preds'] = spans # spans
        
        #text_trans = text
        if self.transforms:
            text_trans = self.transforms(text)
        
        tokens = self.tokenize(text)
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = [
            self.get_hash(token, lem=True) for token in curr_tokens # murmurhash
        ]
        """
        spans_extend = []
        for span in spans:
            text_part = text[span['start']: span['stop']].replace('-', ' ')
            #tokens = self.tokenize(text_part)
            #tokens = [t for t in tokens if t != ' ']
            tokens = text_part.split()
            if len(tokens) == 1:
                spans_extend.append({
                        'start': span['start'],
                        'stop': span['stop'],
                        'type': 'B-' + span['type'],
                })
            else:
                current_len = 0
                for i, token in enumerate(tokens):
                    tag =  ('B-' if i == 0 else 'I-') + span['type']
                    spans_extend.append({
                        'start': span['start'] + current_len,
                        'stop': span['start'] + current_len + len(token),
                        'type': tag,
                    })
                    current_len += len(token) + 1
        """
        spans_extend = spans[:]
        if len(spans_extend) == 0:
            tags = [self.tag2index['O']]*len(tensor)
            
            mask = [int(token != self.pad_tag) for token in curr_tokens]
            #mask = torch.LongTensor(mask)

            return tensor, mask, tags, curr_tokens, idx

        tags = []
        last_span_idx = 0
        span = spans_extend[last_span_idx]
        for token in curr_tokens:
            #print(text[span['start']: span['stop']].replace(')', '').replace('(', '').lower())
            if token == text[span['start']: span['stop']].replace(')', '').replace('(', '').lower() \
                    and span['type'] in self.tag2index:
                tags.append(span['type'])
                last_span_idx += 1
                if len(spans_extend) >= last_span_idx+1:
                    span = spans_extend[last_span_idx]                    
            else:
                tags.append('O')
        
        tags = [
            self.tag2index[tag] for tag in tags 
        ]
        
        """
        skip_next = False
        for i, tag in enumerate(tags):
            if skip_next:
                skip_next = False
                continue
            if tag == 0:
                if random.random() < 0.2:
                    tags[i] = self.tag2index['B-EMPLOYMENT']
                    if i < len(tags)-1 and tags[i+1] == 0:
                        tags[i+1] = self.tag2index['I-EMPLOYMENT']
                        skip_next = True
        """
        
        mask = [int(token != self.pad_tag) for token in curr_tokens]
        #print([(t, m) for t, m in zip(curr_tokens, tags) if m != 0])
        #print(curr_tokens[: 150])
        #print(tags[: 150])
        #import sys
        #sys.exit()
        return tensor, mask, tags, curr_tokens, idx


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