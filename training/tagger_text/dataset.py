from torch.utils.data import Dataset, DataLoader
from nerus import load_nerus
import emoji
import emojis
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

class TagCharDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data = data[data.phase == phase]
        
        """
        if phase == 'train':
            data = data.iloc[:int(len(data) * 0.8)]
        else:
            data = data.iloc[int(len(data) * 0.8):]
        """
        
        self.data = data.dropna().reset_index(drop=True)
        self.transforms = transforms
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.text_field = 'caption'
        self.target_field = 'tag'
        self.params_path = '/root/konst/ml-utils/training/tagger_text/char_{}_params_no_emoji_lower.json'
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
                
        print('Vocab size: ', len(self.vocab))
        #class_proportion = Counter(self.data[self.target_field]).most_common(2)
        #class_proportion = sorted(class_proportion, key=lambda x: x[0])
        #self.class_proportion = [x[1] for x in class_proportion]
        
        self.max_seq_len = 1024
        
        self.tags = ['acting','arts','author','automotive','beauty','business','children','cinema','comedy','comicbooks','diy_and_crafts','education','entrepreneur','fashion','fitness','food','gaming','home_and_gardening','lifestyle','literature','marketing','motivational','music','performance_arts','pets','photography','running','science','spirituality','sports','tattoos','technology','travel','vegan','video_games','wedding','yoga']
        
        all_tags = self.data.tag.to_list()
        self.class_proportion = [all_tags.count(tag) for tag in self.tags]

    def is_emoji(self, c):
        return emojis.count(c) > 0
        
    def create_vocab(self):
        distinct_chars = set()
        max_seq_len = -1
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            text = row[self.text_field].lower()
            #if self.transforms:
            #    text = self.transforms(text)
            #tokens = text.split()
            
            if len(text) > max_seq_len:
                max_seq_len = len(text)
            
            [
                distinct_chars.add(t) 
                for t in text
                #if is_ok_char(t)
            ]
        
        self.length = len(self.data)
        self.max_seq_len = min(max_seq_len, 1024)
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
        #return '\\\\U0001f929'
        #return '\\\\U0001f60c' # last
        #return '\\\\U0001f60f'
        #return '\\\\u043f'
        return None
        
    def __getitem__(self, idx):
        text = self.data.at[idx, self.text_field]
        tag = self.data.at[idx, self.target_field]
        
        #print(text + '\n')
        text = text.replace('\n', ' ').lower()
        text = re.sub(r'\\U[A-Za-zА-Яа-я0-9_]+', '', text)
        text = text.replace('\\n', '\n').replace('-', '')
        text = ''.join([t for t in text if is_ok_char(t)])
        
        if tag == 'comedy':
            if random.random() < 0.4:
                #text = text.replace(' а ', ' ')
                comedy_tokens = text.split()
                comedy_idx = random.randint(0, len(comedy_tokens)-1)
                comedy_word = random.choice(['ахахах', ':)', ';)', ')', '))', ')))', ':3', 'шутка', 'комедия', 'смех', 'смешно', 'фейл', 'комик', 'угар', 'угараю', 'лол', 'кек'])
                comedy_tokens.insert(comedy_idx, comedy_word)
                text = ' '.join(comedy_tokens)
                #print(text)
                #time.sleep(5)
        if self.transforms:
            text = self.transforms(text)
                
        #print(text)
        tokens = [t for t in text]
        tokens = [self.encode_char(t) for t in tokens]
        #print([c for t, c in zip(tokens, text) if t is None])
        tokens = [t for t in tokens if t]
        length = min(len(tokens), self.max_seq_len)
        if length == 0:
            #tokens += [self.encode_char(random.choice(ok_chars).lower())]
            length += 1
        
        #print('='*100)
        #time.sleep(10)
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        #print(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = torch.LongTensor([
            self.vocab[token] 
            for token in curr_tokens
        ])
        
        #label = [int(tag in tags) for tag in self.tags]
        label = self.tags.index(tag)
        label = torch.LongTensor([label])
        
        return tensor, label, idx, length
    
def is_ok_char(c):
    return c in ok_chars
    
ok_chars = [
 '\\n',
 'R',
 'T',
 'o',
 'A',
 'U',
 "'",
 'G',
 '|',
 '`',
 '2',
 '.',
 'I',
 '&',
 'h',
 '(',
 's',
 'N',
 'e',
 'S',
 '/',
 'M',
 ')',
 'x',
 ':',
 't',
 '<',
 'Q',
 'f',
 'm',
 'B',
 'O',
 '6',
 'C',
 'k',
 'w',
 '_',
 'i',
 'E',
 'p',
 '%',
 'L',
 'J',
 '}',
 '"',
 '5',
 'u',
 'b',
 'j',
 'K',
 '~',
 'v',
 ';',
 'Y',
 '9',
 'D',
 'Z',
 'q',
 'a',
 '8',
 '#',
 'V',
 'z',
 ']',
 '$',
 '=',
 '*',
 '^',
 '0',
 'd',
 '4',
 'y',
 'P',
 '3',
 ' ',
 'W',
 'c',
 'r',
 '?',
 '!',
 'H',
 '[',
 '+',
 '7',
 '>',
 'X',
 '1',
 '{',
 '@',
 'g',
 ',',
 '-',
 'n',
 'F',
 'l',
 ' ',
]
cyrill = ('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
ok_chars = ''.join(ok_chars) + cyrill
