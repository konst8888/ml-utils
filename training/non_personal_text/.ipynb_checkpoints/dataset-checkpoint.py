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
import emoji
import re
import os
import torch
from tqdm import tqdm
import json
import yaml
import random
import pandas as pd
import youtokentome as yttm
tqdm.pandas()

class YttmDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, force_vocab=False, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, lineterminator='\n', index_col=0)
        
        if phase == 'train':
            data = data[data.phase == 'train']
        else:
            data = data[data.phase == 'test']
        
        #data = self.data
        self.data = data[~data.text.isna()].reset_index()
        self.transforms = transforms
        self.phase = phase
        self.pad_tag = '<PAD>'
        self.text_field = 'text'
        self.target_field = 'non_personal'
        self.length = len(self.data)
        random.seed(8)
        
        self.create_vocab(force_vocab)
        
        class_proportion = Counter(self.data[self.target_field]).most_common(2)
        class_proportion = sorted(class_proportion, key=lambda x: x[0])
        self.class_proportion = [x[1] for x in class_proportion]
        print(f'Vocab size: {self.bpe.vocab_size()}')
        self.max_seq_len = 1024

    def is_emoji(self, c):
        return emojis.count(c) > 0
        
    def create_vocab(self, force_vocab):
        train_data_path = "train_data.txt"
        model_path = "yttm_vocab_non_personal_no_emoji"

        if (self.phase == 'test') or (not force_vocab and os.path.exists(model_path)):
            self.bpe = yttm.BPE(model=model_path, n_threads=-1)
            return
        
        print('Training yttm...')
        with open(train_data_path, "w") as fout:
            for idx, row in self.data.iterrows():
                text = row[self.text_field]
                text = ''.join([t for t in text.lower() if is_ok_char(t)])
                if text:
                    print(text, file=fout)
        print('Success')

        yttm.BPE.train(data=train_data_path, vocab_size=2000, model=model_path)
        self.bpe = yttm.BPE(model=model_path, n_threads=-1)
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, idx):
        text = self.data.at[idx, self.text_field]
        label = self.data.at[idx, self.target_field]
        
        text = text.lower()
        text = text.replace('\n', ' ').lower()
        if self.transforms:
            text = self.transforms(text)
                    
        text = ''.join([t for t in text if is_ok_char(t)])
        
        #if not text.islower():
        #    print(text)
        tokens = self.bpe.encode(text)
        length = min(len(tokens), self.max_seq_len)
        
        curr_tokens = tokens[: self.max_seq_len]
        pad_count = self.max_seq_len - len(curr_tokens)
        #print(curr_tokens)
        curr_tokens += [self.bpe.subword_to_id(self.pad_tag)] * max(pad_count, 0)
        tensor = torch.LongTensor(curr_tokens)
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
