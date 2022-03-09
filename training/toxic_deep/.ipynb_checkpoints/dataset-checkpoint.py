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

class NERCharDataset(Dataset):
    
    def __init__(self, data_path, phase, transforms=None, test_size=0.2, add_data=None, add_eos_tag=False, add_bos_tag=False):
        data = pd.read_csv(data_path, index_col=0)
        data['origin'] = 'bio'
        #data = data[data.phase == phase]
        """
        data = pd.concat([
            data[(data.phase == 'train') & (data.label == 1)], # (data.index % 2 == 0)
            data[(data.phase == 'train') & (data.label == 0)], # (data.index % 5 == 0)
            data[data.phase == 'test'],
        ], axis=0)
        """
        #"""
        add_data_path = f'{os.sep}'.join(data_path.split(os.sep)[:-2])
        add_data_path = os.path.join(add_data_path, 'toxic_kaggle', 'toxic_kaggle_dataset.csv')
        df_kaggle = pd.read_csv(add_data_path, index_col=0)
        df_kaggle['text_clean'] = df_kaggle.text_clean.apply(lambda x: x[:-1] if x[-1] == '\n' else x)
        df_kaggle['origin'] = 'kaggle'
        data = data.append(df_kaggle[df_kaggle.label == 1])
        
        np.random.seed(8)
        data = data.sample(frac=1).reset_index(drop=True)
        #"""
        
        if phase == 'train':
            data = data.iloc[:int(len(data) * 0.8)]
        else:
            data = data.iloc[int(len(data) * 0.8):]
        
        #self.data = data[~data.text.isna()].reset_index()
        self.data = data.dropna().reset_index()
        self.transforms = transforms
        self.pad_tag = "<pad>"
        self.union_tag = "@"
        self.phase = phase
        self.eos_tag = ["EOS"] if add_eos_tag else []
        self.bos_tag = ["BOS"] if add_bos_tag else []
        self.text_field = 'text_clean' # bio
        self.target_field = 'label'
        self.params_path = '/root/konst/ml-utils/training/toxic_deep/char_{}_params_union_lower.json'
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
                
        class_proportion = Counter(self.data[self.target_field]).most_common(2)
        class_proportion = sorted(class_proportion, key=lambda x: x[0])
        self.class_proportion = [x[1] for x in class_proportion]
        print(f'Vocab size: {len(self.vocab)}')
        self.max_seq_len = 512*8 // 4 // 2

    def is_emoji(self, c):
        return emojis.count(c) > 0
        
    def create_vocab(self):
        distinct_chars = set()
        max_seq_len = -1
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            text = row[self.text_field].lower()
            if self.transforms:
                text = self.transforms(text)
            #tokens = text.split()
            
            if len(text) > max_seq_len:
                max_seq_len = len(text)
            
            [distinct_chars.add(t) for t in text if not self.is_emoji(t)]
        
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
        #return '\\\\U0001f929'
        #return '\\\\U0001f60c' # last
        #return '\\\\U0001f60f'
        #return '\\\\u043f'
        return None
        
    def __getitem__(self, idx):
        text = self.data.at[idx, self.text_field]
        label = self.data.at[idx, self.target_field]
        origin = self.data.at[idx, 'origin']
        
        text = text.replace('\n', ' ').lower()
        if self.transforms:
            text = self.transforms(text)
            
        if self.phase == 'train':            
            if random.random() < 0.4 and label == 1 and origin == 'bio':
                words = text.split()
                new_words = []
                for word in words:
                    if word not in key_toxic_words:
                        new_words.append(word)
                        continue
                    
                    word_idx = random.randint(0, len(key_toxic_words)-1)
                    word = key_toxic_words[word_idx]
                    new_words.append(word)
                
                text = ' '.join(new_words)
                
            if random.random() < 0.4 and label == 1: # and origin == 'bio'
                words = text.split()
                new_words = []
                for word in words:
                    if word not in key_toxic_words or len(word) < 3:
                        new_words.append(word)
                        continue
                    char_idx = random.randint(0, len(word) - 2)
                    char = random.choice(list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'))
                    if random.random() < 0.2:
                        char += random.choice(list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'))
                    #word = word[: char_idx] + char + word[char_idx + 1: ]
                    word = list(word)
                    word.insert(char_idx, char)
                    word = ''.join(word)
                    new_words.append(word)
                    label = 0
                
                text = ' '.join(new_words)
                #print(text)

                    
        #if not text.islower():
        #    print(text)
        tokens = [t for t in text]
        tokens = [self.encode_char(t) for t in tokens]
        tokens = [t for t in tokens if t is not None]
        length = min(len(tokens), self.max_seq_len)
        
        curr_tokens = self.bos_tag + tokens[: self.max_seq_len] + self.eos_tag
        pad_count = self.max_seq_len - len(curr_tokens)
        #print(curr_tokens)
        curr_tokens += [self.pad_tag] * max(pad_count, 0)
        tensor = torch.LongTensor([
            self.vocab[token] 
            for token in curr_tokens
        ])

        label = torch.LongTensor([label])
        
        return tensor, label, idx, length
    
key_toxic_words = [
    'хуй', 
    'хуи',
    'хули ',
    'хуле',
    'хуегер',
    'хуёгер',
    'хуеплет',
    'хуево',
    'хуёво',
    'сука',
    'сучка',
    'сучки',
    'сучары',
    'сучара'
    'блядь', 
    'блять', 
    'хуесос', 
    'пизда', 
    'пизды', 
    'пёзды', 
    'ебу', 
    'ебать',
    'ебашить',
    'ебучий',
    'ебись',
    'мраз', 
    'мразь', 
    'мразота', 
    'мудак', 
    'мудила', 
    'насиловать', 
    'насилую', 
    'изнасилую', 
    'ебан',
    'ебанутый', 
    'ебанутая', 
    'ебать', 
    'выебываться',
    'выебываешься',
    'чмо',
    'жопа', # ?
    #'отбитый',
    'придурок',
    'придурошный',
    'долбаеб',
    'долбоеб',
    'дибил',
    'дебил',
    'дибилка',
    'дебилка',
    'дерьмо',
    'мрраааззь',
    'пошлятина',
    'пошлый',
    'пошлая',
    'задолбалась',
    'задолбался',
    'очко',
    'срать',
    'срал',
    'сру',
    'матерюсь',
    'ненавижу',
    'дрочить',
    'дрочер',
    'пох',
    'похуй',
    'похую',
    'быканул',
    #'плевал',
    'наркоман',
    'психопат',
    'наркоманка',
    'психопатка',
    'сьеби',
    'съеби',
    #'нафиг',
    'торч ',
    'еб*',
    'битч',
    #'наказывать',
    'психушка',
    'дурка',
    #'посмеяться',
    #'убивать',
    #'убью',
    'грязь',
    'ублюдок',
    'ублюдки',
    'порно',
    #'подую',
    'матом',
    'токсичный',
    'токсичная',
    'токсик',
    'даун',
    #'безжалостный',
    'обсирать',
    'высер',
    'накур',
    'накурить',
    'накурил',
    #'клал',
    #'глум', # поглумиться
    'пи*',
    'начхaть',
    'чихать',
    'чихал',
    'чихала',
    'бич',
    'крыса',
    #'кирса',
    'пидр',
    'пидор',
    'пидар',
    'пидоры',
    'пидары',
    'пидорас',
    'пидарас',
    'пидорасы',
    'пидарасы',
    'ебля',
    'ёбля',
    #'припадочный',
    'наюзался',
    'наюзанный',
    #'чсв',
    'нимфоман',
    'нимфоманка',
    'хер',
    'убейся',
    'чиво',
    'соси',
    'х*й',
    #'отвечай',
    'драл',
    'подъёб',
    'подъеб',
    #'грязные',
    #'базар',
    'ебля',
    'ебливый',
    'ебливая',
    'саси',
    'соси',
    #'рот',
    #'хейтер',
    'дура',
    'дурак',
    'трахать',
    'трахну',
    'затрахаю',
    'сдох',
    'сдохну',
    'о4ка',
    'о4ко',
    'ЕБАТЬ',
    'Ебля',
    'ЁБ',
    'еб',
    'уебу',
    'уебать',
    'уебало',
    'уебище',
    'ПИЗДЕЦ',
    'БЛЯДИНА',
    'БЛЯДСКИЙ',
    'БЛЯДСТВО',
    'ВЫЁБЫВАЕТСЯ',
    'ДОЕБАЛСЯ',
    'ЕБАЛО',
    'ЕБАНЁШЬСЯ',
    'ЕБАНУЛ',
    'ЕБАНУЛСЯ',
    'ЕБАШИТ',
    'ебаный',
    'ЖИДОЁБ',
    'ЗАЕБАЛ',
    'ЗАЕБИСЬ',
    'збс',
    'НАБЛЯДОВАЛ',
    'НАЕБАШИЛСЯ',
    'НАЕБНУЛСЯ',
    'ОБЪЕБАЛ',
    'ОДНОХУЙСТВЕННО',
    'ОПИЗДОУМЕЛ',
    'ПЁЗДЫ',
    'ПИЗДА',
    'ПИЗДАБОЛ',
    'ПИЗДАТЫЙ',
    'ПИЗДЕЦ',
    'ПОДЪЁБ',
    'ПОДЪеБал',
    'ПОЕБЕНЬ',
    'РАСПИЗДЯЙ',
    'СПИЗДИЛ',
    'УЕБАЛСЯ',
    'УЁБИЩЕ',
    'ХУЁВО',
    'хуевый',
    'охуел',
    'ХУЙНЯ',
    'ШАРОЁБИТСЯ',
    'подъебать',
]
key_toxic_words = list(set(k.lower() for k in key_toxic_words))
