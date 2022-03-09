from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import torch
from typing import List, Any, Tuple
import random
import emoji
import torch
import re
import os

def tokenize(text, include_space=False):
    #text = text.lower()
    #text = text.replace('\n', ' ') #.replace('-', ' ')
    #text = re.sub("@[A-Za-zА-Яа-я0-9_]+","$MENTION$", text)
    #text = re.sub(r'http\S+', '$URL$', text)
    em_split_emoji = emoji.get_emoji_regexp().split(text)
    em_split_emoji = [re.split(r"([ |,|.|\n|?|!|(|)|\"|«|»|:|;|•|･|゜ﾟ|/|♡|~|'ﾟ|-])", token) for token in em_split_emoji]
    em_split_emoji = [token for tokens in em_split_emoji for token in tokens if token]
    if not include_space:
        em_split_emoji = [t for t in em_split_emoji if t != ' ' and not is_emoji(t)]
        #em_split_emoji = [t for t in em_split_emoji if not is_emoji(t)]
    return em_split_emoji
        
def uppercase(x):
    tokens = tokenize(x, include_space=True)
    out = []
    for i, token in enumerate(tokens):
        if token == token.lower() and i % 5 == 0:
            token = token.upper()
        out.append(token)
        
    out = ''.join(out)
    
    return out

def lowercase(x):
    if x.lower() == x:
        return x
    
    tokens = tokenize(x, include_space=True)
    out = []
    for token in tokens:
        if token == token.upper():
            token = token.lower()
        out.append(token)
        
    out = ''.join(out)
    
    return out

def drop_half_sentences(x):
    #sentences = np.array(x.split('.'))
    sentences = np.array(re.split('[.!?]', x))
    if len(sentences) < 4:
        return x
    
    idxs = list(range(len(sentences)))
    
    return '.'.join(sentences[random.sample(idxs, len(sentences) // 2)])


class augment_context:
    def __init__(self):
        import nlpaug.augmenter.word as naw
        self.special_symbol = '[UNK]'
        self.context_aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-multilingual-uncased', action="insert", top_k=2, aug_min=1, aug_p=0.1)
        
    def __call__(self, x):
        x = self.context_aug.augment(x)
        x = x.replace(self.special_symbol, '')
        return x
    
class augment_w2v:
    def __init__(self):
        import pymorphy2
        from gensim.models import Word2Vec
        from pymystem3 import Mystem

        self.m = Mystem()
        self.morph = pymorphy2.MorphAnalyzer()
        self.w2v = Word2Vec.load('../../../workspace/rureviews.w2v.300d.model').wv
        self.aug_pos = ['INFN', 'VERB', 'GRND', 'ADVB', 'ADJF', 'NOUN'] # NOUN
        
    def lemmatize(self, token):
        token_lem = ''.join(self.m.lemmatize(token)[:-1])
        return token_lem

    def __call__(self, text):
        def parse_sims(sims):
            thresh = 0.7
            tokens = [s[0] for s in sims if s[1] >= thresh]
            scores = [s[1] for s in sims if s[1] >= thresh]

            return tokens, scores

        out = []
        text = text.lower()
        tokens = tokenize(text, include_space=True)
        topn = 10

        for token in tokens:
            pos = self.morph.parse(token)[0].tag.POS
            if pos in self.aug_pos:
                try:
                    sims = self.w2v.most_similar(token, topn=topn)
                    tokens, scores = parse_sims(sims)
                    if len(scores) == 0 or max(scores) <= 0.75 or random.random() <= 1.:
                        token_lem = self.lemmatize(token)
                        sims = self.w2v.most_similar(token_lem, topn=topn)
                        tokens, scores = parse_sims(sims)
                        if len(scores) == 0:
                            raise Exception('len if scores is 0')
                        if max(scores) <= 0.75:
                            raise Exception(f'Not found word: {token_lem}')

                    #scores = [s**2 for s in scores]
                    scores = [s / sum(scores) for s in scores]
                    token = np.random.choice(tokens, p=scores)
                    #print('augmented: ', token)
                except Exception as e:
                    #print(e)
                    pass
            out.append(token)

        return ''.join(out) 

def prob_wrap(func, p):
    return func if random.random() < p else (lambda x: x)


def collate_fn(examples: List[Any]) -> Tuple[torch.Tensor, ...]:
    """Batching examples.

    Parameters
    ----------
    examples : List[Any]
        List of examples

    Returns
    -------
    Tuple[torch.Tensor, ...]
        Tuple of hash tensor, length tensor, and label tensor
    """

    projection = []
    masks = []
    labels = []
    tokens = []
    idxs = []
    
    for example in examples:
        if not isinstance(example, tuple):
            projection.append(np.asarray(example))
        else:
            projection.append(np.asarray(example[0]))
            masks.append(example[1])
            labels.append(example[2])
            tokens.append(example[3])
            idxs.append(example[4])
    #lengths = torch.from_numpy(np.asarray(list(map(len, examples)))).long()
    masks = torch.LongTensor(masks)
    projection_tensor = np.zeros(
        (len(projection), max(map(len, projection)), len(projection[0][0]))
    )
    for i, doc in enumerate(projection):
        projection_tensor[i, : len(doc), :] = doc
    return (
        torch.from_numpy(projection_tensor).float(),
        masks,
        torch.from_numpy(np.asarray(labels)),
        np.array(tokens),
        idxs,
    )

def load_state_dict(model, model_path, device, source=''):
    
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    if source == 'jester':
        pretrained_renamed_dict = {k.replace('module.', ''): v for k, v in pretrained_dict['state_dict'].items()}
        pretrained_dict = pretrained_renamed_dict

    model_dict_new = model_dict.copy()
    counter = 0
    for k, v in pretrained_dict.items():
        if k in model_dict_new and np.all(v.size() == model_dict_new[k].size()):
            model_dict_new.update({k: v})
            counter += 1
    print(f'Loaded: {counter}/{len(model_dict)}')
    model.load_state_dict(model_dict_new)
    return model

def get_split(data_path, csv_path, n_classes=2, test_size=0.1):

    random.seed(0)
    np.random.seed(0)
    
    labels = []
    #for n in range(n_classes):
        #labels += [n] * len(os.listdir(os.path.join(data_path, str(n))))
        #labels += [n] * len(next(os.walk(os.path.join(data_path, str(n))))[1])

    #csv_path = 'non_personal_video_train.csv'
    data = pd.read_csv(csv_path)

    labels = data.label.to_numpy()
    #labels = [0, 1] * 8040
    #labels = np.array(labels)
    #print(next(os.walk(os.path.join(data_path, str(0)))))
    skf = StratifiedKFold(n_splits=int(1 / test_size))
    train_index, test_index = next(skf.split(labels, labels))

    return train_index, test_index