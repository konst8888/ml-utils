import pandas as pd
from tqdm import tqdm
import datetime as dt

class Actions:
    def __init__(self):
        self.action2index = {}
        self.action2count = {}
        self.index2action = {0: "SOS", 1: "EOS"}
        self.n_actions = 2  # Count SOS and EOS

    def addData(self, sequences):
        print('Adding data...')
        for seq in tqdm(sequences):
            self.addSequence(seq)
        print('Success')

    def addSequence(self, sentence):
        for action in sentence:
            self.addAction(action)

    def addAction(self, action):
        if action not in self.action2index:
            self.action2index[action] = self.n_actions
            self.action2count[action] = 1
            self.index2action[self.n_actions] = action
            self.n_actions += 1
        else:
            self.action2count[action] += 1

def get_sequences(df, max_length):
    df_dict = df.to_dict()

    user_idxs = {}
    for k, v in df_dict['user_id'].items():
        user_idxs[v] = user_idxs.get(v, []) + [k]

    sequences = dict()

    for user_id, idxs in tqdm(user_idxs.items()):
        actions = [df_dict['action'][idx] for idx in idxs]
        if max_length:
            actions = actions[-max_length:]
        taus = [df_dict['event_time'][idx] for idx in idxs]
        taus = [dt.datetime.timestamp(dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')) for t in taus]
        taus = [t / max(taus) for t in taus]
        sequences[user_id] = list(zip(actions, taus))
        
    return sequences


def process_data(csv_path, max_length):
    print('Processing data...')
    df = pd.read_csv(csv_path, index_col=0)
    #df = df.iloc[:1000]
    print(df.shape)

    df = df[['user_id', 'event_time', 'action']]

    idxs_drop = []
    user_list = []
    for idx, row in tqdm(df[df.action == 'referral_made'].iterrows()):
        user_id = row.user_id
        if user_id not in user_list:
            user_list.append(user_id)
        else:
            idxs_drop.append(idx)

    df = df[~df.index.isin(idxs_drop)]
    del idxs_drop
    df.reset_index(drop=True, inplace=True)

    df.sort_values(by=['user_id', 'event_time'], inplace=True);
    df.reset_index(drop=True, inplace=True)

    sequences = get_sequences(df, max_length)
    sequences = {k: v for k, v in sequences.items() if len(v) > 1}
    print('Success')
    print('Actions count: ', len([s[0] for _, seq in sequences.items() for s in seq]))

    return sequences