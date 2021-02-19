from tqdm import tqdm
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os

from models import (
    EmbedHandler,
    EncoderTASA,
    DecoderTASA
)
from dataset import (
    Actions,
    process_data
)

SOS_token = 0
EOS_token = 1

def train(cfg):

    def indexesFromSequence(sequence):
        return [(actions_data.action2index[action], tau) for action, tau in sequence]


    def tensorFromSequence(sequence):
        results = indexesFromSequence(sequence)
        results.append((EOS_token, 1))
        indexes = [i[0] for i in results]
        taus = [i[1] for i in results]
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1), taus


    def tensorsFromPair(action):
        input_tensor, tau = tensorFromSequence(action)
        #target_tensor = tensorFromSentence(action)
        target_tensor = input_tensor[:]
        return (input_tensor, target_tensor, tau)


    def adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer):
        if (sample_counter + 1) % adjust_lr_every >= 0 \
        and (sample_counter + 1) % adjust_lr_every < batch_size:  # 500
            for param in optimizer.param_groups:
                param['lr'] = max(param['lr'] / 1.2, 1e-4)

    def train_one_sequence(input_tensor, target_tensor, taus, encoder, decoder, 
        optimizer, criterion, 
        max_length, teacher_forcing_ratio, device):
        encoder_hidden = encoder.initHidden(device)

        #encoder_optimizer.zero_grad()
        #decoder_optimizer.zero_grad()
        #eh_optimizer.zero_grad()
        optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0
        accuracy = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden, taus[ei], eh)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                tau = taus[di-1] if di > 0 else 0
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, tau, eh)
                loss += criterion(decoder_output, target_tensor[di])
                accuracy += decoder_output[0].argmax().numpy() == target_tensor[di][0].numpy()
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                tau = taus[di-1] if di > 0 else 0
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, tau, eh)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                accuracy += decoder_output[0].argmax().numpy() == target_tensor[di][0].numpy()
                decoder_input = target_tensor[di]  # Teacher forcing

                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        #encoder_optimizer.step()
        #decoder_optimizer.step()
        #eh_optimizer.step()
        optimizer.step()
        
        #if pd.isna(loss.item() / target_length):
        #    print(loss.item(), target_length)
        #    print(input_tensor, target_tensor, taus)

        return loss.item() / target_length, accuracy / target_length


    def trainIters(encoder, decoder, device, cfg):
        teacher_forcing_ratio = cfg.teacher_forcing_ratio
        checkpoint_path = cfg.checkpoint_path
        train_size = cfg.train_size
        epochs = cfg.epochs
        adjust_lr_every = cfg.adjust_lr_every
        save_at = cfg.save_at
        learning_rate = cfg.lr

        #eh_optimizer = optim.SGD(eh.parameters(), lr=learning_rate)
        #encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        #decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        params = list(eh.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.SGD(params, lr=learning_rate)
        criterion = nn.NLLLoss()

        random.seed(0)
        training_pairs = [tensorsFromPair(seq) for _, seq in sequences.items()]
        random.shuffle(training_pairs)
        train_count = int(len(training_pairs) * train_size)
        dataset_train = training_pairs[:train_count]
        dataset_test = training_pairs[train_count:]
        batch_size = 1
        sample_counter = 0
        if adjust_lr_every is None:
            adjust_lr_every = np.inf
        else:
            if adjust_lr_every <= 10:
                adjust_lr_every = adjust_lr_every * data_len * batch_size
            adjust_lr_every = int(adjust_lr_every)

        data_len = len(dataset_train)
        saving_points = [int(data_len * x * save_at) -
                       1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]
        print(saving_points)

        for epoch in range(epochs):
            for phase in ['train', 'valid']:
                if phase == 'train':
                    eh.train()
                    encoder.train()
                    decoder.train()
                    dataset = dataset_train
                else:
                    eh.eval()
                    encoder.eval()
                    decoder.eval()
                    dataset = dataset_test

                data_len = len(dataset)
                saving_points = [int(data_len * x * save_at) -
                               1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]

                running_loss = 0
                running_acc = 0

                pbar = tqdm(range(len(dataset)))

                for idx in pbar:
                    training_pair = training_pairs[idx]
                    input_tensor = training_pair[0]
                    target_tensor = training_pair[1]
                    taus = training_pair[2]
                    sample_counter += batch_size
                    adjust_lr(sample_counter, adjust_lr_every, batch_size, encoder)
                    adjust_lr(sample_counter, adjust_lr_every, batch_size, decoder)
                    adjust_lr(sample_counter, adjust_lr_every, batch_size, eh)

                    loss, acc = train_one_sequence(input_tensor, target_tensor, taus, encoder,
                                 decoder, optimizer, 
                                 criterion, MAX_LENGTH, teacher_forcing_ratio, device)

                    if idx % 5000 == 0:
                        print(loss)

                    running_loss += loss
                    running_acc += acc

                    scale_value = 1 / max(idx, 1)
                    pbar.set_description(
                        "Epoch: {}/{}, Phase: {}, Loss: {:.4f} ({:.4f}), Acc: {:.4f} ({:.4f})".format(
                            epoch,
                            epochs,
                            phase,
                            running_loss * scale_value,
                            loss.item(),
                            running_acc * scale_value,
                            acc
                        )
                    )
                    if phase == 'train':
                        last_loss = running_loss * scale_value
                        last_acc = running_acc * scale_value

                    if checkpoint_path is not None and idx in saving_points:
                        models = {
                            'encoder': encoder,
                            'decoder': decoder,
                            'eh': eh
                        }
                        for model_name in models:
                            if phase == 'train' and len(saving_points) > 1:
                                name = '{}_epoch_{}_loss_{:.4f}_acc_{:.4f}.pth'.format(
                                        model_name,
                                        epoch,
                                        running_loss * scale_value,
                                        running_acc * scale_value
                                    )
                            elif phase == 'valid' and len(saving_points) == 1:
                                name = '{}_epoch_{}_train_loss_{:.4f}_test_loss_{:.4f}_train_acc_{:.4f}_test_acc_{:.4f}.pth'.format(
                                        model_name,
                                        epoch,
                                        last_loss,
                                        running_loss * scale_value,
                                        last_acc,
                                        running_acc * scale_value
                                    )
                            else:
                                continue
                            torch.save(models[model_name], os.path.join(checkpoint_path, name))

                
        return encoder, decoder

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    hidden_size = cfg.hidden_size

    sequences = process_data(cfg.csv_path, cfg.max_length)
    actions_data = Actions()
    actions_data.addData([[s[0] for s in seq] for _, seq in sequences.items()])
    MAX_LENGTH = max(len([s[0] for s in seq]) for _, seq in sequences.items()) + 2
    print('Largest class: ', max(actions_data.action2count.items(), key=lambda x: x[1])[1] \
        / sum(x[1] for x in actions_data.action2count.items()))

    eh = EmbedHandler(actions_data.n_actions).to(device)
    encoder = EncoderTASA(actions_data.n_actions, hidden_size).to(device)
    decoder = DecoderTASA(hidden_size, actions_data.n_actions).to(device)

    if cfg.encoder_path and cfg.decoder_path:
        encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder, decoder = trainIters(encoder, decoder, device, cfg)
