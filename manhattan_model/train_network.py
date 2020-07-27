import random

import torch
import torch.nn.utils.rnn as rnn
from dprint import dprint

class Train_Network(object):
    def __init__(self, model_name, manhattan_model, index2word):
        # self.manhattan_lstm = manhattan_lstm
        self.model_name = model_name
        self.manhattan_model = manhattan_model
        self.index2word = index2word
        self.use_cuda = torch.cuda.is_available()

    def train(self, input_sequences, similarity_scores, criterion, model_optimizer=None, evaluate=False):

        sequences_1 = [sequence[0] for sequence in input_sequences]
        sequences_2 = [sequence[1] for sequence in input_sequences]
        batch_size = len(sequences_1)

        '''
        Pad all tensors in this batch to same length.
        PyTorch pad_sequence method doesn't take pad length, making this step problematic.
        Therefore, lists concatenated, padded to common length, and then split.
        '''

        # if self.model_name == 'cnn':
        # else:
        # trick filter_size를 하나 더해서 min_pad_length를 지정
        temp = rnn.pad_sequence(sequences_1 + sequences_2 + [torch.ones(5)])
        sequences_1 = temp[:, :batch_size]
        sequences_2 = temp[:, batch_size:-1]

        ''' No need to send optimizer in case of evaluation. '''
        if model_optimizer: model_optimizer.zero_grad()
        loss = 0.0

        if self.model_name == 'cnn':
            output_scores = self.manhattan_model([sequences_1, sequences_2]).view(-1)
            # dprint('output_scores = {}'.format(output_scores), color='yellow')
        else:
            hidden = self.manhattan_model.init_hidden(batch_size)
            output_scores = self.manhattan_model([sequences_1, sequences_2], hidden).view(-1)

        loss += criterion(output_scores, similarity_scores)

        if not evaluate:
            loss.backward()
            model_optimizer.step()

        return loss.item(), output_scores
