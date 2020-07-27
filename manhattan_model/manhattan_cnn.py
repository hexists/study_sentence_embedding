import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from dprint import dprint


class CNN_Text(nn.Module):
    def __init__(self, V, D):
        super(CNN_Text, self).__init__()

        Ci = 1
        Co = 100
        Ks = [3, 4, 5]
        C = 50

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        # dprint('after unsqueeze(1) = {}'.format(x.size()), color='red')

        # y = self.convs1[0](x)
        # dprint('conv(x) = {}'.format(y.size()), color='red')

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        # dprint('conv(x).squeeze(3) = {}'.format(x[0].size()), color='red')

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # dprint('max_pool1d(i, i.size(2)).squeeze(2) = {}'.format(x[0].size()), color='red')

        x = torch.cat(x, 1)

        dropout_x = self.dropout(x)  # (N, len(Ks)*Co)
        
        logit = self.fc1(dropout_x)  # (N, C)

        return logit


class Manhattan_CNN(nn.Module):
    def __init__(self, data_name, hidden_size, embedding, use_embedding=False, train_embedding=True):
        super(Manhattan_CNN, self).__init__()
        self.data_name = data_name
        self.use_cuda = torch.cuda.is_available()
        self.hidden_size = hidden_size

        if use_embedding:
            self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
            self.embedding.weight = nn.Parameter(embedding)
            self.input_size = embedding.shape[1] # V - Size of embedding vector

        else:
            self.embedding = nn.Embedding(embedding[0], embedding[1])
            self.input_size = embedding[1]

        self.embedding.weight.requires_grad = train_embedding

        self.cnn_1 = CNN_Text(self.input_size, self.hidden_size)
        self.cnn_2 = CNN_Text(self.input_size, self.hidden_size)

    def exponent_neg_manhattan_distance(self, x1, x2):
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))

    def forward(self, input):
        '''
        input           -> (2 x Max. Sequence Length (per batch) x Batch Size)
        '''
        # dprint('input[0] = {}'.format(input[0].size()), color='red')
        # dprint('input[1] = {}'.format(input[1].size()), color='red')

        embedded_1 = self.embedding(input[0]) # L, B, V
        embedded_2 = self.embedding(input[1]) # L, B, V
        embedded_1 = embedded_1.permute(1, 0, 2)
        embedded_2 = embedded_2.permute(1, 0, 2)

        # dprint('embedded_1 = {}'.format(embedded_1.size()), color='red')
        # dprint('embedded_2 = {}'.format(embedded_2.size()), color='red')

        encoded_1 = self.cnn_1(embedded_1)
        encoded_2 = self.cnn_2(embedded_2)

        similarity_scores = self.exponent_neg_manhattan_distance(encoded_1, encoded_2)

        if self.data_name == 'sick': return similarity_scores*5.0
        else: return similarity_scores

    def init_weights(self):
        ''' Initialize weights of cnn 1 '''
        for name_1, param_1 in self.cnn_1.named_parameters():
            if 'bias' in name_1:
                nn.init.constant_(param_1, 0.0)
            elif 'weight' in name_1:
                nn.init.xavier_normal_(param_1)

        ''' Set weights of cnn 2 identical to cnn 1 '''
        cnn_1 = self.cnn_1.state_dict()
        cnn_2 = self.cnn_2.state_dict()

        for name_1, param_1 in cnn_1.items():
            # Backwards compatibility for serialized parameters.
            if isinstance(param_1, torch.nn.Parameter):
                param_1 = param_1.data

            cnn_2[name_1].copy_(param_1)
