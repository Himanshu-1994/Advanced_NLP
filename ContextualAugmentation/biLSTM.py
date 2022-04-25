import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
