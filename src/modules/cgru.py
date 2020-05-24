import torch.nn as nn

from src.utils import init as my_init
from .attention import BahdanauAttention

class CGRUCell(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 context_size):

        super(CGRUCell, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size

        self.gru1 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.attn = BahdanauAttention(query_size=hidden_size, key_size=self.context_size)
        self.gru2 = nn.GRUCell(input_size=self.context_size, hidden_size=hidden_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in self.gru1.parameters():
            my_init.rnn_init(weight)

        for weight in self.gru2.parameters():
            my_init.rnn_init(weight)

    def forward(self,
                input,
                hidden,
                context,
                context_mask=None,
                cache=None):

        hidden1 = self.gru1(input, hidden)
        attn_values, _ = self.attn(query=hidden1, memory=context, cache=cache, mask=context_mask)
        hidden2 = self.gru2(attn_values, hidden1)

        return (hidden2, attn_values), hidden2

    def compute_cache(self, memory):
        return self.attn.compute_cache(memory)


class GRUAttnCell(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 context_size=None):

        super(GRUAttnCell, self).__init__()

        self.hidden_size = hidden_size
        self.context_size = context_size
        self.compute_attention = self.context_size is not None

        self.gru1 = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        # self.gru2 = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)
        if self.compute_attention:
            self.attn = BahdanauAttention(query_size=hidden_size, key_size=self.context_size)

    def forward(self,
                input,
                hidden,
                context=None,
                context_mask=None,
                cache=None):

        hidden = self.gru1(input, hidden)
        if self.compute_attention:
            attn_values, _ = self.attn(query=hidden, memory=context, cache=cache, mask=context_mask)
        else:
            attn_values = None

        return (hidden, attn_values), hidden

    def compute_cache(self, memory):
        return self.attn.compute_cache(memory)