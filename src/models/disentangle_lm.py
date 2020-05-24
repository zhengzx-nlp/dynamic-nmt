# -*- coding: UTF-8 -*- 

# Copyright 2018, Natural Language Processing Group, Nanjing University, 
#
#       Author: Zheng Zaixiang
#       Contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.modules.cgru import GRUAttnCell
from src.modules.embeddings import Embeddings
from src.utils.logging import INFO, WARN
from . import dl4mt


class DisentangleRNNDecoder(nn.Module):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 dropout_rate=0.0):

        super(DisentangleRNNDecoder, self).__init__()

        self.hidden_size = hidden_size

        self.embeddings = Embeddings(num_embeddings=n_words,
                                     embedding_dim=input_size,
                                     dropout=0.0,
                                     add_position_embedding=False)

        self.cell = GRUAttnCell(input_size=input_size, hidden_size=hidden_size)

        self.linear_hidden = nn.Linear(in_features=hidden_size, out_features=input_size)
        # self.linear_ctx = nn.Linear(in_features=context_size, out_features=input_size)

        self.dropout = nn.Dropout(dropout_rate)

    def init_decoder(self, y):
        batch_size = y.size(0)
        dec_init = y.new_zeros(batch_size, self.hidden_size).float()
        return dec_init

    def forward(self, y, hidden, one_step=False):
        emb = self.embeddings(y)  # [batch_size, seq_len, dim]

        if one_step:
            (out, attn), hidden = self.cell(emb, hidden)
        else:
            # emb: [batch_size, seq_len, dim]
            out = []
            attn = []

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=1):
                (out_t, attn_t), hidden = self.cell(emb_t.squeeze(1), hidden)
                out += [out_t]
                attn += [attn_t]

            out = torch.stack(out).transpose(1, 0).contiguous()
            # attn = torch.stack(attn).transpose(1, 0).contiguous()

        logits = self.linear_hidden(out)
        logits = F.tanh(logits)
        logits = self.dropout(logits)  # [batch_size, seq_len, dim]

        return {'logits': logits, 'hiddens': out, 'embeddings': emb}


class DisentangleRNNLM(nn.Module):
    def __init__(self, n_tgt_vocab, **config):
        super().__init__()
        self.config = config
        self.decoder = DisentangleRNNDecoder(
            n_words=n_tgt_vocab, input_size=config['d_word_vec'],
            hidden_size=config['d_model'],
            dropout_rate=config['dropout'])

        self.generator = dl4mt.Generator(n_words=n_tgt_vocab,
                                         hidden_size=config['d_word_vec'], padding_idx=PAD)

        if config['proj_share_weight']:
            self.generator.proj.weight = self.decoder.embeddings.embeddings.weight

    def init_parameters(self, pretrain_path=None, device=None):
        if pretrain_path is None:
            return
        else:
            INFO("Loading pretrained parameters for LM from {}".format(pretrain_path))

            need_pretrain_params_prefix = [
                'decoder.embeddings',
                'decoder.cell',
                'decoder.linear_hidden',
                'generator'
            ]
            pretrain_params = torch.load(pretrain_path, map_location=device)
            for name, params in pretrain_params.items():
                for pp in need_pretrain_params_prefix:
                    if name.startswith(pp):
                        INFO("Loading param: {}...".format(name))
                        try:
                            self.load_state_dict({name: params}, strict=False)
                        except Exception as e:
                            WARN("{}: {}".format(str(Exception), e))

            INFO("Pretrained model loaded.")

        self.decoder.embeddings.embeddings.weight.requires_grad = False
        self.generator.proj.weight.requires_grad = False

    def forward(self, tgt_seq, log_probs=True):
        dec_init = self.decoder.init_decoder(tgt_seq)

        dec_states = self.decoder(tgt_seq,
                                  hidden=dec_init,
                                  one_step=False)
        logits = dec_states['logits']
        hiddens = dec_states['hiddens']

        final_log_probs = self.generator(logits, log_probs)

        return final_log_probs
