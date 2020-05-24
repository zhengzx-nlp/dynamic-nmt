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
from src.modules.rnn import RNN
from src.utils.logging import INFO, WARN
from . import transformer, dl4mt


class WordPredictor(nn.Module):
    def __init__(self, generator=None, **config):
        super().__init__()
        if generator is not None:
            self.generator = generator
        else:
            self.generator = dl4mt.Generator(n_words=config['n_tgt_vocab'],
                                             hidden_size=config['d_word_vec'], padding_idx=PAD)
        self.linear = nn.Linear(config['d_model'], config['d_word_vec'])

    def forward(self, hiddens, logprob=True):
        logits = F.tanh(self.linear(hiddens))
        return self.generator(logits, logprob)


class RNNLM(nn.Module):
    def __init__(self, generator, **config):
        super().__init__()
        self.generator = generator
        self.gru = RNN(type="gru", batch_first=True,
                       input_size=config['d_model'], hidden_size=config['d_model'])
        self.linear = nn.Linear(config['d_model'], config['d_word_vec'])

    def forward(self, embs, logprob=True):
        one_step = (embs.dim() == 2)

        if one_step:
            embs = embs.unsqueeze(1)

        mask = embs.new_zeros(*embs.size()[:-1])

        outs, _ = self.gru(embs, mask)
        logits = F.tanh(self.linear(outs))

        scores = self.generator(logits, logprob)

        if one_step:
            scores = scores.squeeze(1)
        return scores


class TransEncRNNDec(dl4mt.DL4MT, transformer.Transformer):
    def __init__(self, n_src_vocab, n_tgt_vocab, **config):
        super(transformer.Transformer, self).__init__()

        self.config = config

        self.encoder = transformer.Encoder(
            n_src_vocab=n_src_vocab, n_layers=config['n_layers'], n_head=config['n_head'],
            d_word_vec=config['d_word_vec'], d_model=config['d_model'],
            d_inner_hid=config['d_inner_hid'], dropout=config['dropout'])

        self.decoder = dl4mt.Decoder(cell_type=config.setdefault('decoder_cell_type', 'cgru'),
            n_words=n_tgt_vocab, input_size=config['d_word_vec'],
            hidden_size=config['d_model'], context_size=config['d_model'],
            dropout_rate=config['dropout'],
            bridge_type=config['bridge_type'])

        self.generator = dl4mt.Generator(n_words=n_tgt_vocab,
                                         hidden_size=config['d_word_vec'], padding_idx=PAD)

        if config['proj_share_weight']:
            self.generator.proj.weight = self.decoder.embeddings.embeddings.weight

        if config.setdefault('attention_context_word_prediction', False):
            self.attn_wp = WordPredictor(self.generator, n_tgt_vocab=n_tgt_vocab, **config)

        if config.setdefault('target_language_model', False):
            self.tgt_lm = RNNLM(self.generator, **config)

    @property
    def num_scores(self):
        num = 1
        if self.config['attention_context_word_prediction']:
            num += 1
        if self.config['target_language_model']:
            num += 1
        return num

    def encode(self, src_seq):
        return transformer.Transformer.encode(self, src_seq)

    def init_decoder(self, enc_outputs, expand_size=1):
        return dl4mt.DL4MT.init_decoder(self, enc_outputs, expand_size)

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        return dl4mt.DL4MT.reorder_dec_states(self, dec_states, new_beam_indices, beam_size)

    def load_external_lm(self, pretrain_path=None, device=None):
        if pretrain_path is None:
            return
        else:
            INFO("Loading pretrained parameters for LM from {}".format(pretrain_path))

            need_pretrain_params_prefix = [
                'decoder.cell',
                'decoder.linear_hidden',
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

        # self.decoder.embeddings.embeddings.weight.requires_grad = False
        # self.generator.proj.weight.requires_grad = False

    def forward(self, src_seq, tgt_seq, log_probs=True):
        enc_ctx, enc_mask = self.encoder(src_seq)

        dec_init, dec_cache = self.decoder.init_decoder(enc_ctx, enc_mask)

        dec_states = self.decoder(tgt_seq,
                                  context=enc_ctx,
                                  context_mask=enc_mask,
                                  one_step=False,
                                  hidden=dec_init,
                                  cache=dec_cache)
        logits = dec_states['logits']
        hiddens = dec_states['hiddens']
        attention_contexts = dec_states['attention_contexts']

        final_log_probs = self.generator(logits, log_probs)

        if self.config['attention_context_word_prediction']:
            attn_wp_log_probs = self.attn_wp(attention_contexts, log_probs)
            final_log_probs = final_log_probs + attn_wp_log_probs

        if self.config['target_language_model']:
            tgt_lm_log_probs = self.tgt_lm(dec_states['embeddings'], log_probs)
            final_log_probs = final_log_probs + tgt_lm_log_probs

        return final_log_probs / self.num_scores

    def decode(self, tgt_seq, dec_states, log_probs=True):
        ctx = dec_states['ctx']
        ctx_mask = dec_states['ctx_mask']

        prev_dec_hiddens = dec_states['dec_hiddens']
        dec_caches = dec_states['dec_caches']

        final_word_indices = tgt_seq[:, -1].contiguous()

        dec_states = self.decoder(final_word_indices, hidden=prev_dec_hiddens,
                                  context=ctx, context_mask=ctx_mask,
                                  one_step=True, cache=dec_caches)
        logits = dec_states['logits']
        hiddens = dec_states['hiddens']
        attention_contexts = dec_states['attention_contexts']

        final_log_probs = self.generator(logits, log_probs=log_probs)

        if self.config['attention_context_word_prediction']:
            attn_wp_log_probs = self.attn_wp(attention_contexts, log_probs)
            final_log_probs = final_log_probs + attn_wp_log_probs

        if self.config['target_language_model']:
            tgt_lm_log_probs = self.tgt_lm(dec_states['embeddings'], log_probs)
            final_log_probs = final_log_probs + tgt_lm_log_probs

        final_log_probs = final_log_probs / self.num_scores

        dec_states = {"ctx": ctx, "ctx_mask": ctx_mask, "dec_hiddens": hiddens, "dec_caches": dec_caches}

        return final_log_probs, dec_states

