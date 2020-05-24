# -*- coding: UTF-8 -*- 
# 
# MIT License
#
# Copyright (c) 2018 the xnmt authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#       author: Zaixiang Zheng
#       contact: zhengzx@nlp.nju.edu.cn 
#           or zhengzx.142857@gmail.com
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.vocabulary import PAD
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.models.base import NMTModel
from src.modules.basic import BottleLinear as Linear
from src.modules.capsule import ContextualCapsuleLayer, EMContextualCapsuleLayer
from src.modules.criterions import MultiCriterion, NMTCriterion, MSELoss
from src.modules.embeddings import Embeddings
from src.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention, \
    MultiInputPositionwiseFeedForward, MultiInputGates
from src.modules.word_predictor import MultiTargetNMTCriterion, WordPredictor, \
    convert_to_past_labels, convert_to_future_labels
from src.utils import nest
from src.utils.logging import INFO, WARN
from src.utils.nn_ops import get_prev_sequence_average, get_sub_sequence_average


def get_attn_causal_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.

    :param seq: Input sequence.
        with shape [batch_size, time_steps, dim]
    '''
    assert seq.dim() == 3
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1):
        super(EncoderBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)

        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid,
                                               dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        input_norm = self.layer_norm(enc_input)
        context, _, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask)
        out = self.dropout(context) + enc_input

        return self.pos_ffn(out)


class Encoder(nn.Module):

    def __init__(
            self, n_src_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None):
        super().__init__()

        self.num_layers = n_layers
        self.embeddings = Embeddings(num_embeddings=n_src_vocab,
                                     embedding_dim=d_word_vec,
                                     dropout=dropout,
                                     add_position_embedding=True
                                     )
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model=d_model, d_inner_hid=d_inner_hid, n_head=n_head, dropout=dropout,
                          dim_per_head=dim_per_head)
             for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq):
        # Word embedding look up
        batch_size, src_len = src_seq.size()

        emb = self.embeddings(src_seq)

        enc_mask = src_seq.detach().eq(PAD)
        enc_slf_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, src_len, src_len)

        out = emb

        for i in range(self.num_layers):
            out = self.block_stack[i](out, enc_slf_attn_mask)

        out = self.layer_norm(out)

        return out, enc_mask


class DecoderBlock(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner_hid, n_head, dim_per_head, dropout=0.1,
                 dim_capsule=100, num_capsules=0, null_capsule=False):
        super(DecoderBlock, self).__init__()

        self.slf_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
                                             dim_per_head=dim_per_head)
        # self.ctx_attn = MultiHeadedAttention(head_count=n_head, model_dim=d_model, dropout=dropout,
        #                                      dim_per_head=dim_per_head)
        self.pos_ffn = PositionwiseFeedForward(size=d_model, hidden_size=d_inner_hid)

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # contextual capsule layer
        self.apply_capsule = True
        # self.pre_capsule_layer_norm = nn.LayerNorm(d_model)

        assert dim_capsule % num_capsules == 0
        self.dim_per_cap = dim_capsule // num_capsules
        dim_per_part = dim_capsule // 3
        total_num_capsules = num_capsules

        self.null_caps = null_capsule
        if null_capsule:
            INFO("Using Null Capsules to attract irrelevant routing.")
            total_num_capsules += num_capsules // 3

        self.capsule_layer = ContextualCapsuleLayer(
            num_out_caps=total_num_capsules, num_in_caps=None,
            dim_in_caps=d_model,
            dim_out_caps=self.dim_per_cap,
            dim_context=d_model,
            num_iterations=3,
            share_route_weights_for_in_caps=True)

        self.out_and_cap_ffn = MultiInputPositionwiseFeedForward(
            size=d_model, hidden_size=d_inner_hid, dropout=dropout,
            inp_sizes=[dim_per_part, dim_per_part, dim_per_part]
        )

    def compute_cache(self, enc_output):
        return self.ctx_attn.compute_cache(enc_output, enc_output)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None,
                enc_attn_cache=None, self_attn_cache=None, capsule_cache=None):
        # Args Checks
        batch_size, query_len, _ = dec_input.size()

        # Self attention
        input_norm = self.layer_norm_1(dec_input)
        slfattn_out, _, self_attn_cache = self.slf_attn(input_norm, input_norm, input_norm,
                                                        mask=slf_attn_mask,
                                                        self_attn_cache=self_attn_cache)

        slfattn_out = self.dropout(slfattn_out) + dec_input

        # Cross attention
        query = slfattn_out
        query_norm = self.layer_norm_2(query)
        # xattn_out, attn, enc_attn_cache = self.ctx_attn(enc_output, enc_output, query_norm,
        #                                                 mask=dec_enc_attn_mask,
        #                                                 enc_attn_cache=enc_attn_cache)

        # pre-input layer norm

        # capsule layer
        # [batch, length, num_out_caps, d_capsule]
        enc_mask = dec_enc_attn_mask[:, 0, :].squeeze(1)
        capsules, routing_weight = self.capsule_layer.forward_sequence(
            enc_output,
            enc_mask,
            query_norm,
            cache=capsule_cache
        )
        # [batch, length, num_out_caps * d_capsule]
        capsules = capsules.view(batch_size, query_len, -1)

        # get past and future capsules
        if not self.null_caps:
            (present_caps, past_caps, future_caps) = torch.chunk(capsules, 3, -1)
        else:
            (present_caps, past_caps, future_caps, _) = torch.chunk(capsules, 4, -1)

        # combine output and capsules by ffn, with residual connection solely from output
        mid = self.out_and_cap_ffn(query_norm, present_caps, past_caps, future_caps)

        output = self.pos_ffn(self.dropout(mid) + query)

        return {
            "output": output,
            # "attn": attn,
            "self_attn_cache": self_attn_cache,
            "enc_attn_cache": enc_attn_cache,
            "present_caps": present_caps,
            "past_caps": past_caps,
            "future_caps": future_caps,
            "routing_weights": routing_weight
        }


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_layers=6, n_head=8,
            capsule_type="output", routing_type="dynamic_routing", comb_type="ffn",
            dim_capsule=100, num_capsules=8, null_capsule=False,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None, dropout=0.1):

        super(Decoder, self).__init__()

        self.n_head = n_head
        self.num_layers = n_layers
        self.d_model = d_model

        self.embeddings = Embeddings(n_tgt_vocab, d_word_vec,
                                     dropout=dropout, add_position_embedding=True)

        self.block_stack = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_inner_hid=d_inner_hid,
                         n_head=n_head, dropout=dropout,
                         dim_per_head=dim_per_head,
                         dim_capsule=dim_capsule,
                         num_capsules=num_capsules if capsule_type.startswith("layer-wise") else 0,
                         null_capsule=null_capsule)
            for _ in range(n_layers)])

        self.out_layer_norm = nn.LayerNorm(d_model)

        self._dim_per_head = dim_per_head
        #
        # # contextual capsule layer
        # if capsule_type == "output":
        #     self.apply_output_capsule = True
        #     self.pre_capsule_layer_norm = nn.LayerNorm(d_model)
        #
        #     assert dim_capsule % num_capsules == 0
        #     self.dim_per_cap = dim_capsule // num_capsules
        #
        #     self.null_caps = null_capsule
        #     if null_capsule:
        #         INFO("Using Null Capsules to attract irrelevant routing.")
        #
        #     total_num_capsules = num_capsules if not self.null_caps else int(num_capsules * 1.5)
        #
        #     self.routing_type = routing_type
        #     if routing_type == "dynamic_routing":
        #         self.final_capsule_layer = ContextualCapsuleLayer(
        #             num_out_caps=total_num_capsules, num_in_caps=None,
        #             dim_in_caps=d_model,
        #             dim_out_caps=self.dim_per_cap,
        #             dim_context=d_model,
        #             num_iterations=3,
        #             share_route_weights_for_in_caps=True)
        #
        #     elif routing_type == "EM_routing":
        #         self.final_capsule_layer = EMContextualCapsuleLayer(
        #             num_out_caps=total_num_capsules, num_in_caps=None,
        #             dim_in_caps=d_model, dim_out_caps=self.dim_per_cap, dim_context=d_model,
        #             num_iterations=3,
        #             share_route_weights_for_in_caps=True)
        #
        #     dim_per_part = dim_capsule // 2
        #     if comb_type == "ffn":
        #         self.out_and_cap_ffn = MultiInputPositionwiseFeedForward(
        #             size=d_model, hidden_size=d_inner_hid, dropout=dropout,
        #             inp_sizes=[dim_per_part, dim_per_part]
        #         )
        #     elif comb_type == "gate":
        #         self.out_and_cap_ffn = MultiInputGates(
        #             d_model=d_model, input_sizes=[dim_per_part, dim_per_part],
        #             dropout=dropout
        #         )
        # else:
        #     self.apply_output_capsule = False

        if capsule_type == "layer-wise-share":
            for i in range(1, n_layers):
                self.block_stack[i].capsule_layer = self.block_stack[0].capsule_layer
                self.block_stack[i].out_and_cap_ffn = self.block_stack[0].out_and_cap_ffn

    @property
    def dim_per_head(self):
        if self._dim_per_head is None:
            return self.d_model // self.n_head
        else:
            return self._dim_per_head

    def forward(self, tgt_seq, enc_output, enc_mask,
                enc_attn_caches=None, self_attn_caches=None,
                capsule_caches=None, layer_wise_capsule_caches=None):

        batch_size, tgt_len = tgt_seq.size()

        query_len = tgt_len
        key_len = tgt_len

        src_len = enc_output.size(1)

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt_seq)

        if self_attn_caches is not None:
            emb = emb[:, -1:].contiguous()
            query_len = 1

        # Decode mask
        dec_slf_attn_pad_mask = tgt_seq.detach().eq(PAD).unsqueeze(1).expand(batch_size, query_len, key_len)
        dec_slf_attn_sub_mask = get_attn_causal_mask(emb)

        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_mask = enc_mask.unsqueeze(1).expand(batch_size, query_len, src_len)

        output = emb
        new_self_attn_caches = []
        new_enc_attn_caches = []

        for i in range(self.num_layers):
            layer_states = self.block_stack[i](
                output,
                enc_output,
                dec_slf_attn_mask,
                dec_enc_attn_mask,
                enc_attn_cache=enc_attn_caches[i] if enc_attn_caches is not None else None,
                self_attn_cache=self_attn_caches[i] if self_attn_caches is not None else None,
                capsule_cache=layer_wise_capsule_caches[i] if layer_wise_capsule_caches is not None else None
            )
            output = layer_states["output"]

            new_self_attn_caches += [layer_states["self_attn_cache"]]
            new_enc_attn_caches += [layer_states["enc_attn_cache"]]

        present_caps = layer_states["present_caps"]
        past_caps = layer_states["past_caps"]
        future_caps = layer_states["future_caps"]

        # # final capsule layer before softmax layer
        # # pre-input layer norm
        # if self.apply_output_capsule:
        #     capsule_query = self.pre_capsule_layer_norm(output)
        #
        #     # capsule layer
        #     # [batch, length, num_out_caps, d_capsule]
        #     capsules, _ = self.final_capsule_layer.forward_sequence(
        #         enc_output,
        #         enc_mask,
        #         capsule_query,
        #         cache=capsule_caches
        #     )
        #     # [batch, length, num_out_caps * d_capsule]
        #     capsules = capsules.view(batch_size, query_len, -1)
        #
        #     # get past and future capsules
        #     if not self.null_caps:
        #         (past_caps, future_caps) = torch.chunk(capsules, 2, -1)
        #     else:
        #         (past_caps, future_caps, _) = torch.chunk(capsules, 3, -1)
        #
        #     # combine output and capsules by ffn, with residual connection solely from output
        #     output = self.out_and_cap_ffn(output, past_caps, future_caps)
        # else:
        #     past_caps, future_caps = layer_states["past_caps"], layer_states["future_caps"]

        # output layer norm
        output = self.out_layer_norm(output)

        ret = {
            "output": output,
            "self_attn_caches": new_self_attn_caches,
            "enc_attn_caches": new_enc_attn_caches,
            "past_capsules": past_caps, "future_capsules": future_caps,
            "present_capsules": present_caps
        }

        return ret


class Generator(nn.Module):
    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class DynamicTransformer(NMTModel):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_tgt_vocab, **config):
        super().__init__()

        self.config = config

        self.encoder = Encoder(
            n_src_vocab,
            n_layers=config["n_layers"],
            n_head=config["n_head"],
            d_word_vec=config["d_word_vec"],
            d_model=config["d_model"],
            d_inner_hid=config["d_inner_hid"],
            dropout=config["dropout"],
        )

        self.decoder = Decoder(
            n_tgt_vocab,
            n_layers=config["n_layers"],
            n_head=config["n_head"],
            d_word_vec=config["d_word_vec"],
            d_model=config["d_model"],
            d_inner_hid=config["d_inner_hid"],
            dropout=config["dropout"],
            # capsule configs
            capsule_type=config["capsule_type"],
            comb_type=config.setdefault("comb_type", "ffn"),
            routing_type=config.setdefault("routing_type", "dynamic_routing"),
            dim_capsule=config["d_capsule"],
            num_capsules=config["num_capsules"],
            null_capsule=config["null_capsule"]
        )

        assert config["d_model"] == config["d_word_vec"], \
            'To facilitate the residual connections, \
             the dimensions of all module output shall be the same.'

        self.generator = Generator(n_words=n_tgt_vocab,
                                   hidden_size=config["d_word_vec"],
                                   padding_idx=PAD)
        if config["proj_share_weight"]:
            self.generator.proj.weight = self.decoder.embeddings.embeddings.weight

        if config.setdefault("tie_source_target_embeddings", False):
            self.encoder.embeddings.embeddings.weight = self.decoder.embeddings.embeddings.weight

        self.capsule_per_dim = config['d_capsule'] // 3
        # if self.config["apply_word_prediction_loss"]:
        #     self.wp_past = WordPredictor(generator=self.generator,
        #                                  input_size=self.capsule_per_dim,
        #                                  d_word_vec=config['d_word_vec'])
        #     self.wp_future = WordPredictor(generator=self.generator,
        #                                    input_size=self.capsule_per_dim,
        #                                    d_word_vec=config['d_word_vec'])

        # criterion
        self.criterion = MultiCriterion(
            weights=dict(nmt_nll=1.),
            nmt_nll=NMTCriterion(label_smoothing=config['label_smoothing'])
        )

        if "wp" in config["auxiliary_loss"] or self.config["apply_word_prediction_loss"]:
            self.config["apply_word_prediction_loss"] = True
            self.wp_present = WordPredictor(generator=self.generator,
                                            input_size=self.capsule_per_dim,
                                            d_word_vec=config['d_word_vec'])
            self.criterion.add(
                "wploss_present",
                NMTCriterion(label_smoothing=config['label_smoothing']),
                weight=1.
            )
            self.wp_past = WordPredictor(generator=self.generator,
                                         input_size=self.capsule_per_dim,
                                         d_word_vec=config['d_word_vec'])
            self.wp_future = WordPredictor(generator=self.generator,
                                           input_size=self.capsule_per_dim,
                                           d_word_vec=config['d_word_vec'])
            self.criterion.add(
                "wploss_past",
                MultiTargetNMTCriterion(label_smoothing=config['label_smoothing']),
                weight=1.
            )
            self.criterion.add(
                "wploss_future",
                MultiTargetNMTCriterion(label_smoothing=config['label_smoothing']),
                weight=1.
            )
        if "bca" in config["auxiliary_loss"]:
            self.linear_bca_past = nn.Linear(self.capsule_per_dim, config["d_model"])
            self.criterion.add(
                "bca_past",
                MSELoss(),
                weight=1.
            )
            self.linear_bca_future = nn.Linear(self.capsule_per_dim, config["d_model"])
            self.criterion.add(
               "bca_future",
                MSELoss(),
                weight=1.
            )

    def forward(self, src_seq, tgt_seq, log_probs=True):
        enc_output, enc_mask = self.encoder(src_seq)
        dec_states = self.decoder(tgt_seq, enc_output, enc_mask)
        output = dec_states["output"]

        logprobs = self.generator(output, log_probs=log_probs)
        ret = {"logprobs_nmt": logprobs, "dec_outs": output}

        if self.config['apply_word_prediction_loss']:
            logprobs_present = self.wp_present(dec_states['present_capsules'])
            logprobs_past = self.wp_past(dec_states['past_capsules'])
            logprobs_future = self.wp_future(dec_states['future_capsules'])
            ret['logprobs_present'] = logprobs_present
            ret['logprobs_past'] = logprobs_past
            ret['logprobs_future'] = logprobs_future

        if "bca" in self.config["auxiliary_loss"]:
            ret["past_capsules"] = self.linear_bca_past(dec_states["past_capsules"])
            ret["future_capsules"] = self.linear_bca_future(dec_states["future_capsules"])
        return ret

    def critic(self, inputs, labels,
               reduce=True, normalization=1.0, **kwargs):
        next_labels = labels[:, 1:].contiguous()
        prev_lables = labels[:, :-1].contiguous()
        params_nmt = dict(
            inputs=inputs['logprobs_nmt'],
            labels=next_labels,
            update=True)

        params_dict = {"nmt_nll": params_nmt}

        if self.config['apply_word_prediction_loss']:
            past_labels, past_scores = convert_to_past_labels(prev_lables)
            params_wploss_past = dict(
                inputs=inputs['logprobs_past'],
                labels=past_labels,
                target_scores=past_scores,
                update=True)

            future_labels, future_scores = convert_to_future_labels(next_labels)
            params_wploss_future = dict(
                inputs=inputs['logprobs_future'],
                labels=future_labels,
                target_scores=future_scores,
                update=True)
            params_dict["wploss_past"] = params_wploss_past
            params_dict["wploss_future"] = params_wploss_future

            params_wploss_present = dict(
                inputs=inputs['logprobs_present'],
                labels=next_labels,
                update=True)
            params_dict["wploss_present"] = params_wploss_present

        if "bca" in self.config["auxiliary_loss"]:
            dec_non_mask = labels.ne(PAD)
            params_bca_past = dict(
                inputs=inputs["past_capsules"],
                labels=get_prev_sequence_average(inputs["dec_outs"]),
                mask=dec_non_mask,
                update=True
            )
            params_bca_future = dict(
                inputs=inputs["past_capsules"],
                labels=get_sub_sequence_average(inputs["dec_outs"]),
                mask=dec_non_mask,
                update=True
            )
            params_dict["bca_past"] = params_bca_past
            params_dict["bca_future"] = params_bca_future

        loss = self.criterion(
            reduce=reduce,
            normalization=normalization,
            # params for each criterion
            **params_dict
        )

        return loss

    def encode(self, src_seq):

        ctx, ctx_mask = self.encoder(src_seq)

        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def init_decoder(self, enc_outputs, expand_size=1):

        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        capsule_caches, layer_wise_capsule_caches = None, None

        if expand_size > 1:
            ctx = tile_batch(ctx, multiplier=expand_size)
            ctx_mask = tile_batch(ctx_mask, multiplier=expand_size)

            # Note that ctx has been tiled
            if self.config["capsule_type"] == "output":
                capsule_caches = self.decoder.final_capsule_layer.compute_caches(ctx)

            elif self.config["capsule_type"] == "layer-wise":
                layer_wise_capsule_caches = []
                for i in range(self.config["n_layers"]):
                    layer_capsule_cache = self.decoder.block_stack[i].capsule_layer.compute_caches(ctx)
                    layer_wise_capsule_caches.append(
                        layer_capsule_cache
                    )

        return {
            "ctx": ctx,
            "ctx_mask": ctx_mask,
            "enc_attn_caches": None,
            "self_attn_caches": None,
            "capsule_caches": capsule_caches,
            "layer_wise_capsule_caches": layer_wise_capsule_caches
        }

    def decode(self, tgt_seq, dec_states, log_probs=True):
        ctx = dec_states["ctx"]
        ctx_mask = dec_states['ctx_mask']
        enc_attn_caches = dec_states['enc_attn_caches']
        self_attn_caches = dec_states['self_attn_caches']
        capsule_caches = dec_states["capsule_caches"]
        layer_wise_capsule_caches = dec_states["layer_wise_capsule_caches"]

        new_dec_states = self.decoder(
            tgt_seq=tgt_seq,
            enc_output=ctx,
            enc_mask=ctx_mask,
            enc_attn_caches=enc_attn_caches,
            self_attn_caches=self_attn_caches,
            capsule_caches=capsule_caches,
            layer_wise_capsule_caches=layer_wise_capsule_caches
        )

        dec_output = new_dec_states["output"]

        next_scores = self.generator(dec_output[:, -1].contiguous(), log_probs=log_probs)

        dec_states['enc_attn_caches'] = new_dec_states["enc_attn_caches"]
        dec_states['self_attn_caches'] = new_dec_states["self_attn_caches"]

        return next_scores, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):
        self_attn_caches = dec_states['self_attn_caches']

        batch_size = self_attn_caches[0][0].size(0) // beam_size

        n_head = self.decoder.n_head
        dim_per_head = self.decoder.dim_per_head

        self_attn_caches = nest.map_structure(
            lambda t: tensor_gather_helper(
                gather_indices=new_beam_indices,
                gather_from=t,
                batch_size=batch_size,
                beam_size=beam_size,
                gather_shape=[batch_size * beam_size, n_head, -1, dim_per_head]
            ),
            self_attn_caches
        )

        dec_states['self_attn_caches'] = self_attn_caches

        return dec_states

    def load_pretrain_model(self, pretrain_path="", device=None):
        if not pretrain_path:
            return
        else:
            INFO("Loading pretrained parameters from {}".format(
                pretrain_path))

            need_pretrain_params_prefix = [
                'encoder',
                'decoder',
                'generator',
            ]

            pretrained_params_name = []
            pretrain_params = torch.load(pretrain_path, map_location=device)
            for name, param in pretrain_params.items():
                for pp in need_pretrain_params_prefix:
                    if name.startswith(pp):
                        INFO("Loading param: {}...".format(name))
                        try:
                            self.load_state_dict({name: param}, strict=False)
                        except Exception as e:
                            WARN("{}: {}".format(str(Exception), e))
                        pretrained_params_name.append(name)

            # Frozen pretrained parameters
            #            for name, param in self.named_parameters():
            #                if name in pretrained_params_name:
            #                    param.requires_grad = False
            #
            INFO("Pretrained model loaded.")
