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
from src.decoding.utils import tile_batch, tensor_gather_helper
from src.models import TransEncRNNDec
from src.models.dl4mt import Decoder
from src.modules.capsule import CapsuleLayer, ContextualCapsuleLayer
from src.modules.criterions import MultiCriterion, NMTCriterion
from src.modules.word_predictor import WordPredictor, MultiTargetNMTCriterion, \
    convert_to_past_labels, \
    convert_to_future_labels
from src.utils.logging import INFO, WARN


class Decoder(Decoder):
    def __init__(self,
                 n_words,
                 input_size,
                 hidden_size,
                 context_size,
                 num_capsules,
                 capsule_type=['init'],
                 dim_contextual_capsule=240,
                 bridge_type="mlp",
                 dropout_rate=0.0,
                 cell_type='cgru'):
        super().__init__(n_words, input_size, hidden_size, context_size,
                         bridge_type, dropout_rate, cell_type)

        assert hidden_size % num_capsules == 0
        self.num_capsules = num_capsules
        self.capsule_type = capsule_type
        if 'init' in capsule_type:
            self.capsule_layer = CapsuleLayer(
                num_out_caps=num_capsules, num_in_caps=None,
                dim_in_caps=context_size,
                dim_out_caps=hidden_size // num_capsules,
                num_iterations=3, share_route_weights_for_in_caps=True)
            self.linear_capsule = nn.Linear(in_features=hidden_size,
                                            out_features=input_size)
        if 'contextual' in capsule_type:
            self.dim_contextual_capsule = dim_contextual_capsule
            self.contextual_capsule_layer = ContextualCapsuleLayer(
                num_out_caps=num_capsules, num_in_caps=None,
                dim_in_caps=context_size,
                dim_out_caps=self.dim_contextual_capsule // num_capsules,
                dim_context=hidden_size + context_size,
                num_iterations=3,
                share_route_weights_for_in_caps=True)
            self.linear_contextual_capsule = nn.Linear(
                in_features=self.dim_contextual_capsule,
                out_features=input_size)

    def init_decoder(self, context, mask):
        dec_init, dec_cache = super().init_decoder(context, mask)

        if 'init' in self.capsule_type:
            capsule_mask = mask.float()  # [batch_size, context_size, 1]
            # [batch_size, num_capsules, hidden_size // num_capsules]
            capsule = self.capsule_layer(context, capsule_mask)
            capsule_cat = capsule.view(capsule.size(0), self.hidden_size)
        else:
            capsule_cat = None

        return dec_init, dec_cache, capsule_cat

    def forward(self, y, context, context_mask, hidden, dec_capsule=None,
                one_step=False, cache=None):

        emb = self.embeddings(y)  # [batch_size, seq_len, dim]

        if one_step:
            (out, attn), hidden = self.cell(
                emb, hidden, context, context_mask, cache
            )
            if 'contextual' in self.capsule_type:
                ctx_cap, routing_weights = self.contextual_capsule_layer(
                    context, context_mask.float(),
                    torch.cat([out, attn], -1)
                )
                ctx_cap = ctx_cap.view(
                    ctx_cap.size(0), self.dim_contextual_capsule
                )
        else:
            # emb: [batch_size, seq_len, dim]
            out = []
            attn = []

            if 'contextual' in self.capsule_type:
                ctx_cap = []

            for emb_t in torch.split(emb, split_size_or_sections=1, dim=1):
                (out_t, attn_t), hidden = self.cell(
                    emb_t.squeeze(1), hidden, context, context_mask, cache
                )

                out += [out_t]
                attn += [attn_t]

                if 'contextual' in self.capsule_type:
                    ctx_cap_t, _ = self.contextual_capsule_layer(
                        context,
                        context_mask.float(),
                        torch.cat([out_t, attn_t], -1)
                    )
                    ctx_cap += [ctx_cap_t.view(ctx_cap_t.size(0),
                                               self.dim_contextual_capsule)]

            out = torch.stack(out).transpose(1, 0).contiguous()
            attn = torch.stack(attn).transpose(1, 0).contiguous()

            if 'contextual' in self.capsule_type:
                ctx_cap = torch.stack(ctx_cap).transpose(1, 0).contiguous()

        logits = self.linear_input(emb) + self.linear_hidden(
            out) + self.linear_ctx(attn)

        if 'init' in self.capsule_type and dec_capsule is not None:
            # dec_capsule: [batch_size, hidden_size]
            logits_capsule = self.linear_capsule(dec_capsule)
            if not one_step:
                logits_capsule = logits_capsule.unsqueeze(1)
            logits = logits + logits_capsule

        if 'contextual' in self.capsule_type:
            logits = logits + self.linear_contextual_capsule(ctx_cap)
            (past_caps, future_caps) = torch.split(
                ctx_cap,
                self.dim_contextual_capsule // 2,
                dim=-1
            )

        logits = F.tanh(logits)

        logits = self.dropout(logits)  # [batch_size, seq_len, dim]

        ret = {
            'logits': logits, 'hiddens': out,
            'attention_contexts': attn, 'embeddings': emb,
            'past_capsules': past_caps, 'future_capsules': future_caps
        }
        if not self.training:
            ret["routing_weights"] = routing_weights


class CapsuleNMT(TransEncRNNDec):
    def __init__(self, n_src_vocab, n_tgt_vocab, **config):
        super().__init__(n_src_vocab, n_tgt_vocab, **config)
        self.decoder = Decoder(
            cell_type=config.setdefault('decoder_cell_type', 'cgru'),
            n_words=n_tgt_vocab, input_size=config['d_word_vec'],
            hidden_size=config['d_model'], context_size=config['d_model'],
            capsule_type=config['capsule_type'],
            dropout_rate=config['dropout'],
            bridge_type=config['bridge_type'],
            num_capsules=config['num_capsules'])

        if self.config["apply_word_prediction_loss"]:
            per_dim = config['d_contextual_capsule'] // 2
            self.wp_past = WordPredictor(generator=self.generator,
                                         input_size=per_dim,
                                         d_word_vec=config['d_word_vec'])
            self.wp_future = WordPredictor(generator=self.generator,
                                           input_size=per_dim,
                                           d_word_vec=config['d_word_vec'])

        self.criterion = MultiCriterion(
            weights=dict(nmt_nll=1., wploss_past=1., wploss_future=1.),
            nmt_nll=NMTCriterion(label_smoothing=config['label_smoothing']),
            wploss_past=MultiTargetNMTCriterion(
                label_smoothing=config['label_smoothing']),
            wploss_future=MultiTargetNMTCriterion(
                label_smoothing=config['label_smoothing']))

    def critic(self, inputs, labels,
               reduce=True, normalization=1.0, **kwargs):

        params_nmt = dict(
            inputs=inputs['logprobs_nmt'],
            labels=labels,
            update=True)

        if self.config['apply_word_prediction_loss']:
            past_labels, past_scores = convert_to_past_labels(labels)
            params_wploss_past = dict(
                inputs=inputs['logprobs_past'],
                labels=past_labels,
                target_scores=past_scores,
                update=True)

            future_labels, future_scores = convert_to_future_labels(labels)
            params_wploss_future = dict(
                inputs=inputs['logprobs_future'],
                labels=future_labels,
                target_scores=future_scores,
                update=True)
        else:
            params_wploss_past = params_wploss_future = None

        loss = self.criterion(
            reduce=reduce,
            normalization=normalization,
            # params for each criterion
            nmt_nll=params_nmt,
            wploss_past=params_wploss_past,
            wploss_future=params_wploss_future)

        return loss

    def load_pretrain_model(self, pretrain_path=None, device=None):
        if pretrain_path is None:
            return
        else:
            INFO("Loading pretrained parameters for LM from {}".format(
                pretrain_path))

            need_pretrain_params_prefix = [
                'encoder',
                'decoder.embeddings',
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

    def init_decoder(self, enc_outputs, expand_size=1):
        ctx = enc_outputs['ctx']

        ctx_mask = enc_outputs['ctx_mask']

        dec_init, dec_caches, dec_capsule = self.decoder.init_decoder(
            context=ctx, mask=ctx_mask)

        if expand_size > 1:
            ctx = tile_batch(ctx, expand_size)
            ctx_mask = tile_batch(ctx_mask, expand_size)
            dec_init = tile_batch(dec_init, expand_size)
            dec_caches = tile_batch(dec_caches, expand_size)
            if dec_capsule is not None:
                dec_capsule = tile_batch(dec_capsule, expand_size)

        return {"dec_hiddens": dec_init, "dec_caches": dec_caches,
                "ctx": ctx, "ctx_mask": ctx_mask, "dec_capsule": dec_capsule,
                "routing_weights": None}

    def forward(self, src_seq, tgt_seq, log_probs=True):
        enc_ctx, enc_mask = self.encoder(src_seq)

        dec_init, dec_cache, dec_capsule = self.decoder.init_decoder(enc_ctx,
                                                                     enc_mask)

        dec_states = self.decoder(tgt_seq,
                                  context=enc_ctx,
                                  context_mask=enc_mask,
                                  one_step=False,
                                  hidden=dec_init,
                                  cache=dec_cache,
                                  dec_capsule=dec_capsule)
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

        ret_loss = {'logprobs_nmt': final_log_probs / self.num_scores}
        if self.config['apply_word_prediction_loss']:
            logprobs_past = self.wp_past(dec_states['past_capsules'])
            logprobs_future = self.wp_future(dec_states['future_capsules'])
            ret_loss['logprobs_past'] = logprobs_past
            ret_loss['logprobs_future'] = logprobs_future

        return ret_loss

    def decode(self, tgt_seq, dec_states, log_probs=True):
        ctx = dec_states['ctx']
        ctx_mask = dec_states['ctx_mask']
        dec_capsule = dec_states['dec_capsule']

        prev_dec_hiddens = dec_states['dec_hiddens']
        dec_caches = dec_states['dec_caches']
        routing_weights = dec_states["routing_weights"]

        final_word_indices = tgt_seq[:, -1].contiguous()

        dec_states = self.decoder(final_word_indices, hidden=prev_dec_hiddens,
                                  context=ctx, context_mask=ctx_mask,
                                  one_step=True, cache=dec_caches,
                                  dec_capsule=dec_capsule)
        logits = dec_states['logits']
        hiddens = dec_states['hiddens']
        attention_contexts = dec_states['attention_contexts']

        # [batch, num_in_caps, num_out_caps]
        routing_weight = dec_states["routing_weights"]
        # [batch, len, num_in_caps, num_out_caps]
        if routing_weights is None:
            routing_weights = routing_weight.unsqueeze(1)
        else:
            routing_weights = torch.cat(
                [routing_weights, routing_weight.unsqueeze(1)], 1)

        final_log_probs = self.generator(logits, log_probs=log_probs)

        if self.config['attention_context_word_prediction']:
            attn_wp_log_probs = self.attn_wp(attention_contexts, log_probs)
            final_log_probs = final_log_probs + attn_wp_log_probs

        if self.config['target_language_model']:
            tgt_lm_log_probs = self.tgt_lm(dec_states['embeddings'], log_probs)
            final_log_probs = final_log_probs + tgt_lm_log_probs

        final_log_probs = final_log_probs / self.num_scores

        dec_states = {"ctx": ctx, "ctx_mask": ctx_mask,
                      "dec_capsule": dec_capsule,
                      "dec_hiddens": hiddens, "dec_caches": dec_caches,
                      "routing_weights": routing_weights}

        return final_log_probs, dec_states

    def reorder_dec_states(self, dec_states, new_beam_indices, beam_size):

        dec_hiddens = dec_states["dec_hiddens"]

        batch_size = dec_hiddens.size(0) // beam_size

        dec_hiddens = tensor_gather_helper(
            gather_indices=new_beam_indices,
            gather_from=dec_hiddens,
            batch_size=batch_size,
            beam_size=beam_size,
            gather_shape=[batch_size * beam_size, -1]
        )

        dec_states['dec_hiddens'] = dec_hiddens

        # [batch, len, num_in_caps, num_out_caps]
        routing_weights = dec_states["routing_weights"]
        routing_weights = tensor_gather_helper(
            gather_indices=new_beam_indices,
            gather_from=routing_weights,
            batch_size=batch_size,
            beam_size=beam_size,
            gather_shape=[batch_size * beam_size, -1,
                          routing_weights.size(1), routing_weights.size(2)]
        )

        dec_states["routing_weights"] = routing_weights

        return dec_states
