# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8th 10:58:37 2016

beam search modules

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import utils

dtype=theano.config.floatX
#

#TODO: beam search seq2seq with LSTM
class BeamSearchSelGen(object):
    '''
    This is a beam search code for sel gen model
    '''
    def __init__(self, settings):
        print "initializing the beam searcher ... "
        assert (settings['size_beam'] >= 1)
        #assert (settings['path_model'] != None)
        #
        self.path_model = None
        if settings['path_model'] != None:
            self.path_model = os.path.abspath(
                settings['path_model']
            )
        #
        self.size_beam = settings['size_beam']
        self.normalize_mode = settings['normalize_mode']
        # whether to normalize the cost over length of sequence
        #
        self.h_0 = None
        self.c_0 = None
        self.scope_att = None
        self.weights_pre_sel = None
        self.beam_list = []
        self.finish_list = []
        #
        if self.path_model != None :
            with open(self.path_model, 'rb') as f:
                self.model = pickle.load(f)
            # convert float64 to float32
            for param_name in self.model:
                self.model[param_name] = numpy.float32(self.model[param_name])
            #
            self.dim_model = self.model['dim_model']
            self.dim_lang = self.model['dim_lang']
            self.dim_info = self.model['dim_info']
            self.num_sel = self.model['num_sel']
            #
            self.vocabmat = numpy.identity(
                self.dim_lang, dtype = dtype
            )
            #
        #
    #
    def set_model(self, dict_model):
        self.model = dict_model
        for param_name in self.model:
            self.model[param_name] = numpy.float32(self.model[param_name])
        #
        self.dim_model = self.model['dim_model']
        self.dim_lang = self.model['dim_lang']
        self.dim_info = self.model['dim_info']
        self.num_sel = self.model['num_sel']
        #
        self.vocabmat = numpy.identity(
            self.dim_lang, dtype = dtype
        )
        #
    #

    def refresh_state(self):
        #print "refreshing the states of beam search ... "
        self.h_0 = None
        self.c_0 = None
        self.scope_att = None
        self.weights_pre_sel = None
        self.beam_list = []
        self.finish_list = []


    def sigmoid(self, x):
        return 1.0 / (1.0+numpy.exp(-x))

    def set_encoder(self, seq_info_numpy):
        '''
        this function sets the encoder states, given the source_seq_numpy as vector (:,)
        '''
        seq_info_forward = seq_info_numpy
        seq_info_backward = seq_info_numpy[::-1, :]
        #
        seq_emb_info_forward = numpy.dot(
            seq_info_forward, self.model['Emb_enc_forward']
        )
        seq_emb_info_backward = numpy.dot(
            seq_info_backward, self.model['Emb_enc_backward']
        )
        #
        shape_encode = seq_emb_info_backward.shape
        h_forward = numpy.zeros(
            shape_encode, dtype = dtype
        )
        c_forward = numpy.zeros(
            shape_encode, dtype = dtype
        )
        h_backward = numpy.zeros(
            shape_encode, dtype = dtype
        )
        c_backward = numpy.zeros(
            shape_encode, dtype = dtype
        )
        #
        len_source = shape_encode[0]
        for time_stamp in range(-1, len_source-1, 1):
            pretran_forward = numpy.concatenate(
                (
                    seq_emb_info_forward[time_stamp+1, :],
                    h_forward[time_stamp, :]
                ), axis = 0
            )
            pretran_backward = numpy.concatenate(
                (
                    seq_emb_info_backward[time_stamp+1, :],
                    h_backward[time_stamp, :]
                ), axis = 0
            )
            #
            posttran_forward = numpy.dot(
                pretran_forward, self.model['W_enc_forward']
            ) + self.model['b_enc_forward']
            posttran_backward = numpy.dot(
                pretran_backward, self.model['W_enc_backward']
            ) + self.model['b_enc_backward']
            #
            i_t_forward = self.sigmoid(
                posttran_forward[0:self.dim_model]
            )
            f_t_forward = self.sigmoid(
                posttran_forward[
                    self.dim_model:2*self.dim_model
                ]
            )
            g_t_forward = numpy.tanh(
                posttran_forward[
                    2*self.dim_model:3*self.dim_model
                ]
            )
            o_t_forward = self.sigmoid(
                posttran_forward[3*self.dim_model:]
            )
            c_forward[time_stamp+1,:] = numpy.copy(
                f_t_forward * c_forward[time_stamp,:] + i_t_forward * g_t_forward
            )
            h_forward[time_stamp+1, :] = numpy.copy(
                o_t_forward * numpy.tanh(
                    c_forward[time_stamp+1, :]
                )
            )
            #
            i_t_backward = self.sigmoid(
                posttran_backward[0:self.dim_model]
            )
            f_t_backward = self.sigmoid(
                posttran_backward[
                    self.dim_model:2*self.dim_model
                ]
            )
            g_t_backward = numpy.tanh(
                posttran_backward[
                    2*self.dim_model:3*self.dim_model
                ]
            )
            o_t_backward = self.sigmoid(
                posttran_backward[3*self.dim_model:]
            )
            c_backward[time_stamp+1, :] = numpy.copy(
                f_t_backward * c_backward[time_stamp, :] + i_t_backward * g_t_backward
            )
            h_backward[time_stamp+1, :] = numpy.copy(
                o_t_backward * numpy.tanh(
                    c_backward[time_stamp+1, :]
                )
            )
            #
        self.scope_att = numpy.concatenate(
            (
                seq_info_forward,
                h_forward, h_backward[::-1, :]
            ), axis = 1
        )
        #
        pre_Alpha = numpy.tanh(
            numpy.dot(
                self.scope_att, self.model['W_pre_att']
            )
        )
        self.weights_pre_sel = self.sigmoid(
            numpy.dot(
                pre_Alpha, self.model['b_pre_att']
            )
        )
        #
        self.h_0 = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.c_0 = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #

    def init_beam(self):
        #print "initialize beam ... "
        item  = {
            'htm1': numpy.copy(self.h_0),
            'ctm1': numpy.copy(self.c_0),
            'input_word_idx': 0,
            'list_idx_token': [0], # 0 -- Special
            'list_idx_att': [],
            'continue': True,
            'length': 1, 'cost': 0.0, 'norm_cost': 0.0
        }
        self.beam_list.append(item)

    def softmax(self, x):
        # x is a vector
        exp_x = numpy.exp(x - numpy.amax(x))
        return exp_x / numpy.sum(exp_x)

    def decode_step(
        self, input_word_rep, htm1_target, ctm1_target
    ):
        emb_word = numpy.dot(
            input_word_rep, self.model['Emb_dec']
        )
        #
        beta1 = numpy.dot(
            self.scope_att, self.model['U_att']
        )
        beta2 = numpy.dot(
            htm1_target, self.model['W_att']
        )
        beta3 = numpy.tanh( beta1 + beta2 )
        beta4 = numpy.dot(
            beta3, self.model['b_att']
        )
        pre_alpha = self.softmax(
            beta4
        )
        #
        pre_alpha *= self.weights_pre_sel
        alpha = pre_alpha / numpy.sum(pre_alpha)
        #
        z_t = numpy.dot(
            alpha, self.scope_att
        ) # (dim_scope, )
        #
        pre_tran = numpy.concatenate(
            (
                emb_word, htm1_target, z_t
            ), axis = 0
        )
        post_tran = numpy.dot(
            pre_tran, self.model['W_dec']
        ) + self.model['b_dec']
        #
        i_t = self.sigmoid(
            post_tran[:self.dim_model]
        )
        f_t = self.sigmoid(
            post_tran[self.dim_model:2*self.dim_model]
        )
        g_t = numpy.tanh(
            post_tran[2*self.dim_model:3*self.dim_model]
        )
        o_t = self.sigmoid(
            post_tran[3*self.dim_model:]
        )
        ct_target = f_t * ctm1_target + i_t * g_t
        ht_target = o_t * numpy.tanh(ct_target)
        #
        pre_y = numpy.concatenate(
            (
                ht_target, z_t
            ), axis = 0
        )
        y_t_0 = numpy.dot(
            (
                emb_word + numpy.dot(
                    pre_y, self.model['L']
                )
            ),
            self.model['L_0']
        )
        probt = self.softmax(y_t_0)
        log_probt = numpy.log(
            probt + numpy.float32(1e-8)
        )
        #
        return emb_word, alpha, ht_target, ct_target, probt, log_probt

    def search_func(self):
        #print "search for target ... "
        counter, max_counter = 0, 100
        while ((len(self.finish_list)<self.size_beam) and (counter<max_counter) ):
            new_list = []
            for item in self.beam_list:
                xt_item, alpha_item, ht_item, ct_item, probt_item, log_probt_item = self.decode_step(
                    self.vocabmat[
                        :, item['input_word_idx']
                    ],
                    item['htm1'], item['ctm1']
                )
                top_k_list = numpy.argsort(
                    -1.0*log_probt_item
                )[:self.size_beam]
                for top_token_idx in top_k_list:
                    new_item = {
                        'htm1': numpy.copy(ht_item),
                        'ctm1': numpy.copy(ct_item),
                        'input_word_idx': top_token_idx,
                        'list_idx_token': [
                            idx for idx in item['list_idx_token']
                        ],
                        'list_idx_att': [
                            idx for idx in item['list_idx_att']
                        ]
                    }
                    new_item['list_idx_token'].append(
                        top_token_idx
                    )
                    new_item['list_idx_att'].append(
                        numpy.argmax(alpha_item)
                    )
                    if top_token_idx == 0:
                        new_item['continue'] = False
                    else:
                        new_item['continue'] = True
                    new_item['length'] = item['length'] + 1
                    new_item['cost'] = item['cost'] + (-1.0)*log_probt_item[top_token_idx]
                    new_item['norm_cost'] = new_item['cost'] / new_item['length']
                    #
                    new_list.append(new_item)
            if self.normalize_mode:
                new_list = sorted(
                    new_list, key=lambda x:x['norm_cost']
                )[:self.size_beam]
            else:
                new_list = sorted(
                    new_list, key=lambda x:x['cost']
                )[:self.size_beam]
            self.beam_list = []
            while len(new_list) > 0:
                pop_item = new_list.pop(0)
                if pop_item['continue']:
                    self.beam_list.append(pop_item)
                else:
                    self.finish_list.append(pop_item)
            counter += 1
        #
        if len(self.finish_list) > 0:
            if self.normalize_mode:
                self.finish_list = sorted(
                    self.finish_list, key=lambda x:x['norm_cost']
                )
            else:
                self.finish_list = sorted(
                    self.finish_list, key=lambda x:x['cost']
                )
            while len(self.finish_list) > self.size_beam:
                self.finish_list.pop()
        while len(self.finish_list) < self.size_beam:
            self.finish_list.append(self.beam_list.pop(0))

    def count_response(self):
        print "# of finished responses is ", len(self.finish_list)

    def get_top_target(self):
        #print "getting top target as list of token_id ... "
        return self.finish_list[0]['list_idx_token'][1:-1]

    def get_top_att(self):
        #print
        return list(set(self.finish_list[0]['list_idx_att'][:-1]))

    def get_top_target_score(self):
        #print "getting top target score as a value ... "
        if self.normalize_mode:
            return self.finish_list[0]['norm_cost']
        else:
            return self.finish_list[0]['cost']
        #
    #
#
