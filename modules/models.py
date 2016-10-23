# -*- coding: utf-8 -*-
"""

Here are the models
Sel Gen models in NAACL 2016 paper

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

class SelGen(object):
    #
    def __init__(self, settings):
        print "initializing Sel Gen model ... "
        self.size_batch = settings['size_batch']
        self.num_sel = settings['num_sel']
        #
        if settings['path_pre_train'] == None:
            self.dim_model = settings['dim_model']
            self.dim_lang = settings['dim_lang']
            self.dim_info = settings['dim_info']
            # initialize variables
            self.Emb_enc_forward = theano.shared(
                utils.sample_weights(
                    self.dim_info, self.dim_model
                ), name='Emb_enc_forward'
            )
            self.Emb_enc_backward = theano.shared(
                utils.sample_weights(
                    self.dim_info, self.dim_model
                ), name='Emb_enc_backward'
            )
            self.W_enc_forward = theano.shared(
                utils.sample_weights(
                    2*self.dim_model, 4*self.dim_model
                ), name='W_enc_forward'
            )
            self.W_enc_backward = theano.shared(
                utils.sample_weights(
                    2*self.dim_model, 4*self.dim_model
                ), name='W_enc_backward'
            )
            self.b_enc_forward = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_enc_forward'
            )
            self.b_enc_backward = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_enc_backward'
            )
            #
            self.W_pre_att = theano.shared(
                utils.sample_weights(
                    self.dim_info+2*self.dim_model, self.dim_model
                ), name='W_pre_att'
            )
            self.b_pre_att = theano.shared(
                numpy.zeros(
                    (self.dim_model,), dtype=dtype
                ), name='b_pre_att'
            )
            #
            self.W_att = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_model
                ), name='W_att'
            )
            self.U_att = theano.shared(
                utils.sample_weights(
                    self.dim_info+2*self.dim_model, self.dim_model
                ), name='U_att'
            )
            self.b_att = theano.shared(
                numpy.zeros(
                    (self.dim_model,), dtype=dtype
                ), name='b_att'
            )
            #
            self.Emb_dec = theano.shared(
                utils.sample_weights(
                    self.dim_lang, self.dim_model
                ), name='Emb_dec'
            )
            self.W_dec = theano.shared(
                utils.sample_weights(
                    self.dim_info+4*self.dim_model, 4*self.dim_model
                ), name='W_dec'
            )
            self.b_dec = theano.shared(
                numpy.zeros(
                    (4*self.dim_model,), dtype=dtype
                ), name='b_dec'
            )
            self.L_0 = theano.shared(
                utils.sample_weights(
                    self.dim_model, self.dim_lang
                ), name='L_0'
            )
            self.L = theano.shared(
                utils.sample_weights(
                    self.dim_info+3*self.dim_model, self.dim_model
                ), name='L'
            )
            #
        else:
            #
            path_pre_train = os.path.abspath(
                settings['path_pre_train']
            )
            with open(path_pre_train, 'rb') as f:
                model_pre_train = pickle.load(f)
            #
            self.Emb_enc_forward = theano.shared(
                model_pre_train['Emb_enc_forward']
            )
            self.Emb_enc_backward = theano.shared(
                model_pre_train['Emb_enc_backward']
            )
            self.W_enc_forward = theano.shared(
                model_pre_train['W_enc_forward']
            )
            self.W_enc_backward = theano.shared(
                model_pre_train['W_enc_backward']
            )
            self.b_enc_forward = theano.shared(
                model_pre_train['b_enc_forward']
            )
            self.b_enc_backward = theano.shared(
                model_pre_train['b_enc_backward']
            )
            #
            self.W_pre_att = theano.shared(
                model_pre_train['W_pre_att']
            )
            self.b_pre_att = theano.shared(
                model_pre_train['b_pre_att']
            )
            #
            self.W_att = theano.shared(
                model_pre_train['W_att']
            )
            self.U_att = theano.shared(
                model_pre_train['U_att']
            )
            self.b_att = theano.shared(
                model_pre_train['b_att']
            )
            #
            self.Emb_dec = theano.shared(
                model_pre_train['Emb_dec']
            )
            self.W_dec = theano.shared(
                model_pre_train['W_dec']
            )
            self.b_dec = theano.shared(
                model_pre_train['b_dec']
            )
            self.L_0 = theano.shared(
                model_pre_train['L_0']
            )
            self.L = theano.shared(
                model_pre_train['L']
            )
            #
            self.dim_model = self.Emb_enc_forward.shape[1]
            self.dim_lang = self.Emb_dec.shape[0]
            self.dim_info = self.Emb_enc_forward.shape[0]
            #
        #
        self.h_0_mat = tensor.zeros(
            (self.size_batch, self.dim_model), dtype=dtype
        )
        self.c_0_mat = tensor.zeros(
            (self.size_batch, self.dim_model), dtype=dtype
        )
        #
        self.params = [
            self.Emb_enc_forward, self.Emb_enc_backward,
            self.W_enc_forward, self.W_enc_backward,
            self.b_enc_forward, self.b_enc_backward,
            self.W_pre_att, self.b_pre_att,
            self.W_att, self.U_att, self.b_att,
            self.Emb_dec, self.W_dec, self.b_dec,
            self.L_0, self.L
        ]
        self.grad_params = None
        self.cost = None
        #
    #
    def encoder(
        self, info_forward, info_backward,
        h_tm1_forward, c_tm1_forward,
        h_tm1_backward, c_tm1_backward
    ):
        # infomat is a matrix, having # batch * D
        #
        xt_forward = theano.dot(
            info_forward, self.Emb_enc_forward
        )
        xt_backward = theano.dot(
            info_backward, self.Emb_enc_backward
        )
        #
        pretran_forward = tensor.concatenate(
            [xt_forward, h_tm1_forward], axis=1
        )
        pretran_backward = tensor.concatenate(
            [xt_backward, h_tm1_backward], axis=1
        )
        #
        posttran_forward = theano.dot(
            pretran_forward, self.W_enc_forward
        ) + self.b_enc_forward
        posttran_backward = theano.dot(
            pretran_backward, self.W_enc_backward
        ) + self.b_enc_backward
        #
        i_t_forward = tensor.nnet.sigmoid(
            posttran_forward[:, 0:self.dim_model]
        )
        f_t_forward = tensor.nnet.sigmoid(
            posttran_forward[:, self.dim_model:2*self.dim_model]
        )
        g_t_forward = tensor.tanh(
            posttran_forward[:, 2*self.dim_model:3*self.dim_model]
        )
        o_t_forward = tensor.nnet.sigmoid(
            posttran_forward[:, 3*self.dim_model:]
        )
        c_t_forward = f_t_forward * c_tm1_forward + i_t_forward * g_t_forward
        #
        h_t_forward = o_t_forward * tensor.tanh(c_t_forward)
        #
        #
        i_t_backward = tensor.nnet.sigmoid(
            posttran_backward[:, 0:self.dim_model]
        )
        f_t_backward = tensor.nnet.sigmoid(
            posttran_backward[:, self.dim_model:2*self.dim_model]
        )
        g_t_backward = tensor.tanh(
            posttran_backward[:, 2*self.dim_model:3*self.dim_model]
        )
        o_t_backward = tensor.nnet.sigmoid(
            posttran_backward[:, 3*self.dim_model:]
        )
        c_t_backward = f_t_backward * c_tm1_backward + i_t_backward * g_t_backward
        #
        h_t_backward = o_t_backward * tensor.tanh(c_t_backward)
        #
        return h_t_forward, c_t_forward, h_t_backward, c_t_backward

    #
    def decoder(
        self, lang, h_tm1_dec, c_tm1_dec
    ):
        x_t_lang = theano.dot(
            lang, self.Emb_dec
        )
        #
        beta1 = tensor.tensordot(
            self.scope_att, self.U_att,(2,0)
        )
        beta2 = theano.dot(
            h_tm1_dec, self.W_att
        )
        beta3 = tensor.tanh( beta1 + beta2 )
        beta4 = tensor.tensordot(
            beta3, self.b_att, (2,0)
        ) #  |->  # lines * # batch
        pre_alpha = tensor.nnet.softmax(
            tensor.transpose( beta4, axes=(1,0) )
        )
        #
        pre_alpha *= self.weights_pre_sel # Alpha
        alpha = pre_alpha / pre_alpha.sum(axis=1, keepdims=True)
        #
        z_t = tensor.sum(
            alpha[:,:,None] * tensor.transpose(
                self.scope_att, axes=(1,0,2)
            ),
            axis=1
        )
        #
        pre_tran = tensor.concatenate(
            [x_t_lang, h_tm1_dec, z_t], axis=1
        )
        post_tran = theano.dot(pre_tran, self.W_dec) + self.b_dec
        #
        i_t = tensor.nnet.sigmoid(
            post_tran[:, :self.dim_model]
        )
        f_t = tensor.nnet.sigmoid(
            post_tran[:, self.dim_model:2*self.dim_model]
        )
        g_t = tensor.tanh(
            post_tran[:, 2*self.dim_model:3*self.dim_model]
        )
        o_t = tensor.nnet.sigmoid(
            post_tran[:, 3*self.dim_model:]
        )
        c_t_dec = f_t * c_tm1_dec + i_t * g_t
        h_t_dec = o_t * tensor.tanh(c_t_dec)
        #
        pre_y = tensor.concatenate(
            [h_t_dec, z_t], axis=1
        )
        y_t_0 = theano.dot(
            (
                x_t_lang + theano.dot(
                    pre_y, self.L
                )
            ),
            self.L_0
        )
        y_t = tensor.nnet.softmax(y_t_0)
        log_y_t = tensor.log(
            y_t + numpy.float32(1e-8)
        )
        return h_t_dec, c_t_dec, y_t, log_y_t
        #
    #
    #
    def compute_loss(
        self, seq_info, seq_lang, seq_target
    ):
        print "computing loss function of model ... "
        seq_info_forward = seq_info
        seq_info_backward = seq_info[::-1,:,:]
        [h_forward, c_forward, h_backward, c_backward], _ = theano.scan(
            fn = self.encoder,
            sequences = [
                dict(input=seq_info_forward, taps=[0]),
                dict(input=seq_info_backward, taps=[0])
            ],
            outputs_info = [
                dict(initial=self.h_0_mat, taps=[-1]),
                dict(initial=self.c_0_mat, taps=[-1]),
                dict(initial=self.h_0_mat, taps=[-1]),
                dict(initial=self.c_0_mat, taps=[-1])
            ],
            non_sequences = None
        )
        #
        self.scope_att = tensor.concatenate(
            [
                seq_info_forward,
                h_forward, h_backward[::-1, :, :]
            ], axis=2
        )
        #
        pre_Alpha = tensor.tanh(
            tensor.tensordot(
                self.scope_att, self.W_pre_att, (2,0)
            )
        )
        self.weights_pre_sel = tensor.nnet.sigmoid(
            tensor.transpose(
                tensor.tensordot(
                    pre_Alpha, self.b_pre_att, (2,0)
                ), axes = (1,0)
            )
        )
        # size_batch * num_info
        #
        [h_dec, c_dec, y_dec, log_y_dec], _ = theano.scan(
            fn = self.decoder,
            sequences = dict(input=seq_lang, taps=[0]),
            outputs_info = [
                dict(initial=self.h_0_mat, taps=[-1]),
                dict(initial=self.c_0_mat, taps=[-1]),
                None, None
            ],
            non_sequences = None
        )
        #
        cost_term = -tensor.mean(
            tensor.sum(
                seq_target * log_y_dec, [2,0]
            )
        )
        #
        reg_term_1 = (
            tensor.mean(
                tensor.sum(
                    self.weights_pre_sel, axis=1
                )
            ) - self.num_sel
        )**2
        reg_term_2 = numpy.float32(1.0) - tensor.mean(
            tensor.max(self.weights_pre_sel, axis=1)
        )
        #
        self.cost = cost_term + reg_term_1 + reg_term_2
        self.grad_params = tensor.grad(
            self.cost, self.params
        )
        #
    #
    def get_model(self):
        print "getting model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_model'] = self.dim_model
        model_dict['dim_lang'] = self.dim_lang
        model_dict['dim_info'] = self.dim_info
        model_dict['num_sel'] = self.num_sel
        return model_dict
    #
    def save_model(self, file_save):
        print "saving model ... "
        model_dict = {}
        for param in self.params:
            model_dict[param.name] = numpy.copy(
                param.get_value()
            )
        model_dict['dim_model'] = self.dim_model
        model_dict['dim_lang'] = self.dim_lang
        model_dict['dim_info'] = self.dim_info
        model_dict['num_sel'] = self.num_sel
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
#

'''
class BeamSearch(object):
    #
    def __init__(self, settings):
        pass
    #
    def set_model(self, model_dict):
        pass

    def
    #
    #
'''
