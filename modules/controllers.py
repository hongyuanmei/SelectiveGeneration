# -*- coding: utf-8 -*-
"""

Controllers for diffferent models

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
import models
import optimizers

dtype = theano.config.floatX

class ControlSelGen(object):
    #
    def __init__(self, settings):
        #
        print "building controller ... "
        self.seq_info = tensor.tensor3(
            dtype=dtype, name='seq_info'
        )
        self.seq_lang = tensor.tensor3(
            dtype=dtype, name='seq_lang'
        )
        self.seq_target = tensor.tensor3(
            dtype=dtype, name='seq_target'
        )
        #
        self.model = models.SelGen(settings)
        #
        self.model.compute_loss(
            self.seq_info, self.seq_lang, self.seq_target
        )
        #
        assert(
            settings['optimizer'] == 'adam' or settings['optimizer'] == 'sgd'
        )
        if settings['optimizer'] == 'adam':
            self.adam_optimizer = optimizers.Adam(adam_params=None)
        elif settings['optimizer'] == 'sgd':
            self.adam_optimizer = optimizers.SGD(adam_params=None)
        else:
            print "Choose a optimizer ! "
        #
        self.adam_optimizer.compute_updates(
            self.model.params, self.model.grad_params
        )
        #
        print "compiling training function ... "
        self.model_learn = theano.function(
            inputs = [
                self.seq_info, self.seq_lang, self.seq_target
            ],
            outputs = self.model.cost,
            updates = self.adam_optimizer.updates
        )
        print "compiling dev function ... "
        self.model_dev = theano.function(
            inputs = [
                self.seq_info, self.seq_lang, self.seq_target
            ],
            outputs = self.model.cost
        )
        self.save_model = self.model.save_model
        self.get_model = self.model.get_model
    #
#
