# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8th 10:58:37 2016

data processers

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

class DataProcesser(object):
    '''
    this class process raw data into the model-friendly format
    and save them when neccessary
    '''
    def __init__(self, settings):
        #
        print "initialize the data processer ... "
        #
        self.path_data = os.path.abspath(
            settings['path_data']
        ) + '/' + 'data.pickle'
        self.path_stat = os.path.abspath(
            settings['path_data']
        ) + '/' + 'stat.pickle'
        self.path_align = os.path.abspath(
            settings['path_data']
        ) + '/' + 'aligns.pickle'
        #
        with open(self.path_data, 'rb') as f:
            self.data = pickle.load(f)
        #
        with open(self.path_stat, 'rb') as f:
            self.stat = pickle.load(f)
        #
        with open(self.path_align, 'rb') as f:
            self.aligns = pickle.load(f)
        #
        self.ind2word = self.stat['ind2word']
        self.word2ind = self.stat['word2ind']
        self.vocabsize = len(self.ind2word)
        self.vocabmat = numpy.identity(
            self.vocabsize, dtype=dtype
        )
        #
        self.dim_lang = self.vocabmat.shape[0]
        self.dim_info = self.data['train'][0]['info'].shape[1]
        self.num_info = self.data['train'][0]['info'].shape[0]
        self.size_batch = settings['size_batch']
        #
        self.lens = {
            'train': len(self.data['train']),
            'dev': len(self.data['dev']),
            'test': len(self.data['test'])
        }
        self.list_idx = {
            'train': range(self.lens['train']),
            'dev': range(self.lens['dev']),
            'test': range(self.lens['test'])
        }
        self.max_nums = {
            'train': int( self.lens['train']/self.size_batch ),
            'dev': int( self.lens['dev']/self.size_batch ),
            'test': int( self.lens['test']/self.size_batch )
        }
        #
    #
    #
    def get_refs(self, tag_split='dev'):
        list_refs = []
        for data_item in self.data[tag_split]:
            list_refs.append(
                data_item['text']
            )
        return list_refs
    #
    def get_golds(self, tag_split='dev'):
        return self.aligns[tag_split]

    #
    def translate(self, list_idx_token):
        list_token = [
            self.ind2word[idx] for idx in list_idx_token
        ]
        return ' '.join(list_token)
    #
    #

    def shuffle_train_data(self):
        #assert(tag=='train')
        print "shuffling training data idx ... "
        # we shuffle idx instead of the real data
        numpy.random.shuffle(self.list_idx['train'])

    def process_seq(self):
        #print "getting batch ... "
        #
        self.seq_info_numpy = numpy.zeros(
            (self.num_info, self.size_batch, self.dim_info),
            dtype = dtype
        )
        self.max_len = -1
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            data_item = self.data[self.tag_batch][idx_data]
            self.seq_info_numpy[:, idx_in_batch, :] = data_item[
                'info'
            ]
            list_tokens_this_data = data_item['text'].split()
            len_this_data = 0
            for token in list_tokens_this_data:
                if token in self.word2ind:
                    len_this_data += 1
            if len_this_data > self.max_len:
                self.max_len = len_this_data
        #
        self.seq_lang_numpy = numpy.zeros(
            (
                self.max_len+1, self.size_batch, self.dim_lang
            ), dtype = dtype
        )
        self.seq_target_numpy = numpy.zeros(
            (
                self.max_len+1, self.size_batch, self.dim_lang
            ), dtype = dtype
        )
        #
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            data_item = self.data[self.tag_batch][idx_data]
            list_tokens_this_data = data_item['text'].split()
            self.seq_lang_numpy[
                0, idx_in_batch, :
            ] = self.vocabmat[:, 0]
            idx_pos = 0
            for token in list_tokens_this_data:
                if token in self.word2ind:
                    self.seq_target_numpy[
                        idx_pos, idx_in_batch, :
                    ] = self.vocabmat[
                        :, self.word2ind[token]
                    ]
                    self.seq_lang_numpy[
                        idx_pos+1, idx_in_batch, :
                    ] = self.vocabmat[
                        :, self.word2ind[token]
                    ]
                    idx_pos += 1
            self.seq_target_numpy[
                idx_pos, idx_in_batch, :
            ] = self.vocabmat[:, 0]
        #
        #

    def process_data(
        self, tag_batch, idx_batch_current=0
    ):
        #
        #
        self.tag_batch = tag_batch
        self.list_idx_data = [
            idx for idx in self.list_idx[self.tag_batch][
                idx_batch_current * self.size_batch : (idx_batch_current + 1) * self.size_batch
            ]
        ]
        self.process_seq()

    def process_one_data(
        self, tag_batch, idx_data = 0
    ):
        self.tag_batch = tag_batch
        self.list_idx_data = [idx_data]
        #
        data_item = self.data[self.tag_batch][idx_data]
        list_tokens_this_data = data_item['text'].split()
        #
        self.max_len = -1
        len_this_data = 0
        for token in list_tokens_this_data:
            if token in self.word2ind:
                len_this_data += 1
        if len_this_data > self.max_len:
            self.max_len = len_this_data
        #
        self.seq_info_numpy = numpy.zeros(
            (self.num_info, self.dim_info),
            dtype = dtype
        )
        #
        self.seq_info_numpy[:,:] = data_item['info']
        #
        self.seq_lang_numpy = numpy.zeros(
            (self.max_len+1, self.dim_lang), dtype = dtype
        )
        self.seq_target_numpy = numpy.zeros(
            (self.max_len+1, self.dim_lang), dtype = dtype
        )
        #
        self.seq_lang_numpy[0, :] = self.vocabmat[:, 0]
        idx_pos = 0
        for token in list_tokens_this_data:
            if token in self.word2ind:
                self.seq_target_numpy[
                    idx_pos, :
                ] = self.vocabmat[
                    :, self.word2ind[token]
                ]
                self.seq_lang_numpy[
                    idx_pos+1, :
                ] = self.vocabmat[
                    :, self.word2ind[token]
                ]
                idx_pos += 1
        self.seq_target_numpy[idx_pos, :] = self.vocabmat[:, 0]
        #
        #


    def creat_log(self, log_dict):
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        print "creating training log file ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'w') as f:
            f.write('This the training log file. \n')
            f.write('It tracks some statistics in the training process ... \n')
            #
            f.write('Model specs are listed below : \n')
            for the_key in log_dict['args']:
                f.write(
                    the_key+' : '+str(log_dict['args'][the_key])
                )
                f.write('\n')
            #
            f.write('Before training, the compilation time is '+str(log_dict['compile_time'])+' sec ... \n')
            f.write('Things that need to be tracked : \n')
            for the_key in log_dict['tracked']:
                f.write(the_key+' ')
            f.write('\n\n')
        #
        #

    def continue_log(self, log_dict):
        print "continue tracking log ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'a') as f:
            for the_key in log_dict['tracked']:
                f.write(the_key+' is '+str(log_dict['tracked'][the_key])+' \n')
            if log_dict['max_dev_bleu'] < log_dict['tracked']['dev_bleu']:
                f.write('This is a new best model ! \n')
                log_dict['max_dev_loss'] = log_dict['tracked']['dev_bleu']
                #
                # update the tracked_best
                for the_key in log_dict['tracked']:
                    log_dict['tracked_best'][
                        the_key
                    ] = log_dict['tracked'][the_key]
                #
            f.write('\n')
        #
    #
    def finish_log(self, log_dict):
        print "finish tracking log ... "
        current_log_file = os.path.abspath(
            log_dict['log_file']
        )
        with open(current_log_file, 'a') as f:
            f.write('The best model info is shown below : \n')
            for the_key in log_dict['tracked_best']:
                f.write(
                    the_key+' is '+str(log_dict['tracked_best'][the_key])+' \n'
                )
                #
            f.write('\n')
    #
    #

    def track_log(self, log_dict):
        #print "recording training log ... "
        '''
        log dict keys : log_file, compile_time, things need to be tracked
        '''
        assert(log_dict['mode']=='create' or log_dict['mode']=='continue')
        if log_dict['mode'] == 'create':
            self.creat_log(log_dict)
        else:
            self.continue_log(log_dict)
