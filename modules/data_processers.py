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
            settings['path_data'] + 'data.pickle'
        )
        self.path_stat = os.path.abspath(
            settings['path_data'] + 'stat.pickle'
        )
        #
        with open(self.path_data, 'rb') as f:
            self.data = pickle.load(f)
        #
        with open(self.path_stat, 'rb') as f:
            self.stat = pickle.load(f)
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

    def shuffle_train_data(self):
        #assert(tag=='train')
        print "shuffling training data idx ... "
        # we shuffle idx instead of the real data
        numpy.random.shuffle(self.list_idx['train'])

    def process_seq(self):
        print "getting batch ... "
        max_len = 0
        temp_list_seq_type_event = []
        temp_list_seq_time_since_last = []
        for idx_in_batch, idx_data in enumerate(self.list_idx_data):
            temp_seq_type_event = []
            temp_seq_time_since_last = []
            for item in self.data[self.tag_batch][idx_data]:
                temp_seq_type_event.append(
                    item['type_event']
                )
                temp_seq_time_since_last.append(
                    item['time_since_last_event']
                )
            len_seq = len(temp_seq_type_event)
            if max_len < len_seq:
                max_len = len_seq
            temp_list_seq_type_event.append(
                temp_seq_type_event
            )
            temp_list_seq_time_since_last.append(
                temp_seq_time_since_last
            )
        #
        self.seq_type_event_numpy = numpy.zeros(
            (max_len, self.size_batch), dtype=numpy.int32
        )
        self.seq_time_since_last_numpy = numpy.zeros(
            (max_len, self.size_batch), dtype=dtype
        )
        self.seq_mask_numpy = numpy.zeros(
            (max_len, self.size_batch), dtype=dtype
        )
        #
        for idx_in_batch, (seq_type_event, seq_time_since_last) in enumerate(zip(temp_list_seq_type_event, temp_list_seq_time_since_last)):
            for idx_pos, (type_event, time_since_start) in enumerate(zip(seq_type_event, seq_time_since_last)):
                self.seq_type_event_numpy[idx_pos, idx_in_batch] = type_event
                self.seq_time_since_last_numpy[idx_pos, idx_in_batch] = time_since_start
                self.seq_mask_numpy[idx_pos, idx_in_batch] = numpy.float32(1.0)
        #
        #

    def process_data(
        self, tag_batch, idx_batch_current=0
    ):
        #
        '''
        fill in here and correct right parts ... 
        '''
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
        pass
        '''
        fill in here
        '''

    def process_list_seq(
        self, tag_split
    ):
        self.list_seq_type_event = []
        self.list_seq_time_since_last = []
        for data_item in self.data[tag_split]:
            seq_type_event = []
            seq_time_since_last = []
            for event_item in data_item:
                seq_type_event.append(
                    event_item['type_event']
                )
                seq_time_since_last.append(
                    event_item['time_since_last_event']
                )
            self.list_seq_type_event.append(
                seq_type_event
            )
            self.list_seq_time_since_last.append(
                seq_time_since_last
            )

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
            f.write('It tracks some statistics in the training process ... ')
            f.write('Before training, the compilation time is '+str(log_dict['compile_time'])+' sec ... \n')
            f.write('Things that need to be tracked : \n')
            for the_key in log_dict['tracked']:
                f.write(the_key+' ')
            f.write('\n\n')
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
            f.write('\n')

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
