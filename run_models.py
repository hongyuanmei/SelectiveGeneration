# -*- coding: utf-8 -*-
"""
Created on Mar 18th 10:58:37 2016

run models, including training and validating

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
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.controllers as controllers
import modules.data_processers as data_processers
import modules.beam_search as searchers
import modules.evals as evaluations

dtype=theano.config.floatX

#TODO: function to train seq2seq models
def train_selgen(input_train):
    '''
    this function is called to train Sel Gen model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(input_train['seed_random'])
    #
    save_file_path = os.path.abspath(
        input_train['save_file_path']
    )
    command_mkdir = 'mkdir -p ' + save_file_path
    os.system(command_mkdir)
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': save_file_path,
        'mode': 'create', 'compile_time': None,
        'min_dev_loss': 1e6,
        'max_dev_bleu': -1.0,
        #
        'args': input_train['args'],
        #
        'tracked_best': {},
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_loss': None,
            #'dev_loss': None,
            'dev_bleu': None,
            'dev_F1': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_data': input_train['path_rawdata'],
            'size_batch': input_train['size_batch']
        }
    )
    #
    #TODO: build the model
    print "building model ... "

    compile_start = time.time()

    model_settings = {
        'dim_model': input_train['dim_model'],
        'dim_lang': data_process.dim_lang,
        'dim_info': data_process.dim_info,
        'num_sel': input_train['num_sel'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train']
    }

    control = controllers.ControlSelGen(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start

    #'''
    print "building Bleu Scorer ... "
    settings_bs = {
        'size_beam': 1,
        'path_model': None,
        'normalize_mode': True
    }
    beam_search = searchers.BeamSearchSelGen(settings_bs)
    #
    #settings_bleu = {
    #    'path_program': None,
    #    'path_bleu': input_train['path_bleu']
    #}
    bleu_scorer = evaluations.BleuScoreNLTK()
    bleu_scorer.set_refs(
        data_process.get_refs(tag_split='dev')
    )
    #
    f1_computer = evaluations.F1Compute()
    f1_computer.set_golds(
        data_process.get_golds(tag_split='dev')
    )
    #

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        err = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                'train', step_train
            )
            #
            #print "training ... "
            cost_numpy = control.model_learn(
                data_process.seq_info_numpy,
                data_process.seq_lang_numpy,
                data_process.seq_target_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            err += cost_numpy
            #
            log_dict['tracked']['train_loss'] = round(err/(step_train+1), 4)
            train_end = time.time()
            log_dict['tracked']['train_time'] = round(
                (
                    train_end - train_start
                )*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: set model to bleu score
                beam_search.set_model(
                    control.get_model()
                )
                #
                bleu_scorer.reset_gens()
                f1_computer.reset_aligns()
                #TODO: get the dev loss values
                sum_costs = 0.0
                for step_dev in range(data_process.lens['dev']):
                    #
                    data_process.process_one_data(
                        'dev', step_dev
                    )
                    #
                    #print "validating ... "
                    #
                    beam_search.refresh_state()
                    beam_search.set_encoder(
                        data_process.seq_info_numpy
                    )
                    beam_search.init_beam()
                    beam_search.search_func()
                    #
                    f1_computer.add_align(
                        beam_search.get_top_att()
                    )
                    #
                    gen_step_dev = data_process.translate(
                        beam_search.get_top_target()
                    )
                    bleu_scorer.add_gen(gen_step_dev)
                    #
                    if step_dev % 100 == 99:
                        print "in dev, the step is out of ", step_dev, data_process.lens['dev']
                #
                bleu_score = bleu_scorer.evaluate()
                f1_score = f1_computer.evaluate()
                #
                log_dict['tracked']['dev_bleu'] = round(
                    bleu_score, 2
                )
                log_dict['tracked']['dev_F1'] = round(
                    f1_score, 2
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_bleu'] > log_dict['max_dev_bleu']:
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+'model.pkl'
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    print "finish training"
    #
#
def train_selgen_eval_angeli(input_train):
    '''
    this function is called to train Sel Gen model
    '''
    #TODO: pre-settings like random states
    numpy.random.seed(input_train['seed_random'])
    #
    save_file_path = os.path.abspath(
        input_train['save_file_path']
    )
    command_mkdir = 'mkdir -p ' + save_file_path
    os.system(command_mkdir)
    #
    log_dict = {
        'log_file': input_train['log_file'],
        'save_file_path': save_file_path,
        'mode': 'create', 'compile_time': None,
        'min_dev_loss': 1e6,
        'max_dev_bleu': -1.0,
        #
        'args': input_train['args'],
        #
        'tracked_best': {},
        #
        'iteration': 0,
        'track_period': input_train['track_period'],
        'max_epoch': input_train['max_epoch'],
        'size_batch': input_train['size_batch'],
        'tracked': {
            'track_cnt': None,
            'train_loss': None,
            #'dev_loss': None,
            'dev_bleu': None,
            'dev_F1': None,
            #
            'train_time': None, 'dev_time': None
        }
    }

    #TODO: get the data and process the data
    print "reading and processing data ... "

    data_process = data_processers.DataProcesser(
        {
            'path_data': input_train['path_rawdata'],
            'size_batch': input_train['size_batch']
        }
    )
    #
    #TODO: build the model
    print "building model ... "

    #'''
    print "building Bleu Scorer ... "
    settings_bs = {
        'size_beam': 1,
        'path_model': None,
        'normalize_mode': True
    }
    beam_search = searchers.BeamSearchSelGen(settings_bs)
    #
    #settings_bleu = {
    #    'path_program': None,
    #    'path_bleu': input_train['path_bleu']
    #}
    bleu_scorer = evaluations.BleuScoreAngeli(
        {
            'path_jvm': input_train['path_jvm'],
            'path_jar': input_train['path_jar'],
            'max_diff': input_train['max_diff']
        }
    )
    bleu_scorer.set_refs(
        data_process.get_refs(tag_split='dev')
    )
    #
    f1_computer = evaluations.F1Compute()
    f1_computer.set_golds(
        data_process.get_golds(tag_split='dev')
    )
    #

    compile_start = time.time()

    model_settings = {
        'dim_model': input_train['dim_model'],
        'dim_lang': data_process.dim_lang,
        'dim_info': data_process.dim_info,
        'num_sel': input_train['num_sel'],
        'size_batch': input_train['size_batch'],
        'optimizer': input_train['optimizer'],
        'path_pre_train': input_train['path_pre_train']
    }

    control = controllers.ControlSelGen(
        model_settings
    )

    compile_end = time.time()
    compile_time = compile_end - compile_start
    #

    print "model finished, comilation time is ", round(compile_time, 0)

    #TODO: start training, define the training functions
    print "building training log ... "
    log_dict['compile_time'] = round(compile_time, 0)
    data_process.track_log(log_dict)
    log_dict['mode'] = 'continue'

    for epi in range(log_dict['max_epoch']):
        #
        print "training epoch ", epi
        #
        err = 0.0
        #TODO: shuffle the training data and train this epoch
        data_process.shuffle_train_data()
        #
        for step_train in range(data_process.max_nums['train'] ):
            #
            train_start = time.time()
            #print "the step is ", step
            #
            data_process.process_data(
                'train', step_train
            )
            #
            #print "training ... "
            cost_numpy = control.model_learn(
                data_process.seq_info_numpy,
                data_process.seq_lang_numpy,
                data_process.seq_target_numpy
            )
            #
            #
            log_dict['iteration'] += 1
            err += cost_numpy
            #
            log_dict['tracked']['train_loss'] = round(err/(step_train+1), 4)
            train_end = time.time()
            log_dict['tracked']['train_time'] = round(
                (
                    train_end - train_start
                )*log_dict['track_period'], 0
            )
            #
            if step_train % 10 == 9:
                print "in training, the step is out of ", step_train, data_process.max_nums['train']
            ########
            # Now we track the performance and save the model for every # batches, so that we do not miss the convergence within the epoch -- one epoch is too large sometimes
            ########
            if log_dict['iteration'] % log_dict['track_period'] == 0:
                #TODO: go through the dev data and calculate the dev metrics
                print "Now we start validating after batches ", log_dict['track_period']
                dev_start = time.time()
                #
                #TODO: set model to bleu score
                beam_search.set_model(
                    control.get_model()
                )
                #
                bleu_scorer.reset_gens()
                f1_computer.reset_aligns()
                #TODO: get the dev loss values
                sum_costs = 0.0
                for step_dev in range(data_process.lens['dev']):
                    #
                    data_process.process_one_data(
                        'dev', step_dev
                    )
                    #
                    #print "validating ... "
                    #
                    beam_search.refresh_state()
                    beam_search.set_encoder(
                        data_process.seq_info_numpy
                    )
                    beam_search.init_beam()
                    beam_search.search_func()
                    #
                    f1_computer.add_align(
                        beam_search.get_top_att()
                    )
                    #
                    gen_step_dev = data_process.translate(
                        beam_search.get_top_target()
                    )
                    bleu_scorer.add_gen(gen_step_dev)
                    #
                    if step_dev % 100 == 99:
                        print "in dev, the step is out of ", step_dev, data_process.lens['dev']
                #
                bleu_score = bleu_scorer.evaluate()
                f1_score = f1_computer.evaluate()
                #
                log_dict['tracked']['dev_bleu'] = round(
                    bleu_score, 2
                )
                log_dict['tracked']['dev_F1'] = round(
                    f1_score, 2
                )
                #
                dev_end = time.time()
                log_dict['tracked']['dev_time'] = round( dev_end - dev_start, 0 )
                #
                log_dict['tracked']['track_cnt'] = int(
                    log_dict['iteration']/log_dict['track_period']
                )
                #
                #
                if log_dict['tracked']['dev_bleu'] > log_dict['max_dev_bleu']:
                    save_file = os.path.abspath(
                        log_dict['save_file_path']
                    ) + '/'+'model.pkl'
                    control.save_model(save_file)
                #
                data_process.track_log(log_dict)
            ########
    data_process.finish_log(log_dict)
    bleu_scorer.shutdownJVM()
    print "finish training"
    #
#
