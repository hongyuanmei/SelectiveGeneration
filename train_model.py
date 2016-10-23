# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Mar 18th 10:58:37 2016

train model

@author: hongyuan
"""

import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import sys
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams
import modules.utils as utils
import modules.models as models
import modules.optimizers as optimizers
import modules.controllers as controllers
import modules.data_processers as data_processers

import run_models
import datetime

dtype=theano.config.floatX


#
import argparse
__author__ = 'Hongyuan Mei'

def main():

    parser = argparse.ArgumentParser(
        description='Trainning model ... '
    )
    #
    '''
    modify here accordingly ...
    '''
    #
    parser.add_argument(
        '-m', '--Model', required=False,
        help='Model to be trained '
    )
    parser.add_argument(
        '-fd', '--FileData', required=False,
        help='Path of the dataset'
    )
    #
    parser.add_argument(
        '-d', '--DimModel', required=False,
        help='Dimension of LSTM model '
    )
    parser.add_argument(
        '-ns', '--NumSel', required=False,
        help='Number of pre-selection in expectation '
    )
    parser.add_argument(
        '-s', '--Seed', required=False,
        help='Seed of random state'
    )
    #
    parser.add_argument(
        '-fp', '--FilePretrain', required=False,
        help='File of pretrained model'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', required=False,
        help='Track period of training'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', required=False,
        help='Max epoch number of training'
    )
    parser.add_argument(
        '-sb', '--SizeBatch', required=False,
        help='Size of mini-batch'
    )
    parser.add_argument(
        '-op', '--Optimizer', required=False,
        help='Optimizer of training'
    )
    #
    #
    args = parser.parse_args()
    #
    if args.Model == None:
        args.Model = 'selgen'
    if args.FileData == None:
        args.FileData = os.path.abspath('./data')
    #
    if args.TrackPeriod == None:
        args.TrackPeriod = numpy.int32(100)
    else:
        args.TrackPeriod = numpy.int32(args.TrackPeriod)
    if args.MaxEpoch == None:
        args.MaxEpoch = numpy.int32(30)
    else:
        args.MaxEpoch = numpy.int32(args.MaxEpoch)
    if args.SizeBatch == None:
        args.SizeBatch = numpy.int32(50)
    else:
        args.SizeBatch = numpy.int32(args.SizeBatch)
    if args.Optimizer == None:
        args.Optimizer = 'adam'
    else:
        args.Optimizer = args.Optimizer
    #
    if args.DimModel == None:
        args.DimModel = numpy.int32(500)
    else:
        args.DimModel = numpy.int32(args.DimModel)
    if args.NumSel == None:
        args.NumSel = numpy.float32(5.0)
    else:
        args.NumSel = numpy.float32(args.NumSel)
    if args.Seed == None:
        args.Seed = numpy.int32(12345)
    else:
        args.Seed = numpy.int32(args.Seed)
    #
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    tag_model = '_PID='+str(id_process)+'_TIME='+time_current
    #
    path_track = './tracks/track' + tag_model + '/'
    file_log = os.path.abspath(
        path_track + 'log.txt'
    )
    path_save = path_track
    command_mkdir = 'mkdir -p ' + os.path.abspath(
        path_track
    )
    os.system(command_mkdir)
    #
    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    #
    print ("Model is : %s" % args.Model )
    print ("FileData is : %s" % args.FileData )
    print ("DimModel is : %s" % str(args.DimModel) )
    print ("NumSel is : %s" % str(args.NumSel) )
    print ("Seed is : %s" % str(args.Seed) )
    print ("FilePretrain is : %s" % args.FilePretrain)
    print ("TrackPeriod is : %s" % str(args.TrackPeriod) )
    print ("MaxEpoch is : %s" % str(args.MaxEpoch) )
    print ("SizeBatch is : %s" % str(args.SizeBatch) )
    print ("Optimizer is : %s" % args.Optimizer)
    #
    dict_args = {
        'PID': id_process,
        'TIME': time_current,
        'Model': args.Model,
        'FileData': args.FileData,
        'DimModel': args.DimModel,
        'Seed': args.Seed,
        'FilePretrain': args.FilePretrain,
        'TrackPeriod': args.TrackPeriod,
        'MaxEpoch': args.MaxEpoch,
        'SizeBatch': args.SizeBatch,
        'NumSel': args.NumSel,
        'Optimizer': args.Optimizer
    }
    #
    input_train = {
        'seed_random': args.Seed,
        'path_rawdata': args.FileData,
        'path_pre_train': args.FilePretrain,
        'track_period': args.TrackPeriod,
        'max_epoch': args.MaxEpoch,
        'size_batch': args.SizeBatch,
        'dim_model': args.DimModel,
        'optimizer': args.Optimizer,
        'save_file_path': path_save,
        'log_file': file_log,
        'num_sel': args.NumSel,
        'args': dict_args
    }
    #
    if args.Model == 'selgen':
        run_models.train_selgen(input_train)
    else:
        print "Model not cleaned up yet ... "
    #
    #

if __name__ == "__main__": main()
