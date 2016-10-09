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
        '-m', '--Model', required=True,
        help='Model to be trained '
    )
    parser.add_argument(
        '-fd', '--FileData', required=True,
        help='Path of the dataset'
    )
    #
    parser.add_argument(
        '-d', '--DimLSTM', required=False,
        help='Dimension of LSTM model '
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
    if args.DimLSTM == None:
        args.DimLSTM = numpy.int32(64)
    else:
        args.DimLSTM = numpy.int32(args.DimLSTM)
    if args.Seed == None:
        args.Seed = numpy.int32(12345)
    else:
        args.Seed = numpy.int32(args.Seed)
    #
    if 'lstm' in args.FileData:
        tag_data = 'lstm'
    else:
        tag_data = 'hawkes'
    if args.FilePretrain == None:
        tag_pretrain = 'no'
    else:
        tag_pretrain = 'yes'
    #
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    #
    if 'lstm' in args.Model:
        tag_model = '_Model='+args.Model+'_Data='+tag_data+'_DimLSTM='+str(args.DimLSTM)+'_Seed='+str(args.Seed)+'_Pretrain='+tag_pretrain+'_SizeBatch='+str(args.SizeBatch)+'_Opt='+args.Optimizer+'_PID='+str(id_process)+'_TIME='+time_current
    else:
        tag_model = '_Model='+args.Model+'_Data='+tag_data+'_Seed='+str(args.Seed)+'_Pretrain='+tag_pretrain+'_SizeBatch='+str(args.SizeBatch)+'_Opt='+args.Optimizer+'_PID='+str(id_process)+'_TIME='+time_current
    #
    file_log = os.path.abspath(
        './logs/log' + tag_model + '.txt'
    )
    path_save = os.path.abspath(
        './models/models' + tag_model + '/'
    )
    #
    ## show values ##
    print ("PID is : %s" % str(id_process) )
    print ("TIME is : %s" % time_current )
    print ("Model is : %s" % args.Model )
    print ("FileData is : %s" % args.FileData )
    if 'lstm' in args.Model:
        print ("DimLSTM is : %s" % str(args.DimLSTM) )
    print ("Seed is : %s" % str(args.Seed) )
    print ("FilePretrain is : %s" % args.FilePretrain)
    print ("TrackPeriod is : %s" % str(args.TrackPeriod) )
    print ("MaxEpoch is : %s" % str(args.MaxEpoch) )
    print ("SizeBatch is : %s" % str(args.SizeBatch) )
    print ("Optimizer is : %s" % args.Optimizer)
    #
    #
    input_train = {
        'seed_random': args.Seed,
        'path_rawdata': args.FileData,
        'path_pre_train': args.FilePretrain,
        'track_period': args.TrackPeriod,
        'max_epoch': args.MaxEpoch,
        'size_batch': args.SizeBatch,
        'dim_model': args.DimLSTM,
        'optimizer': args.Optimizer,
        'save_file_path': path_save,
        'log_file': file_log
    }
    #
    if args.Model == 'hawkes':
        run_models.train_hawkes_ctsm(input_train)
    else:
        print "Model not implemented yet !!! "
    #

if __name__ == "__main__": main()
