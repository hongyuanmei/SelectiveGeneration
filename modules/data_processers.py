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
        #self.path_data = os.path.abspath(
        #    settings['path_data']
        #) + '/' + 'data_alter.pickle'
        # alter matches data and feature processing ...
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
        self.dim_info = self.data['dim_info']
        self.num_info = self.data['num_info']
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

    #
    #TODO: functions for processing single feature
    def gettype(self, infoline):
        typevec = numpy.zeros((12,),dtype=dtype)
        if infoline['type'] == 'temperature':
            typevec[0] = 1.
        elif infoline['type'] == 'windChill':
            typevec[1] = 1.
        elif infoline['type'] == 'windSpeed':
            typevec[2] = 1.
        elif infoline['type'] == 'windDir':
            typevec[3] = 1.
        elif infoline['type'] == 'gust':
            typevec[4] = 1.
        elif infoline['type'] == 'skyCover':
            typevec[5] = 1.
        elif infoline['type'] == 'precipPotential':
            typevec[6] = 1.
        elif infoline['type'] == 'thunderChance':
            typevec[7] = 1.
        elif infoline['type'] == 'rainChance':
            typevec[8] = 1.
        elif infoline['type'] == 'snowChance':
            typevec[9] = 1.
        elif infoline['type'] == 'freezingRainChance':
            typevec[10] = 1.
        elif infoline['type'] == 'sleetChance':
            typevec[11] = 1.
        else:
            print "wrong in type"
        return typevec
    #
    def getlabel(self, infoline):
        labelvec = numpy.zeros((5,),dtype=dtype)
        if infoline['label'] == 'Tonight':
            labelvec[0] = 1.
        elif infoline['label'] == 'Sunday':
            labelvec[1] = 1.
        elif infoline['label'] == 'Monday':
            labelvec[2] = 1.
        elif infoline['label'] == 'Tuesday':
            labelvec[3] = 1.
        elif infoline['label'] == 'Wednesday':
            labelvec[4] = 1.
        else:
            print "wrong in label"
        return labelvec
    #
    def gettime(self, infoline):
        timevec = numpy.zeros((10,),dtype=dtype)
        if infoline['time'] == '6-9':
            timevec[0] = 1.
        elif infoline['time'] == '6-13':
            timevec[1] = 1.
        elif infoline['time'] == '6-21':
            timevec[2] = 1.
        elif infoline['time'] == '9-21':
            timevec[3] = 1.
        elif infoline['time'] == '13-21':
            timevec[4] = 1.
        elif infoline['time'] == '17-21':
            timevec[5] = 1.
        elif infoline['time'] == '17-26':
            timevec[6] = 1.
        elif infoline['time'] == '17-30':
            timevec[7] = 1.
        elif infoline['time'] == '21-30':
            timevec[8] = 1.
        elif infoline['time'] == '26-30':
            timevec[9] = 1.
        else:
            print "wrong in time"
        return timevec
    #
    def getnum(self, numstr):
        # this is used ot get max/mean/min vector
        # 2^8 is enought for -100 to 100
        if numstr == '':
            numvec = numpy.zeros((9,),dtype=dtype)
            numvec[-1] = 1.
        else:
            deci = int(numstr)
            numvec = numpy.zeros((9,),dtype=dtype)
            if deci > 0:
                numbin = bin(deci)
                binpart = int(numbin[2:])
                ind = 2
                while binpart:
                    numvec[-ind] = numpy.float32(
                        (binpart%10)
                    )
                    binpart /= 10
                    ind += 1
            elif deci < 0:
                numvec[0] = 1.
                numbin = bin(deci)
                binpart = int(numbin[3:])
                ind = 2
                while binpart:
                    numvec[-ind] = numpy.float32(
                        (binpart%10)
                    )
                    binpart /= 10
                    ind += 1
            elif deci == 0:
                pass
            else:
                print "wrong in num"
        return numvec
    #
    def getnums(self, minstr, meanstr, maxstr):
        # used to get max+mean+min vectors
        minvec = self.getnum(minstr)
        meanvec = self.getnum(meanstr)
        maxvec = self.getnum(maxstr)
        numsvec = numpy.concatenate(
            (minvec,meanvec, maxvec), axis=0
        )
        return numsvec
    #
    def gettemp(self, infoline):
        if infoline['type'] == 'temperature':
            tempvec = self.getnums(
                infoline['min'],infoline['mean'],infoline['max']
            )
        else:
            tempvec = self.getnums('','','')
        return tempvec
    #
    def getwindchill(self, infoline):
        if infoline['type'] == 'windChill':
            chillvec = self.getnums(
                infoline['min'],infoline['mean'],infoline['max']
            )
        else:
            chillvec = self.getnums('','','')
        return chillvec
    #
    def getwindspeed(self, infoline):
        if infoline['type'] == 'windSpeed':
            windspeedvec = self.getnums(
                infoline['min'],infoline['mean'],infoline['max']
            )
        else:
            windspeedvec = self.getnums('','','')
        return windspeedvec
    #
    def getbucket20(self, infoline):
        bucket20vec = numpy.zeros((3,),dtype=dtype)
        if infoline['mode_bucket_0_20_2'] == '0-10':
            bucket20vec[0] = 1.
        elif infoline['mode_bucket_0_20_2'] == '10-20':
            bucket20vec[1] = 1.
        elif infoline['mode_bucket_0_20_2'] == '':
            bucket20vec[2] = 1.
        else:
            print "wrong in bucket20"
        return bucket20vec
    #
    def getdirmode(self, infoline):#
        dirvec = numpy.zeros((18, ), dtype=dtype)
        if infoline['type'] == 'windDir':
            if infoline['mode'] == '':
                dirvec[0] = 1.
            elif infoline['mode'] == 'S':
                dirvec[1] = 1.
            elif infoline['mode'] == 'SW':
                dirvec[2] = 1.
            elif infoline['mode'] == 'SSE':
                dirvec[3] = 1.
            elif infoline['mode'] == 'WSW':
                dirvec[4] = 1.
            elif infoline['mode'] == 'ESE':
                dirvec[5] = 1.
            elif infoline['mode'] == 'E':
                dirvec[6] = 1.
            elif infoline['mode'] == 'W':
                dirvec[7] = 1.
            elif infoline['mode'] == 'SE':
                dirvec[8] = 1.
            elif infoline['mode'] == 'NE':
                dirvec[9] = 1.
            elif infoline['mode'] == 'SSW':
                dirvec[10] = 1.
            elif infoline['mode'] == 'NNE':
                dirvec[11] = 1.
            elif infoline['mode'] == 'WNW':
                dirvec[12] = 1.
            elif infoline['mode'] == 'N':
                dirvec[13] = 1.
            elif infoline['mode'] == 'NNW':
                dirvec[14] = 1.
            elif infoline['mode'] == 'ENE':
                dirvec[15] = 1.
            elif infoline['mode'] == 'NW':
                dirvec[16] = 1.
            else:
                print "wrong in dir"
        else:
            dirvec[-1] = 1.
        return dirvec
    #
    def getdirmode_alter(self, infoline):
        #
        dirvec = numpy.zeros((17, ), dtype=dtype)
        if infoline['type'] == 'windDir':
            if infoline['mode'] == 'S':
                dirvec[0] = 1.
            elif infoline['mode'] == 'SW':
                dirvec[1] = 1.
            elif infoline['mode'] == 'SSE':
                dirvec[2] = 1.
            elif infoline['mode'] == 'WSW':
                dirvec[3] = 1.
            elif infoline['mode'] == 'ESE':
                dirvec[4] = 1.
            elif infoline['mode'] == 'E':
                dirvec[5] = 1.
            elif infoline['mode'] == 'W':
                dirvec[6] = 1.
            elif infoline['mode'] == 'SE':
                dirvec[7] = 1.
            elif infoline['mode'] == 'NE':
                dirvec[8] = 1.
            elif infoline['mode'] == 'SSW':
                dirvec[9] = 1.
            elif infoline['mode'] == 'NNE':
                dirvec[10] = 1.
            elif infoline['mode'] == 'WNW':
                dirvec[11] = 1.
            elif infoline['mode'] == 'N':
                dirvec[12] = 1.
            elif infoline['mode'] == 'NNW':
                dirvec[13] = 1.
            elif infoline['mode'] == 'ENE':
                dirvec[14] = 1.
            elif infoline['mode'] == 'NW':
                dirvec[15] = 1.
            else:
                pass
                #print "wrong in dir"
        else:
            dirvec[-1] = 1.
        return dirvec
    #
    #
    def getgust(self, infoline):
        if infoline['type'] == 'gust':
            gustvec = self.getnums(
                infoline['min'],infoline['mean'],infoline['max']
            )
        else:
            gustvec = self.getnums('','','')
        return gustvec
    #
    def getcover(self, infoline):
        covervec = numpy.zeros((5,),dtype=dtype)
        if infoline['type'] == 'skyCover':
            if infoline['mode_bucket_0_100_4'] == '0-25':
                covervec[0] = 1.
            elif infoline['mode_bucket_0_100_4'] == '25-50':
                covervec[1] = 1.
            elif infoline['mode_bucket_0_100_4'] == '50-75':
                covervec[2] = 1.
            elif infoline['mode_bucket_0_100_4'] == '75-100':
                covervec[3] = 1.
            else:
                print "wrong in cover"
        else:
            covervec[-1] = 1.
        return covervec
    #
    def getprec(self, infoline):
        if infoline['type'] == 'precipPotential':
            precvec = self.getnums(
                infoline['min'],infoline['mean'],infoline['max']
            )
        else:
            precvec = self.getnums('','','')
        return precvec
    #
    def getthundermode(self, infoline):
        thundervec = numpy.zeros((6,),dtype=dtype)
        if infoline['type'] == 'thunderChance':
            if infoline['mode'] == '--':
                thundervec[0] = 1.
            elif infoline['mode'] == 'SChc':
                thundervec[1] = 1.
            elif infoline['mode'] == 'Chc':
                thundervec[2] = 1.
            elif infoline['mode'] == 'Lkly':
                thundervec[3] = 1.
            elif infoline['mode'] == 'Def':
                thundervec[4] = 1.
            else:
                print "wrong in thunder"
        else:
            thundervec[-1] = 1.
        return thundervec
    #
    def getrainmode(self, infoline):
        rainvec = numpy.zeros((6,),dtype=dtype)
        if infoline['type'] == 'rainChance':
            if infoline['mode'] == '--':
                rainvec[0] = 1.
            elif infoline['mode'] == 'SChc':
                rainvec[1] = 1.
            elif infoline['mode'] == 'Chc':
                rainvec[2] = 1.
            elif infoline['mode'] == 'Lkly':
                rainvec[3] = 1.
            elif infoline['mode'] == 'Def':
                rainvec[4] = 1.
            else:
                print "wrong in rain"
        else:
            rainvec[-1] = 1.
        return rainvec
    #
    def getsnowmode(self, infoline):
        snowvec = numpy.zeros((6,),dtype=dtype)
        if infoline['type'] == 'snowChance':
            if infoline['mode'] == '--':
                snowvec[0] = 1.
            elif infoline['mode'] == 'SChc':
                snowvec[1] = 1.
            elif infoline['mode'] == 'Chc':
                snowvec[2] = 1.
            elif infoline['mode'] == 'Lkly':
                snowvec[3] = 1.
            elif infoline['mode'] == 'Def':
                snowvec[4] = 1.
            else:
                print "wrong in snow"
        else:
            snowvec[-1] = 1.
        return snowvec
    #
    def getfreezmode(self, infoline):
        freezvec = numpy.zeros((6,),dtype=dtype)
        if infoline['type'] == 'freezingRainChance':
            if infoline['mode'] == '--':
                freezvec[0] = 1.
            elif infoline['mode'] == 'SChc':
                freezvec[1] = 1.
            elif infoline['mode'] == 'Chc':
                freezvec[2] = 1.
            elif infoline['mode'] == 'Lkly':
                freezvec[3] = 1.
            elif infoline['mode'] == 'Def':
                freezvec[4] = 1.
            else:
                print "wrong in freezvec"
        else:
            freezvec[-1] = 1.
        return freezvec
    #
    def getsleetmode(self, infoline):
        sleetvec = numpy.zeros((6,),dtype=dtype)
        if infoline['type'] == 'sleetChance':
            if infoline['mode'] == '--':
                sleetvec[0] = 1.
            elif infoline['mode'] == 'SChc':
                sleetvec[1] = 1.
            elif infoline['mode'] == 'Chc':
                sleetvec[2] = 1.
            elif infoline['mode'] == 'Lkly':
                sleetvec[3] = 1.
            elif infoline['mode'] == 'Def':
                sleetvec[4] = 1.
            else:
                print "wrong in sleetChance"
        else:
            sleetvec[-1] = 1.
        return sleetvec
    #
    def getinfovec(self, infoline):
        # typepart
        typevec = self.gettype(infoline)
        labelvec = self.getlabel(infoline)
        timevec = self.gettime(infoline)
        tempvec = self.gettemp(infoline)
        chillvec = self.getwindchill(infoline)
        speedvec = self.getwindspeed(infoline)
        bucket20vec = self.getbucket20(infoline)
        #dirvec = self.getdirmode_alter(infoline)
        dirvec = self.getdirmode(infoline)
        gustvec = self.getgust(infoline)
        covervec = self.getcover(infoline)
        precvec = self.getprec(infoline)
        thundervec = self.getthundermode(infoline)
        rainvec = self.getrainmode(infoline)
        snowvec = self.getsnowmode(infoline)
        freezvec = self.getfreezmode(infoline)
        sleetvec = self.getsleetmode(infoline)
        infovec = numpy.concatenate(
            (
                typevec, labelvec, timevec, tempvec, chillvec, speedvec, bucket20vec, dirvec, gustvec, covervec, precvec, thundervec, rainvec, snowvec, freezvec, sleetvec
            ),axis=0
        )
        return infovec
    #
    def getinfo(self, rawdata):
        infomat = []
        for theid in range(36):
            infoline = rawdata['id'+str(theid)]
            infomat.append(self.getinfovec(infoline))
        return numpy.float32(numpy.array(infomat) )
    #
    #

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
            self.seq_info_numpy[:, idx_in_batch, :] = self.getinfo(
                data_item
            )
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
        #print "processing one batch of data ... "
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
        self.seq_info_numpy[:,:] = self.getinfo(data_item)
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
                log_dict['max_dev_bleu'] = log_dict['tracked']['dev_bleu']
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
