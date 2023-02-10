# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:49:32 2022

@author: martho
"""


import os
from scipy import stats
import pickle
import h5py

from tqdm import tqdm
import numpy as np
import re
import scipy.signal as sig
import argparse
from scalib.metrics import SNR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model 
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.experimental.numpy as tnp
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten,AlphaDropout,LayerNormalization, Dense, Lambda,Conv1D, AveragePooling1D, BatchNormalization

tnp.experimental_enable_numpy_behavior()
from scalib.modeling import LDAClassifier
from utils.generate_intermediate_values import save_real_values, multGF256
import tensorflow_probability as tfp
from gmpy2 import f_divmod_2exp
from numba import njit
# Opening dataset specific variables 

file = open('utils/dataset_parameters','rb')
parameters = pickle.load(file)
file.close()


DATASET_FOLDER  = parameters['DATASET_FOLDER']
METRICS_FOLDER = DATASET_FOLDER + 'metrics/' 
MODEL_FOLDER = DATASET_FOLDER + 'models/' 
TRACES_FOLDER = DATASET_FOLDER + 'traces/'
REALVALUES_FOLDER = DATASET_FOLDER + 'real_values/'
POWERVALUES_FOLDER = DATASET_FOLDER + 'powervalues/'
TIMEPOINTS_FOLDER = DATASET_FOLDER + 'timepoints/'
KEY_FIXED = parameters['KEY']
FILE_DATASET = parameters['FILE_DATASET']
MASKS = parameters['MASKS']
ONE_MASKS = parameters['ONE_MASKS']
INTERMEDIATES = parameters['INTERMEDIATES']
VARIABLE_LIST = parameters['VARIABLE_LIST']
MASK_INTERMEDIATES = parameters['MASK_INTERMEDIATES']
MASKED_INTERMEDIATES = parameters['MASKED_INTERMEDIATES']




shift_rows_s = list([
    0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11
    ])






class PoolingCrop(tf.keras.layers.Layer):
    def __init__(self, input_dim=1, use_dropout = True,name = ''):
        if name == '':
            name = 'Crop_'+str(np.random.randint(0,high = 99999))
        super(PoolingCrop, self).__init__(name = name )
        self.w = self.add_weight(shape=(input_dim,1), dtype="float32",
                                  trainable=True,name = 'weights'+name,  regularizer = tf.keras.regularizers.L1L2(0.001)
                                  
        )
        self.input_dim = input_dim
        self.pooling = tf.keras.layers.AveragePooling1D(pool_size = 2,strides = 2,padding = 'same')
        self.use_dropout = use_dropout
        if self.use_dropout:
            self.dropout = tf.keras.layers.AlphaDropout(0.01)
        self.bn = tf.keras.layers.BatchNormalization()
        
        
    
        
    def call(self, inputs):
        kernel = tf.multiply(self.w, inputs)       
        pooling = self.pooling(kernel)
        output = self.bn(pooling)
        if self.use_dropout:
            output = self.dropout(output)
        return output
    
    def get_config(self):
        config = {'w':self.w,
                  'input_dim' : self.input_dim,
                  'pooling' : self.pooling,
                  'dropout' : self.dropout
                  }
        base_config = super(PoolingCrop,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    



class XorLayer(tf.keras.layers.Layer):
  def __init__(self,classes =256 ,name = ''):
    super(XorLayer, self).__init__(name = name)
    all_maps = np.load('xor_mapping.npy')
    mapping1 = []
    mapping2 = []
    for classe in range(classes):
        mapped = np.where(all_maps[classe] == 1)
        mapping1.append(mapped[0])
        mapping2.append(mapped[1])
    self.mapping1 = np.array(mapping1)
    self.mapping2 = np.array(mapping2)
    self.classes = classes
    
  def call(self, inputs):  
 
    pred1 = tnp.asarray(inputs[0])
    pred2 = tnp.asarray(inputs[1])
    p1 = pred1[:,self.mapping1]
    p2 = pred2[:,self.mapping2]

    res = tf.reduce_sum(tf.multiply(p1,p2),axis =2)   
    return res

    def get_config(self):
        config = {'mapping':self.mapping}
        base_config = super(XorLayer,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
log_table=[ 0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
    100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
    125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
    101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
    150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
    102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
    126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
    43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
    175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
    44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
    127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
    204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
    151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
    83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
    68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
    103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7 ]

alog_table =[1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
    95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
    229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
    83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
    76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
    131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
    181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
    254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
    251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
    195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
    159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
    155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
    252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
    69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
    18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
    57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1 ]

# Multiplication function in GF(2^8)
def multGF256(a,b):
    if (a==0) or (b==0):
        return 0
    else:
        return alog_table[(log_table[a]+log_table[b]) %255]
    
class MultiLayer(tf.keras.layers.Layer):
    def __init__(self,classes = 256 ,name = ''):
        super(MultiLayer, self).__init__(name = name)
        all_maps = np.load('mult_mapping.npy')
        mapping1 = []
        mapping2 = []
        for classe in range(classes):
            mapped = np.where(all_maps[classe] == 1)
            mapping1.append(mapped[0])
            mapping2.append(mapped[1])
        self.mapping1 = np.array(mapping1)
        self.mapping2 = np.array(mapping2)
        self.classes = classes
    
    def call(self, inputs):  
 
        pred1 = tnp.asarray(inputs[0])
        pred2 = tnp.asarray(inputs[1])
        p1 = pred1[:,self.mapping1]
        p2 = pred2[:,self.mapping2]
    
        res = tf.reduce_sum(tf.multiply(p1,p2),axis =2)   
        return res

    def get_config(self):
        config = {'mapping':self.mapping,
                  'classes':self.classes}
        base_config = super(MultiLayer,self).get_config()
        base_config.update(config)
        return base_config



class InvSboxLayer(tf.keras.layers.Layer):
  def __init__(self,name = ''):
    super(InvSboxLayer, self).__init__(name = name)
    self.mapping = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

  def call(self, inputs):  
    pred = tnp.asarray(inputs)[:,self.mapping]
    return tf.convert_to_tensor(pred)

    def get_config(self):
        config = {'mapping':self.mapping}
        base_config = super(InvSboxLayer,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))






class Add_Shares(tf.keras.layers.Layer):
    def __init__(self, input_dim=256,units = 256,shares = 1,name = ''):
        super(Add_Shares, self).__init__(name = name )
        
        self.w = self.add_weight(shape=(shares,input_dim,units), dtype="float32",trainable=True, name ='weights',  regularizer = tf.keras.regularizers.L1L2(0.002))
        self.b = self.add_weight(shape=(units,), dtype="float32",trainable=True, name ='biases')
        self.shares = shares
        self.input_dim = input_dim
        self.shares = shares
        self.selu = tf.keras.activations.selu
        self.layer_normalization = LayerNormalization()        
        
    def call(self, inputs):  
        activated_inputs = self.selu(inputs)
        normalised_inputs = tf.add(self.layer_normalization(activated_inputs),2)
        out = self.b        
        for share in range(self.shares):
            out = out + tf.matmul(normalised_inputs[share],self.w[share])
        return out

    def get_config(self):
        config = {'weights':self.w,
                  'biases': self.biases,
                  'input_dim' :self.input_dim,
                  'shares' : self.shares}
        base_config = super(Add_Shares,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





def get_pow_rank(x):
    if x == 2 :
        return 1
    if x == 1 :
        return 0
    n = 1
    q  , r = f_divmod_2exp(x , n)    
    while q > 0:
        q  , r = f_divmod_2exp(x , n)
        n += 1
    return n

def get_rank(result,true_value):
    key_probabilities_sorted = np.argsort(result)[::-1]
    key_ranking_good_key = list(key_probabilities_sorted).index(true_value) + 1
    return key_ranking_good_key

def load_model_from_target(structure , target,combine = False, window_type = 'classic',input_layer = 'classic'):
    model_file  = MODEL_FOLDER+ ('all_{}_{}_wt{}_{}.h5'.format(target,'cnn_best' ,window_type, input_layer) )
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure    

def load_model_unmasking_from_target(structure , target, window_type = 'classic',input_layer = 'classic'):
    model_file  = MODEL_FOLDER+ ('{}_{}_wt{}_{}.h5'.format(target ,'cnn_unmasking',window_type,input_layer)  )
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure   


def load_model_composed_from_target(structure , target, window_type = 'classic',input_layer = 'classic'):
    model_file  = MODEL_FOLDER+ ('{}_{}_wt{}_{}.h5'.format(target ,'cnn_unmasking_multiple_shares',window_type,input_layer)  )
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure   

def load_model_propagation_from_target(structure, window_type = 'classic',input_layer = 'classic',shared = False):
    model_file  = MODEL_FOLDER+ ('all_k1_{}_wt{}_{}.h5'.format('cnn_propagation{}'.format('' if not shared else '_shared'),window_type,input_layer ))
    print('Loading model {}'.format(model_file))
    structure.load_weights(model_file)
    return structure  








def read_from_h5_file(file = None,n_traces = 1000,masked = False,dataset = 'training'):   
    
    f = h5py.File(DATASET_FOLDER + FILE_DATASET,'r')[dataset]  
    labels_dict = f['labels']
    data =  {'keys':f['keys']   ,'plaintexts':f['plaintexts']}

    return  f['traces'][:n_traces] , labels_dict, data
def get_byte(i):
    for b in range(17,1,-1):
        if str(b) in i:
            return b
    return -1
    


def to_matrix(text):

    matrix = []
    for i in range(4):
        matrix.append([0,0,0,0])
    for i in range(len(text)):
        elem = text[i]
        matrix[i%4][i//4] =elem
    return matrix    








    

def load_dataset(target,intermediate,n_traces = None,load_masks = False,window_type = "classic",dataset = 'training',encoded_labels = True,print_logs = True):    
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for {}'.format(target)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces_val = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces if intermediate in ['p','rin','alpha','beta'] else traces[:,3150:3350]

    
    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val if intermediate in ['p','rin','alpha','beta'] else traces_val[:,3150:3350]
        

    Y_profiling_dict = {}
    real_values = np.array(labels_dict[intermediate],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output'] = get_hot_encode(real_values,classes = 256 if not intermediate == 'p' else 16) if encoded_labels else  real_values 
   
    if training:
        Y_validation_dict = {}
        real_values_val = np.array(labels_dict_val[intermediate],dtype = np.uint8)
        Y_validation_dict['output'] = get_hot_encode(real_values_val,classes = 256 if not intermediate == 'p' else 16) if encoded_labels else  real_values_val 

    if training:       
        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)   
       



def load_dataset_multi(n_traces = None,dataset = 'training',encoded_labels = True,print_logs = True):
    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for all together but obviously permuted'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces_val = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces 

    
    X_profiling_dict = {}  
    permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val
        
        permutations_val = np.array(labels_dict_val['p'],dtype = np.uint8)
        X_validation_dict = {}  

    all_inputs = None
    all_inputs_val = None


    if print_logs :
        print('Loaded inputs')    
        print('Shape : ',all_inputs.shape)
        

    real_values_t1_rin = np.array(labels_dict['t1^rin'],dtype = np.uint8)[:n_traces]
    real_values_s1_beta = np.array(labels_dict['s1^beta'],dtype = np.uint8)[:n_traces]
    permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict = {}  
    real_values_beta = np.array(labels_dict['beta'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output_beta'] = get_hot_encode(real_values_beta) if encoded_labels else  real_values_beta    
    real_values_alpha = np.array(labels_dict['alpha'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output_alpha'] = get_hot_encode(real_values_alpha) if encoded_labels else  real_values_alpha       
    real_values_rin = np.array(labels_dict['rin'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output_rin'] = get_hot_encode(real_values_rin) if encoded_labels else  real_values_rin 
    Y_profiling_dict['output_t1_rin'] = get_hot_encode(real_values_t1_rin) if encoded_labels else  real_values_t1_rin 
    Y_profiling_dict['output_s1_beta'] = get_hot_encode(real_values_s1_beta) if encoded_labels else  real_values_s1_beta
    Y_profiling_dict['output_permutation'] = get_hot_encode(permutations,classes = 16) if encoded_labels else  permutations


     

                                              
    if training:
        real_values_t1_rin_val = np.array(labels_dict_val ['t1^rin'],dtype = np.uint8)
        real_values_s1_beta_val  = np.array(labels_dict_val ['s1^beta'],dtype = np.uint8)
        real_values_beta_val = np.array(labels_dict_val['beta'],dtype = np.uint8)
        permutations_val = np.array(labels_dict_val['p'],dtype = np.uint8)[:n_traces]
        Y_validation_dict = {}
        Y_validation_dict['output_beta'] = get_hot_encode(real_values_beta_val) if encoded_labels else  real_values_beta_val    
        real_values_alpha_val = np.array(labels_dict_val['alpha'],dtype = np.uint8)
        Y_validation_dict['output_alpha'] = get_hot_encode(real_values_alpha_val) if encoded_labels else  real_values_alpha_val       
        real_values_rin_val = np.array(labels_dict_val['rin'],dtype = np.uint8)
        Y_validation_dict['output_rin'] = get_hot_encode(real_values_rin_val) if encoded_labels else  real_values_rin_val          
        
        Y_validation_dict['output_t1_rin'] = get_hot_encode(real_values_t1_rin_val) if encoded_labels else  real_values_t1_rin_val 
        Y_validation_dict['output_s1_beta'] = get_hot_encode(real_values_s1_beta_val) if encoded_labels else  real_values_s1_beta_val
        Y_validation_dict['output_permutation'] = get_hot_encode(permutations_val,classes = 16) if encoded_labels else  permutations_val




        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)    



def load_dataset_hierarchical(n_traces = 2500000,load_masks = False,dataset = 'training',encoded_labels = True,print_logs = True):

    training = dataset == 'training' 
    if print_logs :
        str_targets = 'Loading samples and labels for all together but obviously permuted'
        print(str_targets)
        
    traces , labels_dict = read_from_h5_file(n_traces=n_traces,dataset = dataset)
    traces_val = np.expand_dims(traces,2)
    X_profiling_dict = {}  
    X_profiling_dict['traces'] = traces 

    
    X_profiling_dict = {}  
    permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]

    if training:
        traces_val , labels_dict_val = read_from_h5_file(n_traces=n_traces,dataset = 'test')
        traces_val = np.expand_dims(traces_val,2)
        X_validation_dict = {}  
        X_validation_dict['traces'] = traces_val
        
        permutations_val = np.array(labels_dict_val['p'],dtype = np.uint8)
        X_validation_dict = {}  

    all_inputs = None
    all_inputs_val = None


    if print_logs :
        print('Loaded inputs')    
        print('Shape : ',all_inputs.shape)
        

    real_values_t1_rin = np.array(labels_dict['t1^rin'],dtype = np.uint8)[:n_traces]
    real_values_t1 = np.array(labels_dict['t1'],dtype = np.uint8)[:n_traces]
    real_values_s1_beta = np.array(labels_dict['s1^beta'],dtype = np.uint8)[:n_traces]
    permutations = np.array(labels_dict['p'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict = {}  
    real_values_beta = np.array(labels_dict['beta'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output_beta'] = get_hot_encode(real_values_beta) if encoded_labels else  real_values_beta    
    real_values_alpha = np.array(labels_dict['alpha'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output_alpha'] = get_hot_encode(real_values_alpha) if encoded_labels else  real_values_alpha       
    real_values_rin = np.array(labels_dict['rin'],dtype = np.uint8)[:n_traces]
    Y_profiling_dict['output_rin'] = get_hot_encode(real_values_rin) if encoded_labels else  real_values_rin 
    Y_profiling_dict['output_t1_rin'] = get_hot_encode(real_values_t1_rin) if encoded_labels else  real_values_t1_rin 
    Y_profiling_dict['output_s1_beta'] = get_hot_encode(real_values_s1_beta) if encoded_labels else  real_values_s1_beta
    Y_profiling_dict['output_permutation'] = get_hot_encode(permutations,classes = 16) if encoded_labels else  permutations
    Y_profiling_dict['output'] = get_hot_encode(real_values_t1) if encoded_labels else  real_values_t1

     

                                              
    if training:
        real_values_t1_rin_val = np.array(labels_dict_val ['t1^rin'],dtype = np.uint8)
        real_values_s1_beta_val  = np.array(labels_dict_val ['s1^beta'],dtype = np.uint8)
        real_values_t1_val = np.array(labels_dict_val['t1'],dtype = np.uint8)[:n_traces]
        real_values_beta_val = np.array(labels_dict_val['beta'],dtype = np.uint8)
        permutations_val = np.array(labels_dict_val['p'],dtype = np.uint8)[:n_traces]
        Y_validation_dict = {}
        Y_validation_dict['output_beta'] = get_hot_encode(real_values_beta_val) if encoded_labels else  real_values_beta_val    
        real_values_alpha_val = np.array(labels_dict_val['alpha'],dtype = np.uint8)
        Y_validation_dict['output_alpha'] = get_hot_encode(real_values_alpha_val) if encoded_labels else  real_values_alpha_val       
        real_values_rin_val = np.array(labels_dict_val['rin'],dtype = np.uint8)
        Y_validation_dict['output_rin'] = get_hot_encode(real_values_rin_val) if encoded_labels else  real_values_rin_val          
        
        Y_validation_dict['output_t1_rin'] = get_hot_encode(real_values_t1_rin_val) if encoded_labels else  real_values_t1_rin_val 
        Y_validation_dict['output_s1_beta'] = get_hot_encode(real_values_s1_beta_val) if encoded_labels else  real_values_s1_beta_val
        Y_validation_dict['output_permutation'] = get_hot_encode(permutations_val,classes = 16) if encoded_labels else  permutations_val
        Y_validation_dict['output'] = get_hot_encode(real_values_t1_val) if encoded_labels else  real_values_t1_val

        return tf.data.Dataset.from_tensor_slices((X_profiling_dict ,Y_profiling_dict)), tf.data.Dataset.from_tensor_slices(( X_validation_dict,Y_validation_dict)) 
   
    else:
        return (X_profiling_dict,Y_profiling_dict)    



def get_hw(k):
    hw = 0
    for _ in range(8):
        hw += k & 1
        k = k >> 1
    return hw 

def convert_to_binary(e):
    return [1 if e & (1 << (7-n)) else 0 for n in range(8)]   




def get_rank_list_from_prob_dist(probdist,l):
    res =  []
    accuracy = 0
    size = len(l)
    res_score = []
    accuracy_top5 = 0
    for i in range(size):
        rank = get_rank(probdist[i],l[i])
        res.append(rank)
        res_score.append(probdist[i][l[i]])
        accuracy += 1 if rank == 1 else 0
        accuracy_top5 += 1 if rank <= 5 else 0
    return res,(accuracy/size)*100,res_score , (accuracy_top5/size)*100

def get_variable_name(string):
    # sk102-0 -> sk
    try:
        return re.search(r'^[a-z]+', string).group(0)
    except AttributeError:
        return None

def get_variable_number(string):
    # sk102-0 -> 102
    try:
        return int(re.search(r'\d{3}', string).group(0))
    except AttributeError:
        return None
def normalise_neural_trace(v):
    # Shift up
    return v - np.min(v)

def normalise_neural_trace_single(v):
    return divide_rows_by_max(normalise_neural_trace(v))

def divide_rows_by_max(X):
    if len(X.shape) == 1:
        return X / np.max(X)
    else:
        return X / np.max(X, axis=1)[:, None]
def normalise_neural_traces(X):


        # DEBUG

    divided_by_max = divide_rows_by_max(X)
    return divided_by_max

def normalise_traces_to_int8(x):
    x = normalise_neural_traces(x)
    x = x  * 128
    return x.astype(np.int8)


def get_hot_encode(label_set,classes = 256):    
    return np.eye(classes)[label_set]





def get_windowed(traces,timepoints,window,window_type = "classic"):

    indexes = None
    for timepoint in timepoints:
        window_timepoint = window // len(timepoints)
        index = np.arange(timepoint - window_timepoint//2, timepoint + window_timepoint//2,step = 1)
        
        if window_type == "gaussian":
            try:
                distr_samples = np.load(TIMEPOINTS_FOLDER + 'gaussian_dist_samples_{}_timepoint_{}.npy'.format(window_timepoint,timepoint))
            except:
                x = np.arange(-window_timepoint* 5, window_timepoint * 5 + 1)
                xU, xL = x + 0.5, x - 0.5 
                prob = stats.norm.cdf(xU, scale = window_timepoint*2) - stats.norm.cdf(xL, scale = window_timepoint*2)
                prob = prob / prob.sum() # normalize the probabilities so their sum is 1
                distr_samples = sorted(np.random.choice(x, size = window_timepoint, p = prob, replace=False))
                np.save(TIMEPOINTS_FOLDER + 'gaussian_dist_samples_{}_timepoint_{}.npy'.format(window_timepoint,timepoint),distr_samples)
            index = distr_samples + timepoint
        if indexes is None:
            indexes = index
        else:
            indexes = np.append(indexes,index,axis = 0)
            
            
    return traces[:,indexes]  
    

def save_labels(dataset,n_traces = None):
    masks = None
    keys = None
    ciphertexts = None
    training = 'training' in dataset
    f = h5py.File(DATASET_FOLDER + FILE_DATASET,'r')
    dataset_h5 = f.get(dataset)              
    if 'masks' in dataset_h5:
        masks = np.array(dataset_h5.get('masks'))
    plaintexts = np.array(dataset_h5.get('plaintexts'))
    
    if 'keys' in dataset_h5:
        
        keys = np.array(dataset_h5.get('keys'))
    if 'ciphertexts' in dataset_h5:
        ciphertexts = np.array(dataset_h5.get('ciphertexts'))
  
    labels_dict = save_real_values(ciphertexts=ciphertexts, plaintexts= plaintexts, random =masks,keys=keys,n_traces = n_traces,save_file = False)   

    file = open(REALVALUES_FOLDER + 'labels_dict{}'.format('' if training else '_{}'.format(dataset)),'wb')
    pickle.dump(labels_dict,file)
    file.close()         


def save_power_values_windowed(traces,dataset = 'training',target = "s003",window= 2000,window_type = "classic",add_mask_leakage = False):

    if window_type == 'averaged':
        if traces.shape[0] > 10000:
            chunk_size = 10000
            chunk_n = traces.shape[0] // chunk_size
            for iterations in range(chunk_n):
                if iterations == 0: 
                    windowed_traces =preprocess(traces[:chunk_size],10)
                else:
                    windowed_traces = np.append(windowed_traces,preprocess(traces[iterations * chunk_size:(iterations+1) * chunk_size],10),axis = 0)
            
        else:
            windowed_traces = preprocess(traces,10)        
        np.save(POWERVALUES_FOLDER + ('{}_traces_windowt_{}_{}.npy'.format(dataset,window_type,'whole_trace')),normalise_neural_traces(windowed_traces),allow_pickle=True)
        print('Saved {}'.format(POWERVALUES_FOLDER + ('{}_traces_windowt_{}_{}.npy'.format(dataset,window_type,'whole_trace'))))    
    
    elif window_type =='first_round':
        np.save(POWERVALUES_FOLDER + ('{}_traces{}_windowt_{}_{}.npy'.format(dataset,'_with_mask_leakages' if add_mask_leakage else '',window_type,'all')),traces,allow_pickle=True)
        print('Saved {}'.format(POWERVALUES_FOLDER + ('{}_traces{}_windowt_{}_{}.npy'.format(dataset,'_with_mask_leakages' if add_mask_leakage else '',window_type,'all'))))
            
    
    
    elif window_type == 'best_snr':
        snr = np.load(TIMEPOINTS_FOLDER +target+'_array.npy')
        sorted_snr = sorted(np.argsort(snr)[-window:])
        np.save(TIMEPOINTS_FOLDER + '{}_snr_values_wt{}.npy'.format(target,window_type),sorted_snr)
        windowed_traces = traces[:,sorted_snr]
        np.save(POWERVALUES_FOLDER + ('{}_traces{}_windowt_{}_{}.npy'.format(dataset,'_with_mask_leakages' if add_mask_leakage else '',window_type,target)),windowed_traces,allow_pickle=True)
        print('Saved {}'.format(POWERVALUES_FOLDER + ('{}_traces{}_windowt_{}_{}.npy'.format(dataset,'_with_mask_leakages' if add_mask_leakage else '',window_type,target))))
    else:
        if target == "alpha":
            windowed_traces = traces[:,:5000]
        else:
            timepoints = sorted(np.load(TIMEPOINTS_FOLDER +target+'.npy'))
            print('Timepoint : ',timepoints)
            snr = np.load(TIMEPOINTS_FOLDER +target+'_array.npy')
            windowed_traces = get_windowed(traces,timepoints,window = window, window_type = window_type)
            snr_window = get_windowed(snr.reshape(1,-1) ,timepoints,window = window, window_type = window_type)[0]
            np.save(TIMEPOINTS_FOLDER + '{}_snr_values_wt{}.npy'.format(target,window_type),snr_window)
            if add_mask_leakage and (not len(MASKS) == 0):
                if not target in MASKS:
                    for masks in MASKS:
                        if masks in target:                    
                            timepoint = np.load(TIMEPOINTS_FOLDER+masks+'.npy')                    
                            windows_masks = get_windowed(traces, timepoint, window,window_type= window_type)  
                            windowed_traces = np.concatenate([windowed_traces,windows_masks],axis = 1)
        
        np.save(POWERVALUES_FOLDER + ('{}_traces{}_windowt_{}_{}.npy'.format(dataset,'_with_mask_leakages' if add_mask_leakage else '',window_type,target)),windowed_traces,allow_pickle=True)
        print('Saved {}'.format(POWERVALUES_FOLDER + ('{}_traces{}_windowt_{}_{}.npy'.format(dataset,'_with_mask_leakages' if add_mask_leakage else '',window_type,target))))
    return


def save_meaned_trace(target = "s1",window_type = "classic",add_mask_leakage = False):
    traces = None
    labels = None
    file = open(REALVALUES_FOLDER+ 'labels_dict','rb')
    labels_dict =np.array(pickle.load(file)[target])
    file.close()
    byte_count = 0
    
    count_traces = {}
    for byte_target in VARIABLE_LIST[target]:
        print(byte_target)
        traces = np.load(POWERVALUES_FOLDER + ('training_traces{}_windowt_{}_{}.npy'.format('_with_mask_leakages' if add_mask_leakage else '',window_type,byte_target)))
        print(traces.shape)
        labels = labels_dict[:,byte_count]
        
        for val in range(256):
            indexes_val = np.where(labels == val)[0]
            mean_trace_intermediate= np.mean(traces[indexes_val],axis = 0)
            np.save(POWERVALUES_FOLDER +'meaned_traces_val{}_target_{}.npy'.format(val,byte_target),normalise_neural_trace_single(mean_trace_intermediate))
       
            # np.save(POWERVALUES_FOLDER +'meaned_traces_val{}_target{}.npy'.format(val,byte_target),mean_trace_intermediate[byte_target][val] )
        byte_count += 1

    return

 
def normalise_predictions(pred):
    return pred / sum(pred)
def normalise_correlations(pred):
    return pred / max(pred)



def preprocess(traces,coef = 10):
    return  sig.resample(traces,traces.shape[1]//coef,axis=1)    
 


def save_average_correlation(target):
    l = []
    for byte_target in VARIABLE_LIST[target]:
        l.append(np.load(TIMEPOINTS_FOLDER+ byte_target +'_array.npy'))
    np.save(TIMEPOINTS_FOLDER+ 'average_correlations.npy',np.mean(np.array(l),axis = 0))
        

def get_var(f):
    return f.split('_')[0]

def get_factor(f):
    end = f.split('_')[-1]
    try:
        return int(end.split('.pkl')[0])
    except:
        return float(end.split('.pkl')[0])

def get_factor_h5(f):
    end = f.split('_')[-2]
    try:
        return int(end)
    except:
        return float(end)
def get_factor_elem(f):
    end = f.split('_')[1]
    try:
        return int(end)
    except:
        return float(end)
  
def get_epochs(file):
    end = file.split('_')[-1]
    try:
        return int(end.split('.h5')[0])
    except:
        return float(end.split('.h5')[0])   
  
def corrLossWrapper(weights):

    def correlationLoss(x,y, axis=-2):
      """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
      while trying to have the same mean and variance"""
      x = tf.convert_to_tensor(x)
      y = tf.cast(y, x.dtype)
      n = tf.cast(tf.shape(x)[axis], x.dtype)
      xsum = tf.reduce_sum(x, axis=axis)
      ysum = tf.reduce_sum(y, axis=axis)
      xmean = xsum / n
      ymean = ysum / n
      xsqsum = tf.reduce_sum( tf.cast((weights),tf.float32) * tf.math.squared_difference(x, xmean), axis=axis)
      ysqsum = tf.reduce_sum( tf.cast((weights),tf.float32) * tf.math.squared_difference(y, ymean), axis=axis)
      cov = tf.reduce_sum( tf.cast((weights),tf.float32) * (x - xmean) * (y - ymean), axis=axis)
      corr = cov / tf.sqrt(xsqsum * ysqsum)
      # absdif = tmean(tf.abs(x - y), axis=axis) / tf.sqrt(yvar)
      sqdif = tf.reduce_sum(tf.math.squared_difference(x, y), axis=axis) / n / tf.sqrt(ysqsum / n)
      # meandif = tf.abs(xmean - ymean) / tf.abs(ymean)
      # vardif = tf.abs(xvar - yvar) / yvar
      # return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (meandif * 0.01) + (vardif * 0.01)) , dtype=tf.float32 )
      return tf.convert_to_tensor( K.mean(1 - corr + (0.01 * sqdif)) , dtype=tf.float32 )
    return  correlationLoss

def tf_median_probability_loss(y_true, y_pred):
    # undo one-hot
    
    argmaxed_onehot = tf.argmax(y_true, output_type=tf.int32, axis=1)
    # reshape

    reshaped_onehot = tf.expand_dims(argmaxed_onehot, 1)
    # get tensor ([0,1,2,...])
    tf_range = tf.range(tf.shape(y_pred)[0], dtype=tf.int32)
    # reshape

    reshaped_tf_range = tf.expand_dims(tf_range, 1)
    # Concatenate range to onehot

    
    concatenated_onehot = tf.concat([reshaped_tf_range, reshaped_onehot], 1)
    # Gather the probabilities together!
    gathered = tf.gather_nd(y_pred, concatenated_onehot)
    # Take the mean of these ranks (float value)
    median = 1 - tf.cast(tfp.stats.percentile(gathered, 50.0), tf.float32)
    # print "Our Rank Median:\ntype {} ({}), shape {}".format(type(median), median.dtype, median.get_shape())
    return median


def tf_rank_loss(y_true, y_pred):
    output = y_pred
    target = y_true
    axis = -1

    # scale preds so that the class probas of each sample sum to 1
    output = output / tf.math.reduce_sum(output, keepdims=True)
    # manual computation of crossentropy
    epsilon_ = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
    clipped_output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)
    return_val = -tf.math.reduce_sum(target * tf.math.log(clipped_output))

    # Our bit: Rank!

    # Sort once to get each index ranking
    argsort1 = tf.argsort(y_pred, direction='DESCENDING')
    # Sort the sorted to put them in a nice order for us
    argsort2 = tf.argsort(argsort1, direction='ASCENDING')
    # undo one-hot
    argmaxed_onehot = tf.argmax(y_true, output_type=tf.int32, axis=1)
    # reshape
    reshaped_onehot = tf.expand_dims(argmaxed_onehot, 1)
    # get tensor ([0,1,2,...])
    tf_range = tf.range(tf.shape(argsort2)[0], dtype=tf.int32)
    # reshape
    reshaped_tf_range = tf.expand_dims(tf_range, 1)
    # Concatenate range to onehot
    concatenated_onehot = tf.concat([reshaped_tf_range, reshaped_onehot], 1)
    # Gather the ranks together!
    gathered = tf.gather_nd(argsort2, concatenated_onehot)
    # Take the mean of these ranks (float value)
    mean = tf.cast(tf.reduce_mean(gathered), tf.float32)
    return return_val + mean
    


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Save power traces values for targets')

    parser.add_argument('-w', '-window',  action="store", dest="WINDOW", help='Size window saved',
                        type=int, default=2000)
    parser.add_argument('-t', '-traces',  action="store", dest="N_TRACES", help='Size window saved',
                        type=int, default=100000)

    
    parser.add_argument('-target', action="store", dest="TARGET",
                        help='Target the selected target', type=str, default='t1^uv')
    parser.add_argument('-wt', action="store", dest="WINDOW_TYPE",
                        help='Target the selected target', type=str, default='classic')
    parser.add_argument('--ADD', action="store_true", dest="ADD_MASK_LEAKAGE",
                        help='Add leakage of masks',  default=False)
    parser.add_argument('--HW', action="store_true", dest="HW",
                        help='perform on HW instead of identity',  default=False)

    parser.add_argument('--MEAN', action="store_true", dest="MEAN", help='for test dataset', default=False)
    parser.add_argument('--SAVE_POWER_VAL', action="store_true", dest="SAVE_POWER_VAL", help='for test dataset', default=False)
    parser.add_argument('--LABELS', action="store_true", dest="LABELS", help='for test dataset', default=False)
    parser.add_argument('--TEST', action="store_true", dest="TEST", help='for test dataset', default=False)
    parser.add_argument('--ATTACK', action="store_true", dest="ATTACK", help='for attack dataset', default=False)
    parser.add_argument('--PROPAGATION', action="store_true", dest="PROPAGATION", help='for attack dataset', default=False)
    args            = parser.parse_args()
  
    TARGET         = args.TARGET
    ATTACK         = args.ATTACK
    ADD_MASK_LEAKAGE = args.ADD_MASK_LEAKAGE
    LABELS = args.LABELS
    TEST = args.TEST
    MEAN = args.MEAN
    PROPAGATION = args.PROPAGATION
    SAVE_POWER_VAL = args.SAVE_POWER_VAL
    WINDOW = args.WINDOW
    WINDOW_TYPE = args.WINDOW_TYPE
    N_TRACES = args.N_TRACES
    
    
    print('Loading traces ! ')
    dataset = 'training'
    if TEST:
        dataset = 'test'
    if ATTACK:
        dataset = 'attack'
    
    if LABELS:
        save_labels(dataset,n_traces = N_TRACES)
    
    if SAVE_POWER_VAL :
        traces = load_traces_from_dataset(dataset = dataset,n_traces = N_TRACES)
        
        if PROPAGATION:
            windowed_traces = None
            for intermediate in INTERMEDIATES:
                
                if intermediate == 's2' or intermediate == 'k2' or intermediate == 't2':
                    continue
                print(intermediate)
                n_round = '1' if '1' in intermediate else '2'
                
                intermediate_split = intermediate.split(n_round)[0]
                start = 0 if n_round == '1' else 16
                end =  16 if n_round == '1' else 32
                end = 12 if intermediate == 'h1' else end
                timepoints = np.load(TIMEPOINTS_FOLDER + '{}.npy'.format(intermediate_split))[start:end]                  
                if windowed_traces is None:
                    windowed_traces = get_windowed(traces, timepoints, WINDOW)
                else:
                    windowed_traces = np.append(windowed_traces,get_windowed(traces, timepoints, WINDOW),axis = 1)
            np.save(POWERVALUES_FOLDER + '{}_propagation_traces.npy'.format(dataset),windowed_traces,allow_pickle = True)
        else:
            if not WINDOW_TYPE == "averaged" and not  WINDOW_TYPE == "first_round" :
                for byte_target in VARIABLE_LIST[TARGET]:    
                    print('Starting to save for {}'.format(byte_target))
                    save_power_values_windowed(traces,target = byte_target,dataset = dataset,window_type = WINDOW_TYPE,window = WINDOW,add_mask_leakage = ADD_MASK_LEAKAGE)
                  
            else:
                save_power_values_windowed(traces,target = TARGET,dataset = dataset,window_type = WINDOW_TYPE,window = WINDOW,add_mask_leakage = ADD_MASK_LEAKAGE)
    if MEAN:
        save_meaned_trace(target = TARGET,window_type=WINDOW_TYPE,add_mask_leakage = ADD_MASK_LEAKAGE) 
        
            


