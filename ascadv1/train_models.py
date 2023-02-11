# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:10:35 2021

@author: martho
"""


import argparse
import os
import numpy as np
import pickle
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from tensorflow.keras.regularizers import L2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Softmax, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization,Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from multiprocessing import Process



# import dataset paths and variables
from utility import VARIABLE_LIST , METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import PoolingCrop , XorLayer , InvSboxLayer, Add_Shares
from utility import load_dataset, load_dataset_multi ,load_dataset_hierarchical



seed = 42


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################



### Single-Task Models

def block_reg(inputs):
    x = Lambda(lambda x: K.l2_normalize(x,axis=1))(inputs)
    x = BatchNormalization()(x)
    return x

def cnn_best(input_length=250000, learning_rate=0.001, classes=256, dense_units=200 , snr =None,input_layer = 'classic'):
  
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    
    x = input_layer_creation(inputs, input_layer,input_length,target_size=25000)    
    x = core_cnn(x,dense_units)
     
    output = Dense(classes, activation='softmax' if classes > 1 else 'sigmoid',name = 'output')(x)
    
    outputs = {}
    outputs['output'] = output
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = [inputs],outputs = outputs,name='cnn_best')
    # model.summary()
    model.compile(loss='categorical_crossentropy' if classes > 1 else 'mse', optimizer=optimizer, metrics=['accuracy'])
 
    return model

def input_layer_creation(inputs,input_layer,input_length,target_size = 2500,use_dropout = True,name = '',snr = None):

        
    size = input_length
    
    iteration  = 0
    crop = inputs
    
    while size > target_size:
        crop = PoolingCrop(input_dim = size,use_dropout=use_dropout,name = name)(crop)
        iteration += 1
        size = math.ceil(size/2)

        x = crop  
    return x


def core_cnn(inputs,dense_units):
    x = Conv1D(kernel_size=34, strides=17, filters=4, activation='selu', padding='same')(inputs)
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)
    x = block_reg(x)     
    x = Flatten()(x)
    x = Dense(dense_units, activation='selu', kernel_initializer='random_uniform',kernel_regularizer = L2(0.0001))(x)
    x = Dense(dense_units, activation='selu', kernel_initializer='random_uniform',kernel_regularizer = L2(0.0001))(x)
    return x


def core_cnn_shared(inputs):
    x = Conv1D(kernel_size=34, strides=17, filters=4, activation='selu', padding='same')(inputs)
    x = AveragePooling1D(pool_size=2, strides=2, padding='same')(x)
    x = block_reg(x)       
    x = Flatten()(x)
    return x


def predictions_branch(inputs,dense_units,n_blocks = 2,dropout = 0.1):
    x = inputs
    for block in range(n_blocks):
        x = Dense(dense_units, activation='selu', kernel_initializer='random_uniform',kernel_regularizer = tf.keras.regularizers.L2(0.0001))(x)
    return Dense(256)(x)

### Multi-Task Models

def cnn_hierarchical_subbytes_inputs(input_length=250000, learning_rate=0.001, classes=256, dense_units=200 ,input_layer = 'classic',snr = None,n_byte = 3):      
    inputs  = Input(shape=(input_length, 1),name = 'inputs')
    target_size = 25000    
    outputs = {}
    losses = {}    
    weights = {}
    xor_branches = []

    main_inputs = input_layer_creation(inputs, input_layer,input_length,target_size= target_size) 
    main_branch = core_cnn_shared(main_inputs)
    

    pred_i = predictions_branch(main_branch,dense_units)
    output_i = Softmax(name = 'output_i')(pred_i) 
    outputs['output_i'] = output_i
 
    pred_r = predictions_branch(main_branch,dense_units)
    output_r = Softmax(name = 'output_r')(pred_r) 
    outputs['output_r'] = output_r   
    
    pred_ri = XorLayer(name = 'Xor_r_i')([output_r,output_i])
 
 
    pred_t1_ri = predictions_branch(main_branch,dense_units)
    output_t1_ri = Softmax(name = 'output_t1_ri')(pred_t1_ri)
    outputs['output_t1_ri'] = output_t1_ri            
            

    pred_t1_i = predictions_branch(main_branch,dense_units)
    output_t1_i = Softmax(name = 'output_t1_i')(pred_t1_i)
    outputs['output_t1_i'] = output_t1_i       


    pred_t1_r = predictions_branch(main_branch,dense_units)
    output_t1_r = Softmax(name = 'output_t1_r')(pred_t1_r)
    outputs['output_t1_r'] = output_t1_r      

    
    xor_t1_fixed_i = XorLayer(name = 'Xor_i_fixed_i')([pred_t1_i,output_i])
    xor_t1_fixed_t1_i = XorLayer(name = 'Xor_t1_fixed_t1_i')([output_t1_i,pred_i])
    
    xor_t1_fixed_r = XorLayer(name = 'Xor_r_fixed_r')([pred_t1_r,output_r])
    xor_t1_fixed_t1_r = XorLayer(name = 'Xor_r_fixed_t1_r')([output_t1_r,pred_r])
    
    xor_ri = XorLayer(name = 'Xor_ri')([pred_t1_ri,pred_ri])
    
    xor_branches = [xor_t1_fixed_i,xor_t1_fixed_t1_i, xor_t1_fixed_r,xor_t1_fixed_t1_r , xor_ri]
    
    pred_output = Add_Shares(name = 'Add_shares',shares = len(xor_branches),input_dim = classes,units = classes)(xor_branches)
    
    output = Softmax(name = 'output')(pred_output)
    outputs['output'] = output
    
    
    for k , v in outputs.items():
        weights[k] = 1
        losses[k] = 'categorical_crossentropy'    

    inputs_dict = {}

      
    inputs_dict['traces'] = inputs    

    print(output.keys())
    
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_hierarchical_subbytes_inputs')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'],loss_weights = weights)
    model.summary()
 
 
    return model  


def cnn_multi_task_subbytes_inputs(input_length=250000, learning_rate=0.001, classes=256, dense_units=200 ,input_layer = 'classic',snr = None,n_byte = 3):
    inputs  = Input(shape=(input_length, 1),name = 'inputs')
    target_size = 25000    
    outputs = {}
    losses = {}    
    weights = {}

    main_inputs = input_layer_creation(inputs, input_layer,input_length,target_size= target_size) 
    main_branch = core_cnn_shared(main_inputs)
    

    pred_i = predictions_branch(main_branch,dense_units)
    output_i = Softmax(name = 'output_i')(pred_i) 
    outputs['output_i'] = output_i
 
    pred_r = predictions_branch(main_branch,dense_units)
    output_r = Softmax(name = 'output_r')(pred_r) 
    outputs['output_r'] = output_r   
    
    pred_t1_ri = predictions_branch(main_branch,dense_units)
    output_t1_ri = Softmax(name = 'output_t1_ri')(pred_t1_ri)
    outputs['output_t1_ri'] = output_t1_ri            
            

    pred_t1_i = predictions_branch(main_branch,dense_units)
    output_t1_i = Softmax(name = 'output_t1_i')(pred_t1_i)
    outputs['output_t1_i'] = output_t1_i       


    pred_t1_r = predictions_branch(main_branch,dense_units)
    output_t1_r = Softmax(name = 'output_t1_r')(pred_t1_r)
    outputs['output_t1_r'] = output_t1_r      

    

    
    
    for k , v in outputs.items():
        weights[k] = 1
        losses[k] = 'categorical_crossentropy'    

    inputs_dict = {}

      
    inputs_dict['traces'] = inputs    


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task_subbytes_inputs')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'],loss_weights = weights)
    model.summary()
 
 
    return model  

def cnn_multi_task_multi_target(input_length=250000, learning_rate=0.001, classes=256, dense_units=200,timepoints = None,input_layer = "weighted_pooling"):
    inputs = Input(shape=(input_length, 1),name = 'traces')
    
    target_size = 25000
    input_dict = {'traces' : inputs}
    losses = {}
    weights = {}
    outputs = {}
    
    inputs_main = input_layer_creation(inputs, input_layer,input_length,target_size= target_size,name ='main') 
    branch_main = core_cnn_shared(inputs_main)

    
    pred_s1_r = predictions_branch(branch_main,dense_units)    
    pred_t1_i = predictions_branch(branch_main,dense_units)
    pred_r = predictions_branch(branch_main,dense_units)
    pred_i = predictions_branch(branch_main,dense_units)
    
    output_s1_r = Softmax(name = 'output_s1_r')(pred_s1_r)
    outputs['output_s1_r'] = output_s1_r

    output_t1_i = Softmax(name = 'output_t1_i')(pred_t1_i)
    outputs['output_t1_i'] = output_t1_i
    
    output_i = Softmax(name = 'output_i')(pred_i)
    outputs['output_i'] = output_i
    
    output_r = Softmax(name = 'output_r')(pred_r)
    outputs['output_r'] = output_r    
    
    
    for k , v in outputs.items():
        weights[k] = 1
        losses[k] = 'categorical_crossentropy'
    
    

    model = Model(inputs = input_dict,outputs = outputs,name='cnn_multi_task_multi_target')
    optimizer = Adam(learning_rate=learning_rate)   
    
    model.compile(loss=losses, optimizer=optimizer,loss_weights=weights,metrics=['accuracy'])
    model.summary()
 
    return model





def cnn_hierarchical_multi_target(input_length=250000, learning_rate=0.001, classes=256, dense_units=200,timepoints = None,input_layer = "weighted_pooling"):
    inputs = Input(shape=(input_length, 1),name = 'traces')
    plaintexts = Input(shape = (256,),name = 'plaintext')
    target_size = 25000
    input_dict = {'traces' : inputs, 'plaintext':plaintexts}
    losses = {}
    weights = {}
    outputs = {}      
            
    inputs_main = input_layer_creation(inputs, input_layer,input_length,target_size= target_size,name ='main') 
    branch_main = core_cnn_shared(inputs_main)

    
    pred_s1_r = predictions_branch(branch_main,dense_units)    
    pred_t1_i = predictions_branch(branch_main,dense_units)
    pred_r = predictions_branch(branch_main,dense_units)
    pred_i = predictions_branch(branch_main,dense_units)
    
    output_s1_r = Softmax(name = 'output_s1_r')(pred_s1_r)
    outputs['output_s1_r'] = output_s1_r

    output_t1_i = Softmax(name = 'output_t1_i')(pred_t1_i)
    outputs['output_t1_i'] = output_t1_i
    
    output_i = Softmax(name = 'output_i')(pred_i)
    outputs['output_i'] = output_i
    
    output_r = Softmax(name = 'output_r')(pred_r)
    outputs['output_r'] = output_r    
         


    xor_s1_fixed_r = XorLayer(name = 'xor_s1_fixed_r')([pred_s1_r,output_r])
    xor_s1_fixed_s1_r = XorLayer(name = 'xor_s1_fixed_s1_r')([output_s1_r,pred_r])


    xor_t1_fixed_i = XorLayer(name = 'xor_t1_fixed_i')([pred_t1_i,output_i])
    xor_t1_fixed_t1_i = XorLayer(name = 'xor_t1_fixed_t1_i')([output_t1_i,pred_i])
    
    pred_s1_from_r =  Add_Shares(name = 'Add_shares_to_get_s1',shares = 2,input_dim = classes,units = classes)([xor_s1_fixed_r,xor_s1_fixed_s1_r])  
    output_s1 = Softmax(name = 'output_s1')(pred_s1_from_r)
    outputs['output_s1'] = output_s1
    
    inv_sbox_t1_r = InvSboxLayer(name = 'InvSbox_s1r_t1r')(pred_s1_from_r)
    
    pred_t1_from_i =  Add_Shares(name = 'Add_shares_to_get_t1',shares = 2,input_dim = classes,units = classes)([xor_t1_fixed_i,xor_t1_fixed_t1_i])  
    output_t1 = Softmax(name = 'output_t1')(pred_t1_from_i)
    outputs['output_t1'] = output_t1    
    
    

    #pred_t1 = Dense(256)(pred_t1)
    final_addition = [pred_t1_from_i,inv_sbox_t1_r]
    pred_t1 =  Add_Shares(name = 'Add_shares_t1',shares = 2,input_dim = classes,units = classes)(final_addition)


    pred_output = XorLayer(name = 'final_xor')([pred_t1,plaintexts])
    output = Softmax(name = 'output')(pred_output)
    outputs['output'] = output 
    
    
    for k , v in outputs.items():
        weights[k] = 1
        losses[k] = 'categorical_crossentropy'
    
    

    model = Model(inputs = input_dict,outputs = outputs,name='cnn_hierarchical_multi_target')
    optimizer = Adam(learning_rate=learning_rate)   
    
    model.compile(loss=losses, optimizer=optimizer,loss_weights=weights,metrics=['accuracy'])
    model.summary()
 
    return model

#### Training high level function
def train_model(training_type,variable,intermediate,multi_target):
    epochs = 100
    batch_size = 250
    n_traces = 50000
    
    if training_type =='classical':
        X_profiling , validation_data = load_dataset(variable,intermediate,n_traces = n_traces)
        model_t = 'cnn_best' 
    elif training_type == 'multi':
        X_profiling , validation_data = load_dataset_multi(VARIABLE_LIST[intermediate].index(variable),n_traces = n_traces,multi_target = multi_target,dataset = 'training') 
        if multi_target:
            model_t = 'cnn_multi_task_multi_target'
        else:
            model_t = 'cnn_multi_task_subbytes_inputs'
    else:
        X_profiling , validation_data = load_dataset_hierarchical(VARIABLE_LIST[intermediate].index(variable),n_traces = n_traces,multi_target = multi_target,dataset = 'training') 
        if multi_target:
            model_t = 'cnn_hierarchical_multi_target'
        else:
            model_t = 'cnn_hierarchical_subbytes_inputs'
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    
    if model_t == "cnn_best" :
        model = cnn_best(input_length =window ,name = intermediate.replace('^','_') if '^' in intermediate else intermediate)
    elif model_t == 'cnn_multi_task_multi_target':
        model = cnn_multi_task_multi_target()        
    elif model_t == 'cnn_multi_task_subbytes_inputs':
        model = cnn_multi_task_subbytes_inputs()   
    elif model_t == 'cnn_hierarchical_multi_target':
        model = cnn_hierarchical_multi_target()        
    elif model_t == 'cnn_hierarchical_subbytes_inputs':
        model = cnn_hierarchical_subbytes_inputs()                                   
    else:
        print('Some error here')

    model_t = model.name
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    monitor = 'val_accuracy'
    if  training_type == 'multi':
        monitor = 'val_loss'
    if training_type == 'propagation':
        monitor = 'val_output_accuracy'
    file_name = '{}_{}'.format( variable ,model_t) 
    print(file_name)
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                filepath= MODEL_FOLDER+ file_name+'.h5',
                                save_weights_only=True,
                                monitor=monitor,
                                mode='max' if not training_type == 'multi' else 'min',
                                save_best_only=True)

    

    
    history = model.fit(X_profiling, batch_size=batch_size, verbose = 1, epochs=epochs, validation_data=validation_data,callbacks =callbacks)
    print('Saved model {} ! '.format(file_name))
 
    file = open(METRICS_FOLDER+'history_training_'+(file_name ),'wb')
    pickle.dump(history.history,file)
    file.close()

    
    
    return model






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')
    parser.add_argument('--CLASSICAL', action="store_true", dest="CLASSICAL",
                        help='Classical training of the intermediates', default=False)
    parser.add_argument('--MULTI',   action="store_true", dest="MULTI", help='Adding the masks to the labels', default=False)
    parser.add_argument('--HIERARCHICAL',   action="store_true", dest="HIERARCHICAL", help='Adding the masks to the labels', default=False)
    parser.add_argument('--MULTI_TARGET',   action="store_true", dest="MULTI_TARGET", help='Adding the masks to the labels', default=False)
    args            = parser.parse_args()
  

    HIERARCHICAL        = args.HIERARCHICAL
    CLASSICAL        = args.CLASSICAL
    MULTI = args.MULTI
    MULTI_TARGET = args.MULTI_TARGET
    TARGETS = {}
    if CLASSICAL:   
       training_types = ['classical']
       TARGETS['classical'] = ['t1^r','s1^r','r','i','t1^ri','t1^i'] 
       BYTES = [i for i in range(2,16)]
    elif MULTI:
        training_types = ['multi']
        TARGETS['multi'] = ['t1'] if not MULTI_TARGET else ['k1']
        BYTES = [i for i in range(2,16)]

    elif HIERARCHICAL:
        training_types = ['hierarchical']
        TARGETS['hierarchical'] = ['t1'] if not MULTI_TARGET else ['k1']
        BYTES = [i for i in range(2,16)]
        #BYTES = [4,5,6,7,9,10,11,12,13,15,16]
    
    else:
        print('No training mode selected')
        
    

    for training_type in training_types:
        for TARGET in TARGETS[training_type]:
            for BYTE in BYTES:
                for target_byte in VARIABLE_LIST[TARGET]:
                    if not BYTE == VARIABLE_LIST[TARGET].index(target_byte) + 1 and not len( VARIABLE_LIST[TARGET]) == 1:    
                        continue
                    process_eval = Process(target=train_model, args=(training_type,target_byte ,TARGET,MULTI_TARGET))
                    process_eval.start()
                    process_eval.join()

    print("$ Done !")
            
        
        
