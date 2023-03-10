# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:10:35 2021

@author: martho
"""


import argparse
import os
import numpy as np
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Add,Softmax,Average,AlphaDropout, Dense, Input,Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam

from multiprocessing import Process


# import dataset paths and variables
from utility import VARIABLE_LIST , METRICS_FOLDER , MODEL_FOLDER

# import custom layers
from utility import PoolingCrop, MultiLayer , XorLayer , InvSboxLayer, Add_Shares

from utility import load_dataset, load_dataset_multi ,load_dataset_hierarchical



seed = 7


tf.random.set_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

###########################################################################


### Base Architecture for subbytes inputs and output single-task models

def cnn_from_masure(inputs):
    x = Conv1D(kernel_size=16, strides=1, filters=11, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)    
    x = AveragePooling1D()(x)
    x = Conv1D(kernel_size=32, strides=1, filters=11, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)    
    x = AveragePooling1D()(x)
    x = Conv1D(kernel_size=64, strides=1, filters=11, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)    
    x = AveragePooling1D()(x)    
    x = Conv1D(kernel_size=128, strides=1, filters=11, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)    
    x = AveragePooling1D()(x)    
    x = Flatten()(x)
    x = Dense(2048, activation='relu', kernel_regularizer =  L1L2(0.0001))(x)
    x = BatchNormalization()(x)  
    x = Dense(256)(x)
    return x



### Single-Task Models

def cnn_best(input_length=1000, learning_rate=0.0001, classes=256, dense_units=200 , name ='',input_layer = 'classic'):
    inputs_dict = {}
    
    inputs  = Input(shape = (input_length,1) ,name = 'traces')
    inputs_dict['traces'] = inputs   


    x = resnet_core(inputs)
    x = predictions_branch(x,2,dense_units,name = name,permutation = name =='p')
     
    output = tf.keras.layers.Softmax(name = 'output')(x)
    
    outputs = {}
    outputs['output'] = output
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_best')

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
 
    return model



### Multi-Task Models


def cnn_multi_task(learning_rate=0.0001, classes=256, dense_units=200):
    
    inputs_dict = {}
    
    input_traces = Input(shape=(4749, 1),name = 'traces')
    inputs_dict['traces'] = input_traces


    preds = {}
    outputs = {}   
    weights = {}
    main_branch = resnet_core(input_traces,name = 'main_branch')
    targets_name  = ['alpha','beta','rin','t1_rin','s1_beta','permutation']
    for name in targets_name:

        pred = predictions_branch(main_branch,2,dense_units,name =name,permutation = name == 'permutation' )       
        preds['pred_{}'.format(name)] = pred
        
        output = Softmax(name = 'output_{}'.format(name))(pred)
        outputs['output_{}'.format(name)] = output
        



    losses = {}   
    



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy'
        weights[k] = 1 
    


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_multi_task')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'],loss_weights = weights)
    model.summary()
    return model   







### Hierarchical Multi-Task Models

def cnn_hierarchical(learning_rate=0.0001, classes=256, dense_units=200):
    
    inputs_dict = {}
    input_traces = Input(shape=(4749, 1),name = 'traces')
    inputs_dict['traces'] = input_traces

    preds = {}
    outputs = {}
    
    main_branch = resnet_core(input_traces,name = 'main_branch')
    targets_name  = ['alpha','beta','rin','t1_rin','s1_beta','permutation']
    
    for name in targets_name:

        pred = predictions_branch(main_branch,2, dense_units,name =name, permutation = name == 'permutation' )       
        preds['pred_{}'.format(name)] = pred
        
        
        output = Softmax(name = 'output_{}'.format(name))(pred)
        outputs['output_{}'.format(name)] = output


    
    couple_beta_fixed = [preds['pred_s1_beta'],outputs['output_beta']] 
    #
    #couple_s1_beta_fixed = [outputs['output_s1_beta'],preds['pred_beta']] 
    xor_beta_fixed  = XorLayer(name = 'XorLayer_beta_fixed')(couple_beta_fixed) 
    #xor_s1_beta_fixed  = XorLayer(name = 'XorLayer_s1_beta_fixed')(couple_s1_beta_fixed) 

    
    multi_s1_1 = MultiLayer(classes = classes,name = 'multi_s1_1')([xor_beta_fixed,outputs['output_alpha']])
    #multi_s1_2 = MultiLayer(classes = classes,name = 'multi_s1_2')([xor_s1_beta_fixed,outputs['output_alpha']])

    mult_t1_from_inv_sbox_1 = InvSboxLayer(name = 'inv_sbox_1')(multi_s1_1)
    #mult_t1_from_inv_sbox_2 = InvSboxLayer(name = 'inv_sbox_2')(multi_s1_2)


    couple_rin_fixed = [preds['pred_t1_rin'],outputs['output_rin']] 
    #couple_t1_rin_fixed = [outputs['output_t1_rin'],preds['pred_rin']]
    
    xor_rin_fixed  = XorLayer(name = 'XorLayer_rin_fixed')(couple_rin_fixed) 
    #xor_t1_rin_fixed  = XorLayer(name = 'XorLayer_t1_rin_fixed')(couple_t1_rin_fixed) 
    

    
    multi_t1_1 = MultiLayer(classes = classes,name = 'multi_t1_1')([xor_rin_fixed,outputs['output_alpha']])
    #multi_t1_2 = MultiLayer(classes = classes,name = 'multi_t1_2')([xor_t1_rin_fixed,outputs['output_alpha']])
    
    #pred_output = Add_Shares(name = 'Add_shares',shares = 2,input_dim = classes,units = classes)([mult_t1_from_inv_sbox_1,multi_t1_1])
    pred_output = Average()([mult_t1_from_inv_sbox_1,multi_t1_1])
    output = Softmax(name = 'output')(pred_output)
    outputs['output'] = output

    losses = {}   
    weights = {}



    for k , v in outputs.items():
        losses[k] = 'categorical_crossentropy' 
        weights[k] = 1 if not '_' in k else 0.25

    


    model = Model(inputs = inputs_dict,outputs = outputs,name='cnn_hierarchical')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=losses, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model   


### Resnet for shared layers and mask/permutations single task models.

def resnet_core(inputs_core,name = ''):
    
    ## First Block 
    x = Conv1D(kernel_size=16, strides=1, filters=16, activation='selu', padding='same')(inputs_core)    
    x = BatchNormalization()(x)
    x = AveragePooling1D(pool_size = 2)(x)
    x = Flatten()(x) 
    return x


##### Dense Layers for all models
def predictions_branch(input_branch,n_blocks,dense_units,name = '',reg = 0.0001,permutation = False):
    reg = reg
    x = input_branch
    for block in range(n_blocks):      
        x = Dense(dense_units, activation='selu')(x)   
    x = Dense(256 if not permutation else 16, name = 'pred_{}'.format(name))(x)
    return x

#### Training high level function
def train_model(training_type,variable,intermediate):
    epochs = 100 if not intermediate == 'alpha' else 5
    batch_size = 250
    n_traces = 250000
    
    if training_type =='classical':
        X_profiling , validation_data = load_dataset(variable,intermediate,n_traces = n_traces)
        model_t = 'cnn_best' 
    elif training_type == 'multi':
        X_profiling , validation_data = load_dataset_multi(n_traces = n_traces,dataset = 'training') 
        model_t = 'cnn_multi_task'
    else:
        X_profiling , validation_data = load_dataset_hierarchical(n_traces = n_traces,dataset = 'training') 
        model_t = 'cnn_hierarchical'
    window =  X_profiling.element_spec[0]['traces'].shape[0]
    
    if model_t == "cnn_best" :
        model = cnn_best(input_length =window ,name = intermediate.replace('^','_') if '^' in intermediate else intermediate)
    elif model_t == 'cnn_multi_task':
        model = cnn_multi_task()        
    elif model_t == 'cnn_hierarchical':
        model = cnn_hierarchical()                                   
    else:
        print('Some error here')

    model_t = model.name
    X_profiling = X_profiling.shuffle(len(X_profiling)).batch(batch_size) 
    validation_data = validation_data.batch(batch_size)
    monitor = 'val_accuracy'
    if  training_type == 'multi':
        monitor = 'val_loss'
    if training_type == 'hierarchical':
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
    parser.add_argument('--ALL',   action="store_true", dest="ALL", help='Adding the masks to the labels', default=False)
        
    args            = parser.parse_args()
  

    HIERARCHICAL        = args.HIERARCHICAL
    CLASSICAL        = args.CLASSICAL
    MULTI = args.MULTI
    ALL = args.ALL

    TARGETS = {}
    if CLASSICAL:   
       training_types = ['classical']
       TARGETS['classical'] = ['p','t1^rin','s1^beta','rin','alpha'] 
       BYTES = ['all']
    elif MULTI:
        training_types = ['multi']
        TARGETS['multi'] = ['t1']
        BYTES = ['all']

    elif HIERARCHICAL:
        training_types = ['hierarchical']
        TARGETS['hierarchical'] = ['t1']
        BYTES = ['all']
    elif ALL:
        training_types = ['multi','classical']
        TARGETS['hierarchical'] = ['t1']
        TARGETS['multi'] = ['t1']
        TARGETS['classical'] = ['p','t1^rin','s1^beta','rin','alpha','beta'] 
        BYTES = ['all']   
    else:
        print('No training mode selected')
        
    

    for training_type in training_types:
        for TARGET in TARGETS[training_type]:
            for BYTE in BYTES:
                for target_byte in VARIABLE_LIST[TARGET]:
                    if not BYTE == VARIABLE_LIST[TARGET].index(target_byte) + 1 and not len( VARIABLE_LIST[TARGET]) == 1 and not BYTE == 'all':    
                        continue
                    process_eval = Process(target=train_model, args=(training_type,target_byte if not BYTE == 'all' else 'all_{}'.format(TARGET),TARGET))
                    process_eval.start()
                    process_eval.join()
                    if BYTE == 'all':
                        break

    print("$ Done !")
            
        
        
