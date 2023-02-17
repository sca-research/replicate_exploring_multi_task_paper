# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:32:19 2022

@author: martho
"""

from utility import XorLayer, MultiLayer
from utility import load_model_hierarchical, load_model_from_target, load_model_multi_task, read_from_h5_file  ,get_hot_encode, METRICS_FOLDER
from utility import get_rank, get_pow_rank

from gmpy2 import mpz,mul
from tqdm import tqdm
from train_models import cnn_best,cnn_multi_task,cnn_hierarchical

import numpy as np
import tensorflow as tf
import pickle 
import argparse
seed = 7


tf.random.set_seed(seed)
np.random.seed(seed)




class Attack:
    def __init__(self,n_experiments = 1,individual = False,multi = False,hierarchical = False,target = 't'):
        
        self.models = {}
        self.individual = individual
        self.multi = multi
        self.hierarchical = hierarchical
        self.target = target
        if individual:
            model_struct =  cnn_best(input_length=4749,name = 'p')
            self.models['permutation'] = load_model_from_target(model_struct,'p') 
            
            if target == 's':
                model_struct =  cnn_best(input_length=4749)
                self.models['beta'] =  load_model_from_target(model_struct,'beta') 
                model_struct_intermediate =  cnn_best(input_length=200,name = '_') 
                self.models['s1_beta'] = load_model_from_target(model_struct_intermediate,'s1^beta') 
                
            else:
                model_struct =  cnn_best(input_length=4749)
                self.models['beta'] =  load_model_from_target(model_struct,'beta') 
                model_struct_intermediate =  cnn_best(input_length=4749,name = '_') 
                self.models['s1_beta'] = load_model_from_target(model_struct_intermediate,'s1^beta')        
                model_struct_intermediate =  cnn_best(input_length=4749,name = '_') 
                self.models['t1_rin'] = load_model_from_target(model_struct_intermediate,'t1^rin') 
                model_struct =  cnn_best(input_length=4749)
                self.models['rin'] = load_model_from_target(model_struct,'rin') 
        elif multi:
            model_struct_propagation = cnn_multi_task()
            self.models['multi'] = load_model_multi_task(model_struct_propagation)
        elif hierarchical:
            model_struct_propagation = cnn_hierarchical()
            self.models['hierarchical'] = load_model_hierarchical(model_struct_propagation)
        else:
            print('Im confused, you didnt chose a model type --INDIV, --MULTI, --HIERARCHICAL')
            return
        self.n_experiments = n_experiments
        self.powervalues = {}

        mapping = (
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
        
        traces , labels_dict, metadata = read_from_h5_file(n_traces = 200000,dataset = 'attack',load_plaintexts = True)
        self.correct_guesses = {}
        self.history_score = {}
        self.traces_per_exp = 100
        self.n_total_attack_traces = 200000
        self.predictions = np.zeros((16,self.n_total_attack_traces,256))
        predictions_non_permuted = np.empty((16,self.n_total_attack_traces,256))
        predictions_permutation = np.empty((16,self.n_total_attack_traces,16))

        predictions_t1_rin = np.empty((16,self.n_total_attack_traces,256))
        predictions_s1_beta = np.empty((16,self.n_total_attack_traces,256))
        predictions_rin = np.empty((self.n_total_attack_traces,256))
        predictions_beta = np.empty((self.n_total_attack_traces,256))
        predictions_alpha = np.empty((self.n_total_attack_traces,256))
            
        plaintexts = metadata['plaintexts']
        
        self.key = 0x00112233445566778899AABBCCDDEEFF
        master_key =[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                          0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF ]  
        
        self.alpha = np.array(labels_dict['alpha'],dtype = np.uint8)[:self.n_total_attack_traces]
        self.permutations = np.array(labels_dict['p'],dtype = np.uint8)[:self.n_total_attack_traces]
        self.plaintexts = plaintexts
        # encoded_plaintexts = np.array([get_hot_encode(self.plaintexts[:,byte]) for byte in range(16)],dtype = np.float32)
        # encoded_plaintexts = np.swapaxes(encoded_plaintexts, 1, 0)
        batch_size = 1000
        powervalues_common = traces[:,:3150]
        for byte in range(16):
            self.powervalues[byte] = np.concatenate([powervalues_common,traces[:,3150 + byte * (4749 - 3150):3150 + (byte+1) * (4749 - 3150) ]],axis = 1)

            
        for batch in tqdm(range(self.n_total_attack_traces// batch_size)):
            for byte in range(16):                  
                
                
                if not individual  and not multi:
                    all_predictions = self.models['hierarchical'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)] },verbose=0 ,batch_size = 250)    
                    predictions_permutation[byte,batch_size*batch:batch_size*(batch +1)] = all_predictions['output_permutation']
                    predictions_non_permuted[byte,batch_size*batch:batch_size*(batch +1)] = all_predictions['output']
                    
                elif multi:
                    all_predictions = self.models['multi'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)] },verbose=0 ,batch_size = 250)
                    if byte == 0:
                        predictions_alpha[batch_size*batch:batch_size*(batch +1)] = all_predictions['output_alpha']
                        predictions_rin[batch_size*batch:batch_size*(batch +1)] = all_predictions['output_rin']
                        predictions_beta[batch_size*batch:batch_size*(batch +1)] = all_predictions['output_beta']
                    predictions_t1_rin[byte,batch_size*batch:batch_size*(batch +1)] = all_predictions['output_t1_rin']
                    predictions_s1_beta[byte,batch_size*batch:batch_size*(batch +1)] = all_predictions['output_s1_beta']
    
                    predictions_permutation[byte,batch_size*batch:batch_size*(batch +1)] = all_predictions['output_permutation']
                    
                    predictions_t1_before_alpha = XorLayer()([predictions_t1_rin[byte,batch_size*batch:batch_size*(batch +1)],predictions_rin[batch_size*batch:batch_size*(batch +1)]])
                    predictions_s1_before_alpha = XorLayer()([predictions_s1_beta[byte,batch_size*batch:batch_size*(batch +1)],predictions_beta[batch_size*batch:batch_size*(batch +1)]])
                    predictions_t1 = MultiLayer()([predictions_t1_before_alpha,predictions_alpha[batch_size*batch:batch_size*(batch +1)]])
                    predictions_s1 = MultiLayer()([predictions_s1_before_alpha,predictions_alpha[batch_size*batch:batch_size*(batch +1)]])
                    predictions_t1_from_s1 = predictions_s1[:,mapping]
                    predictions_sum_t1 = predictions_t1_from_s1 + predictions_t1
                    # expand = tf.expand_dims(all_predictions['output_permutation'], 2)
                    # permuted_plaintext = tf.reduce_sum(tf.multiply(expand,encoded_plaintexts),axis = 1)
                    predictions_non_permuted[byte,batch_size*batch:batch_size*(batch +1)] = predictions_t1_from_s1 + predictions_t1              
                else:
                    if byte == 0:
                        if self.target == 't' or self.target == 'k':
                            predictions_rin[batch_size*batch:batch_size*(batch +1)] = self.models['rin'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)]},verbose=0)['output']
                        if self.target == 's' or self.target == 'k':
                            predictions_beta[batch_size*batch:batch_size*(batch +1)] = self.models['beta'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)]},verbose=0)['output']
                    predictions_permutation[byte,batch_size*batch:batch_size*(batch +1)] = self.models['permutation'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)]},verbose=0)['output']
                                        
                    if self.target == 's':
                    
                        predictions_s1_beta[byte,batch_size*batch:batch_size*(batch +1)] = self.models['s1_beta'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1),3150:3350]},verbose=0)['output']
                        predictions_s1_before_alpha = XorLayer()([predictions_s1_beta[byte,batch_size*batch:batch_size*(batch +1)],predictions_beta[batch_size*batch:batch_size*(batch +1)]])
                        predictions_s1 = MultiLayer()([predictions_s1_before_alpha,get_hot_encode(self.alpha[batch_size*batch:batch_size*(batch +1)])])
                        predictions_t1_from_s1 = predictions_s1[:,mapping]
                        predictions_sum_t1 = predictions_t1_from_s1 
                    else:
                        predictions_s1_beta[byte,batch_size*batch:batch_size*(batch +1)] = self.models['s1_beta'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)]},verbose=0)['output']
                        predictions_s1_before_alpha = XorLayer()([predictions_s1_beta[byte,batch_size*batch:batch_size*(batch +1)],predictions_beta[batch_size*batch:batch_size*(batch +1)]])
                        predictions_s1 = MultiLayer()([predictions_s1_before_alpha,get_hot_encode(self.alpha[batch_size*batch:batch_size*(batch +1)])])
                        predictions_t1_from_s1 = predictions_s1[:,mapping]
                        
                        predictions_t1_rin[byte,batch_size*batch:batch_size*(batch +1)] = self.models['t1_rin'].predict({'traces':self.powervalues[byte][batch_size*batch:batch_size*(batch +1)]},verbose=0)['output']
                        predictions_t1_before_alpha = XorLayer()([predictions_t1_rin[byte,batch_size*batch:batch_size*(batch +1)],predictions_rin[batch_size*batch:batch_size*(batch +1)]])
                        predictions_t1 = MultiLayer()([predictions_t1_before_alpha,get_hot_encode(self.alpha[batch_size*batch:batch_size*(batch +1)])])                
                        predictions_sum_t1 = predictions_t1_from_s1 + predictions_t1
                    # expand = tf.expand_dims(predictions_permutation[byte], 2)
                    # permuted_plaintext = tf.reduce_sum(tf.multiply(expand,encoded_plaintexts),axis = 1)                
                    predictions_non_permuted[byte,batch_size*batch:batch_size*(batch +1)] = predictions_sum_t1
                
                
                
        
            for byte in range(16):
                for byte_perm in range(16):
                    self.predictions[byte_perm][batch_size*batch:batch_size*(batch +1)] = tf.add(self.predictions[byte_perm,batch_size*batch:batch_size*(batch +1)], tf.expand_dims(predictions_permutation[byte,batch_size*batch:batch_size*(batch +1),byte_perm],1) * predictions_non_permuted[byte,batch_size*batch:batch_size*(batch +1)] ) 
        
        for batch in tqdm(range(self.n_total_attack_traces// batch_size)):
            for byte in range(16):                   
        
                self.predictions[byte][batch_size*batch:batch_size*(batch +1)] = XorLayer()([self.predictions[byte,batch_size*batch:batch_size*(batch +1)],get_hot_encode(self.plaintexts[batch_size*batch:batch_size*(batch +1),byte])])
        
        master_key = np.array(master_key,dtype = np.int32)
        self.subkeys = master_key
        
        


        
    def run(self):
       
       for experiment in range(self.n_experiments):
           print('====================')
           print('Experiment {} '.format(experiment))
           self.history_score[experiment] = {}
           self.history_score[experiment]['total_rank'] =  [] 
           self.subkeys_guess = {}
           for i in range(16):
               self.subkeys_guess[i] = np.zeros(256,)            
           
               self.history_score[experiment][i] = []
           traces_order = np.random.permutation(self.n_total_attack_traces)[:self.traces_per_exp] 
           count_trace = 1
           
           for trace in traces_order:
               
               
               
               recovered  = {}
               all_recovered = True
               ranks = {}

               print('========= Trace {} ========='.format(count_trace))
               rank_string = ""
               total_rank = mpz(1)
               
               
               for byte in range(16):
                   self.subkeys_guess[byte] += np.log(self.predictions[byte][trace] + 1e-36)
                  
                   ranks[byte] = get_rank(self.subkeys_guess[byte],self.subkeys[byte])
                   self.history_score[experiment][byte].append(ranks[byte])
                   total_rank = mul(total_rank,mpz(ranks[byte]))
                   rank_string += "| rank for byte {} : {} | \n".format(byte,ranks[byte])
                   if np.argmax(self.subkeys_guess[byte]) == self.subkeys[byte]:
                       recovered[byte] = True                        
                   else:
                       recovered[byte] = False
                       all_recovered = False                
              
               self.history_score[experiment]['total_rank'].append(get_pow_rank(total_rank))
               print(rank_string)
               print('Total rank 2^{}'.format( self.history_score[experiment]['total_rank'][-1]))
               print('\n')
               if all_recovered:                    
                   print('All bytes Recovered at trace {}'.format(count_trace))
                   
                   for elem in range(count_trace,self.traces_per_exp):
                       for i in range(16):
                           self.history_score[experiment][byte].append(ranks[byte])
                       self.history_score[experiment]['total_rank'].append(1)
                   break
                   count_trace += 1
               else:
                   count_trace += 1
               print('\n')
       array_total_rank = np.empty((self.n_experiments,self.traces_per_exp))
       for i in range(self.n_experiments):
           for j in range(self.traces_per_exp):
               array_total_rank[i][j] =  self.history_score[i]['total_rank'][j] 
       whe = np.where(np.mean(array_total_rank,axis=0) < 2)[0]
       print('The full key is recovered on average at trace (Guessing entropy < 2) : ',(np.min(whe) if whe.shape[0] >= 1 else 50))                   
       if self.individual:       
           typ = 'indiv' 
       elif self.hierarchical:
           typ = 'hierarchical' 
       elif self.multi :
           typ = 'multi' 
       else :
           typ = 'typ'
       file = open(METRICS_FOLDER + 'history_attack_experiments_on_{}_{}_{}_{}'.format(self.target,typ,self.n_experiments,'weighted'),'wb')
       pickle.dump(self.history_score,file)
       file.close()

            
                
                
                    
        # file = open('history_attack_experiments_{}'.format(self.n_experiments),'wb')
        # pickle.dump(self.history_score,file)
        # file.close()
                
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains Neural Network Models')

    parser.add_argument('-e', '-experiment', action="store", dest="EXPERIMENT", help='Number of Epochs in Training (default: 75 CNN, 100 MLP)',
                        type=int, default=1000)
    parser.add_argument('--INDIV', action="store_true", dest="INDIV", help='for attack dataset', default=False)
    parser.add_argument('--MULTI', action="store_true", dest="MULTI", help='for attack dataset', default=False)
    parser.add_argument('--HIERARCHICAL', action="store_true", dest="HIERARCHICAL", help='for attack dataset', default=False)
    parser.add_argument('-t', action="store", dest="TARGET", help='Number of Epochs in Training (default: 75 CNN, 100 MLP)',
                        type=str, default='k')
    
    args            = parser.parse_args()
    
    


    EXPERIMENT = args.EXPERIMENT
    INDIV = args.INDIV
    MULTI = args.MULTI
    HIERARCHICAL = args.HIERARCHICAL
    TARGET = args.TARGET
    
    

    attack = Attack(n_experiments = EXPERIMENT,individual= INDIV,multi = MULTI,hierarchical = HIERARCHICAL,target = TARGET)
    attack.run()
                  
                            
            
            
    
    
            
            
        
        