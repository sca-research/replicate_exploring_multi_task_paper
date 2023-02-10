# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:21:56 2022

@author: martho
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:34:05 2022

@author: martho
"""
########################################
# Paths to dataset and folder have to be changed






import pickle
import sys
import numpy as np



dict_parameters = {}

dict_parameters['KEY'] = 0x00112233445566778899AABBCCDDEEFF 

########### PATHS & FILE MANAGEMENT ######################################
DATASET_FOLDER= 'D:/dataset_masked_AES/dataset_ASCADv2/'  if sys.platform == 'win32' else '/srv/datasets/dataset_ASCAD_v2/'
FILE_DATASET = 'Ascad_v1_dataset.h5'
PROJECT_FOLDER = 'C:/Users/martho/Documents/replicate_multi_task_paper/ascadv2/'  if sys.platform == 'win32' else '/home/martho/Projets/replicate_multi_task_paper/ascadv2/'

dict_parameters['DATASET_FOLDER'] = DATASET_FOLDER
dict_parameters['FILE_DATASET'] = FILE_DATASET
dict_parameters['PROJECT_FOLDER'] = PROJECT_FOLDER

###########################################################################

############### MASKING SCHEME ############################################


## Attacked intermediate states, '1' for first round

MASKS = ['rin','alpha','beta','m','p','beta_mj','rin_mj','mj']
INTERMEDIATES= []
MASK_INTERMEDIATES = {}
for i in range(1,2):
    INTERMEDIATES.append('s{}'.format(i))
    INTERMEDIATES.append('t{}'.format(i))
    INTERMEDIATES.append('k{}'.format(i))
    
    MASK_INTERMEDIATES['s{}'.format(i)] = ['beta_mj','beta','mj']
    MASK_INTERMEDIATES['t{}'.format(i)] = ['rin_mj','rin','mj']
    MASK_INTERMEDIATES['k{}'.format(i)] = ['alpha']
MASKED_INTERMEDIATES = {}
VARIABLE_LIST = {}


for intermediate in INTERMEDIATES:
    print(intermediate)
    round_k = int(intermediate[1:]) - 1
    name_intermediate = intermediate[0]
    VARIABLE_LIST[intermediate] = [ name_intermediate + '0'+ ('0'+str(i) if i < 10 else '' + str(i)) for i in range(1 + 16 * round_k , 17 + 16 * round_k )  ]
    for mask in MASK_INTERMEDIATES[intermediate]:
        MASKED_INTERMEDIATES[intermediate + '^' + mask ] = [(x + '^' + mask) for x in VARIABLE_LIST[intermediate]]
        
        
        
########################################################################                        

for intermediate in MASKED_INTERMEDIATES:
    VARIABLE_LIST[intermediate ] = [x for x in MASKED_INTERMEDIATES[intermediate]]
    INTERMEDIATES.append(intermediate  )
    
INTERMEDIATES += MASKS
for mask in MASKS:
    VARIABLE_LIST[mask ] = [mask] if mask == 'rin' or mask == 'alpha' or mask == 'beta' else ['{}0{}'.format(mask,i+1 if i >= 9 else '0' + str(i+1)) for i in range(16)]
    


print(VARIABLE_LIST)

dict_parameters['INTERMEDIATES'] = INTERMEDIATES
dict_parameters['VARIABLE_LIST'] = VARIABLE_LIST
dict_parameters['MASKS']= MASKS
dict_parameters['ONE_MASKS']= MASKS
dict_parameters['MASK_INTERMEDIATES'] = MASK_INTERMEDIATES
dict_parameters['MASKED_INTERMEDIATES']= MASKED_INTERMEDIATES

file = open('dataset_parameters','wb')
pickle.dump(dict_parameters,file)
file.close()