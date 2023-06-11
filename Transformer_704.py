# -*- coding: utf-8 -*-

experiment = ''

#%% Library

#Expression
from argparse import  Namespace

# Neural Network
import torch

# Directory
from datetime import date
import os

os.chdir('G:/其他電腦/MacBook Pro/Researches/_CODE_')
from preprocess2 import  execute_preprocess_phase2
from utils import domain_weightedAVG, domain_KQsampler
from utils import avg, remove_outlier_IQR
from vm import dataloader_init, initialization, pureVM_main, mismatch_test
from adaptation import fineTune_test, transferLearning, multitask_fineTune, KQ_fineTuneTest
from meta import metaDomains_init, meta_init, meta_main, meta_fineTuneTest

DB = 'G:/其他電腦/MacBook Pro/PHM Data Challenge 2016 (phm_cmp_removal_rates)'
recordsDB = 'G:/其他電腦/MacBook Pro/Researches/Model Records/'
os.chdir(DB)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%%# Hyperparmaeters
# TODO: hehe
args = Namespace(
    
    model = 'Transformer', 
    #experiment = 'pureVM', # pureVM / fineTuneTest / multiTune / metaTune
    #experiment = 'vmK_allDomains',
    #experiment = 'vmK_multiExp',
    #experiment = 'fineTuneTest',
    #experiment = 'transferLearning',
    #experiment = 'multiTune',
    experiment = 'vm_multiExpTune',
    #experiment = 'metaTune',
    #experiment = 'meta_multiExpTune',
    no_leak = True,
    
    # Data
    preprocessedV = 'v2',

    # VM Model
    with_stats = False,
    stablize = False,
    dim_feedforward='input_dim',
    dim_reduction = 1,
    num_encoder_layers = 1,     # number of enocder block (nn.TransformerEncoderLayer)
    nhead = 3,                  # number of heads in nn.MultiheadAttention <- embed_dim must be divisible by num_heads
    dropout = 0.1,
    last_hidden = 'input_dim',
    act_fn = 'Sigmoid',
    set_target = 'bias',
    pos_encoding = False,

    # Optimization
    opt_type = 'Adam',          # 'SGD' / 'Adam' / 'AMSGrad' / 'Noam' / 'Adagrad'
    pureVM_lr= 1e-4,           
    weight_decay = 1e-3,   
    sgd_momentum = 1e-4,
    lr_regulator = False,     
    clip = False,
    
    # Pure VM Training
    batch_size=2,
    pureVM_numEpoch = 3000,
    pureVM_lossReduction = 'sum',
    
    #reload_datetime = '2022-07-05/15.47.18', # FS noPE DR1 multi T1
    #reload_datetime = '2022-07-05/15.50.07', # FS noPE DR1 multi T2
    reload_datetime = '2022-07-05/15.54.54', # FS noPE DR1 multi T3
    
    # Fine Tune Test
    pureVM_fineTune_numEpoch = 5000,
    fineTune_numExps = 30,
    fineTune_targetBias = True,
    noLeak_tuningDomain = 3,        # 1 / 2 / 3
    
    vm_K = 10,
    vm_Q = 10,
    vm_sampleMode = 'uniform', 
    
    # MAML
    meta_K = 20,
    meta_Q = 'all',
    numInterp = 0,
    num_addOrigDomains = 2,
    randFreq = 'virtTask',      # virtTask / metaBatch  
    meta_sampleMode = 'normal',      # normal / uniform
    metaTrain_initMean = 'kSample', # 'kSample' / 'target' 
    metaVal_initMean = 'target',# 'kSample' / 'target'
    base_lr = 1e-2,             # Within Task Step Size (Alpha)
    base_adaptSteps = 5,
    base_adaptSteps_test = 5,
    meta_lr = 1e-4,             # Meta Model Step Size (Beta)
    meta_wd = 1e-6,
    meta_numEpoch = 50 ,
    meta_lossReduction = 'mean',
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    
)
#%% multi

#reload_datetime = '2022-07-05/15.47.18', # FS noPE DR1 multi T1
#reload_datetime = '2022-07-05/15.50.07', # FS noPE DR1 multi T2
#reload_datetime = '2022-07-05/15.54.54', # FS noPE DR1 multi T3

#%% meta

#reload_datetime = '2022-07-05/22.36.15', # FS noPE DR1 meta T1
#reload_datetime = '2022-07-05/22.38.13', # FS noPE DR1 meta T2
#reload_datetime = '2022-07-05/22.34.03', # FS noPE DR1 meta T3


#%%
args_noam = Namespace(
    # Noam
    noam_factor=0.3,          # noam max lr tuning
    noam_warmup=10,           # noam max lr epoch
    adamW_lr=0,
    adamW_beta1= 0.9,
    adamW_beta2= 0.98,
    adamW_eps= 1e-9,
)


#%% Variable Groups

usageVars = ['USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER', 'USAGE_OF_POLISHING_TABLE', 
            'USAGE_OF_DRESSER_TABLE', 'USAGE_OF_MEMBRANE', 'USAGE_OF_PRESSURIZED_SHEET']

pressureVars = ['PRESSURIZED_CHAMBER_PRESSURE', 'MAIN_OUTER_AIR_BAG_PRESSURE', 'CENTER_AIR_BAG_PRESSURE', 
                'RETAINER_RING_PRESSURE', 'RIPPLE_AIR_BAG_PRESSURE', 'EDGE_AIR_BAG_PRESSURE']

slurryVars = ['SLURRY_FLOW_LINE_A', 'SLURRY_FLOW_LINE_B', 'SLURRY_FLOW_LINE_C']

rotationVars = ['WAFER_ROTATION', 'STAGE_ROTATION', 'HEAD_ROTATION']

oneHotVars = ['CHAMBER_1.0', 'CHAMBER_2.0', 'CHAMBER_3.0', 'CHAMBER_4.0', 'CHAMBER_5.0', 'CHAMBER_6.0', 
              'MACHINE_DATA_1', 'MACHINE_DATA_2', 'MACHINE_DATA_3', 'MACHINE_DATA_4', 'MACHINE_DATA_5', 'MACHINE_DATA_6']

oneHotVars_v6 = ['CHAMBER_1.0', 'CHAMBER_2.0', 'CHAMBER_3.0', 'CHAMBER_4.0', 'CHAMBER_5.0', 'CHAMBER_6.0']

oneHotVars = oneHotVars_v6 if args.preprocessedV == 'v6' else oneHotVars

categoricalVars = ['MACHINE_ID']

binaryVars = [ 'DRESSING_WATER_STATUS', 'STAGE_A']


continuousCols = usageVars + pressureVars + slurryVars + rotationVars

discreteVars = oneHotVars + categoricalVars + binaryVars

allVars = continuousCols + discreteVars

#%% Preprocess phase2 execute

Outlier_MRR_Threshold = 4000
resampleMode, ToLen = 'uniform', 340

Drop_Col_v2 = ['MACHINE_ID'] 
if args.model == 'Transformer' and args.nhead == 3:
    Drop_Col_v2 = []
    
Drop_Col_v3 = ['TIMESTAMP']
Drop_Col_v4 = ['TIMESTAMP']        # v5
Drop_Col_v6 = ['TIMESTAMP', 'MACHINE_ID', 'MACHINE_DATA']        # v6

if args.preprocessedV == 'v6':
    Drop_Col = Drop_Col_v6
elif args.preprocessedV == 'v3':
    Drop_Col = Drop_Col_v3
else:
    Drop_Col = Drop_Col_v2


statsList = ['mean', 'std']
ComputeCols = [i for i in allVars if i not in Drop_Col]

config_preprocess2 = {'Outlier_MRR_Threshold' : Outlier_MRR_Threshold,
                      'resampleMode': resampleMode,
                      'ToLen'       : ToLen,
                      'Drop_Col'    : Drop_Col,
                      'statsList'   : statsList,
                      'ComputeCols' : ComputeCols
                      }

all_datasets, all_datasets_df, all_datasets_list = execute_preprocess_phase2(args, config_preprocess2)

orig_datasets = {'df': all_datasets_df, 'list':  all_datasets_list}

[train_listed, val_listed, test_listed] = all_datasets
[support_1, query_1, test_1, support_2, query_2, test_2, support_3, query_3, test_3] = all_datasets_df
[support_1_list, query_1_list, test_1_list, support_2_list, query_2_list, test_2_list, support_3_list, query_3_list, test_3_list] = all_datasets_list

domain_means = [151.24, 73.08, 79.97]
domain_sizes = { 'Train': [364,798,815] , 'Valid': [67,185,172], 'Test': [73,165,186] }


#%% Execute pureVM_main

if args.experiment == 'pureVM':
    trainData, valData, testData = train_listed, val_listed, test_listed
    bias_init = domain_means[0]
    savingDir = recordsDB + str(date.today())
    
    testDatasets =  [ query_1 ,  query_2 ,  query_3 ,  test_1 ,  test_2 ,  test_3 ]
    testset_names = ['query_1', 'query_2', 'query_3', 'test_1', 'test_2', 'test_3']
    
    
    if __name__ == "__main__" :
    
        model, best_val_model, bestLosses = pureVM_main(
            
              args, trainData, valData, testData, savingDir, verbose= False, plotLearningCurve = True, bias_init=bias_init )
        
        mismatch_test(args, best_val_model, testDatasets, testset_names)

    
#%%

if args.experiment == 'vmK_allDomains' and __name__ == "__main__":
    
    savingDir = recordsDB + str(date.today())
    
    Taskset_1, Taskset_2, Taskset_3 = domain_KQsampler(orig_datasets).sampleKQ(args)
    
    vm_returns_1 = pureVM_main( 
        args, Taskset_1[0], Taskset_1[1], Taskset_1[2], savingDir, verbose= False, plotLearningCurve = True, bias_init = domain_means[0] )
    
    vm_returns_2 = pureVM_main( 
        args, Taskset_2[0], Taskset_2[1], Taskset_2[2], savingDir, verbose= False, plotLearningCurve = True, bias_init = domain_means[1] )
    
    vm_returns_3 = pureVM_main( 
        args, Taskset_3[0], Taskset_3[1], Taskset_3[2], savingDir, verbose= False, plotLearningCurve = True, bias_init = domain_means[2] )
    
    bestLossTrain = vm_returns_1[2][0], vm_returns_2[2][0], vm_returns_3[2][0]
    bestLossValid = vm_returns_1[2][1], vm_returns_2[2][1], vm_returns_3[2][1]
    bestLossTest  = vm_returns_1[2][2], vm_returns_2[2][2], vm_returns_3[2][2]
    
    print("\n- - - - - -【Dataset Total Loss】- - - - - -")
    temp = domain_weightedAVG( bestLossTrain, 'Train')
    temp = domain_weightedAVG( bestLossValid, 'Valid')
    temp = domain_weightedAVG( bestLossTest, 'Test' )

#%%

if __name__ == "__main__" and args.experiment == 'vmK_multiExp':
        
    savingDir = recordsDB + str(date.today())
    if not os.path.exists(savingDir):
        os.makedirs(savingDir)
    os.chdir(savingDir)
    
    train_loader, dev_loader, test_loader, args = dataloader_init(train_listed, val_listed, test_listed, args, verbose=True)
    model, best_val_model, optimizer, criterion = initialization(args, plot_scheduler = False, trained_model=None, bias_init=0)
    
    domain_KQsampler = domain_KQsampler(orig_datasets)
    exp_losses, exp_losses_desc = KQ_fineTuneTest(model, args, domain_KQsampler, savingDir)
    exp_losses2, exp_losses_desc2 = remove_outlier_IQR(exp_losses)

#%% Execute Fine Tune Test

if args.experiment == 'fineTuneTest':
    
    savingDir = recordsDB + str(date.today())
    
    Taskset_1, Taskset_2, Taskset_3 = domain_KQsampler(orig_datasets).sampleKQ(args)

    best_losses = {'train':[], 'valid':[], 'test':[]}
    
    print("- - - - - - - - -【TEST 1】- - - - - - - - -")
    ordered_tasks = {'name': ['Task 1'   , 'Task 2'   ,'Task 3'] , 'data': [ Taskset_1 ,  Taskset_2 , Taskset_3]  }
    domains_order = [1,2,3]
    m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2 = fineTune_test(ordered_tasks, domains_order, args, savingDir)
    best_losses['train'] += [bestLosses_main[0]]
    best_losses['valid'] += [bestLosses_main[1]]
    best_losses['test' ] += [bestLosses_main[2]]
    
    
    print("- - - - - - - - -【TEST 2】- - - - - - - - -")
    ordered_tasks = {'name': ['Task 2'   , 'Task 1'   ,'Task 3'] , 'data': [ Taskset_2 ,  Taskset_1 , Taskset_3]  }
    domains_order = [2,1,3]
    m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2 = fineTune_test(ordered_tasks, domains_order, args, savingDir)
    best_losses['train'] += [bestLosses_main[0]]
    best_losses['valid'] += [bestLosses_main[1]]
    best_losses['test' ] += [bestLosses_main[2]]
    
    print("- - - - - - - - -【TEST 3】- - - - - - - - -")
    ordered_tasks = {'name': ['Task 3'   , 'Task 1'   ,'Task 2'] , 'data': [ Taskset_3 ,  Taskset_1 , Taskset_2]  }
    domains_order = [3,1,2]
    m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2 = fineTune_test(ordered_tasks, domains_order, args, savingDir)
    best_losses['train'] += [bestLosses_main[0]]
    best_losses['valid'] += [bestLosses_main[1]]
    best_losses['test' ] += [bestLosses_main[2]]
    
    print("\n- - - - - -【Dataset Total Loss】- - - - - -")
    temp = domain_weightedAVG( best_losses['train'], 'Train')
    temp = domain_weightedAVG( best_losses['valid'], 'Valid')
    temp = domain_weightedAVG( best_losses['test' ], 'Test' )

#%%
if args.experiment == 'transferLearning':
    
    savingDir = recordsDB + str(date.today())
    
    orig_Tasks1 = [support_1_list, query_1_list, test_1_list]
    orig_Tasks2 = [support_2_list, query_2_list, test_2_list]
    orig_Tasks3 = [support_3_list, query_3_list, test_3_list]

    Taskset_1, Taskset_2, Taskset_3 = domain_KQsampler(orig_datasets).sampleKQ(args)

    best_losses = {'train':[], 'valid':[], 'test':[]}
    
    print("- - - - - - - - -【TEST 1】- - - - - - - - -")
    ordered_tasks = {'name': ['Task 1'   , 'Task 2'   ,'Task 3'] , 'data': [ orig_Tasks1 ,  Taskset_2 , Taskset_3]  }
    domains_order = [1,2,3]
    m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2 = transferLearning(ordered_tasks, domains_order, args, savingDir)
    best_losses['train'] += [bestLosses_main[0]]
    best_losses['valid'] += [bestLosses_main[1]]
    best_losses['test' ] += [bestLosses_main[2]]
    
    
    print("- - - - - - - - -【TEST 2】- - - - - - - - -")
    ordered_tasks = {'name': ['Task 2'   , 'Task 1'   ,'Task 3'] , 'data': [ orig_Tasks2 ,  Taskset_1 , Taskset_3]  }
    domains_order = [2,1,3]
    m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2 = transferLearning(ordered_tasks, domains_order, args, savingDir)
    best_losses['train'] += [bestLosses_main[0]]
    best_losses['valid'] += [bestLosses_main[1]]
    best_losses['test' ] += [bestLosses_main[2]]
    
    print("- - - - - - - - -【TEST 3】- - - - - - - - -")
    ordered_tasks = {'name': ['Task 3'   , 'Task 1'   ,'Task 2'] , 'data': [ orig_Tasks3 ,  Taskset_1 , Taskset_2]  }
    domains_order = [3,1,2]
    m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2 = transferLearning(ordered_tasks, domains_order, args, savingDir)
    best_losses['train'] += [bestLosses_main[0]]
    best_losses['valid'] += [bestLosses_main[1]]
    best_losses['test' ] += [bestLosses_main[2]]
    
    print("\n- - - - - -【Dataset Total Loss】- - - - - -")
    temp = domain_weightedAVG( best_losses['train'], 'Train')
    temp = domain_weightedAVG( best_losses['valid'], 'Valid')
    temp = domain_weightedAVG( best_losses['test' ], 'Test' )


#%% Execute Multi-task Fine Tune

if args.experiment == 'multiTune' :
    savingDir = recordsDB + str(date.today())
    
    Taskset_1, Taskset_2, Taskset_3 = domain_KQsampler(orig_datasets).sampleKQ(args)
    
    fullTasks = [[support_1_list , query_1_list , test_1_list], 
                 [support_2_list , query_2_list , test_2_list],
                 [support_3_list , query_3_list , test_3_list] ]

    Pooled_tasks = []
    ds = [0,1,2]
    if args.no_leak:
        ds.remove(args.noLeak_tuningDomain-1)
    for split in range(3):
        pooled_split = []
        for d in ds:
            pooled_split += fullTasks[d][split]
        Pooled_tasks += [pooled_split]
            
    ordered_tasks = {'name': ['Pooled' ,'Task 1'   , 'Task 2'   ,'Task 3'] , 'data': [ Pooled_tasks, Taskset_1 ,  Taskset_2 , Taskset_3]  }
    mPooled, m1, m2, m3, bestLosses_main, bestLosses_tune1, bestLosses_tune2, bestLosses_tune3 = multitask_fineTune(ordered_tasks, args, savingDir)
    
    bestLossTrain = bestLosses_tune1[0], bestLosses_tune2[0], bestLosses_tune3[0]
    bestLossValid = bestLosses_tune1[1], bestLosses_tune2[1], bestLosses_tune3[1]
    bestLossTest  = bestLosses_tune1[2], bestLosses_tune2[2], bestLosses_tune3[2]
    
    print("\n- - - - - -【Dataset Total Loss】- - - - - -")
    temp = domain_weightedAVG( bestLossTrain, 'Train')
    temp = domain_weightedAVG( bestLossValid, 'Valid')
    temp = domain_weightedAVG( bestLossTest, 'Test' )
    
    domain_KQsampler = domain_KQsampler(orig_datasets)
    exp_losses, exp_losses_desc = KQ_fineTuneTest(mPooled, args, domain_KQsampler, savingDir)
    
    exp_losses2, exp_losses_desc2 = remove_outlier_IQR(exp_losses)
    
#%% multiTune FFT

if __name__ == "__main__" and args.experiment == 'vm_multiExpTune':
        
    savingDir = recordsDB + str(date.today())
    if not os.path.exists(savingDir):
        os.makedirs(savingDir)
    os.chdir(savingDir)
    
    train_loader, dev_loader, test_loader, args = dataloader_init(train_listed, val_listed, test_listed, args, verbose=True)
    model, best_val_model, optimizer, criterion = initialization(args, plot_scheduler = False, trained_model=None, bias_init=0)
    
    best_val_model.load_state_dict(torch.load(recordsDB + args.reload_datetime + ' Best_valid_loss_state.pt'))
    
    domain_KQsampler = domain_KQsampler(orig_datasets)
    exp_losses, exp_losses_desc = KQ_fineTuneTest(best_val_model, args, domain_KQsampler, savingDir)
    
    exp_losses2, exp_losses_desc2 = remove_outlier_IQR(exp_losses)

#%% Execute L2L MAML without flags


if __name__ == "__main__" and args.experiment== 'metaTune':
    
    savingDir = recordsDB + str(date.today())
    
    with torch.backends.cudnn.flags(enabled=False):
        best_meta_model = meta_main(args= args, config_preprocess2= config_preprocess2, 
                                    orig_datasets=orig_datasets, savingDir= savingDir)
        
        domain_KQsampler = domain_KQsampler(orig_datasets)
        
        losses = meta_fineTuneTest(best_meta_model, args, domain_KQsampler, savingDir)
        
        exp_losses, exp_losses_desc = KQ_fineTuneTest(best_meta_model.module, args, domain_KQsampler, savingDir)
        
        exp_losses2, exp_losses_desc2 = remove_outlier_IQR(exp_losses)


#%% metaTune FFT

if __name__ == "__main__" and args.experiment == 'meta_multiExpTune':
    
    savingDir = recordsDB + str(date.today())
    if not os.path.exists(savingDir):
        os.makedirs(savingDir)
    os.chdir(savingDir)
        
    train_domains_DFs, val_domains_lists, test_domains_lists, args, orig_domain_means = metaDomains_init(
        orig_datasets, args, verbose= True)
    
    meta_model, best_meta_model, opt, loss_fn = meta_init(args)
    
    best_meta_model.load_state_dict(torch.load(recordsDB + args.reload_datetime + ' Best_valid_loss_state.pt'))
    
    domain_KQsampler = domain_KQsampler(orig_datasets)
    exp_losses, exp_losses_desc = KQ_fineTuneTest(best_meta_model.module, args, domain_KQsampler, savingDir)
    exp_losses2, exp_losses_desc2 = remove_outlier_IQR(exp_losses)

