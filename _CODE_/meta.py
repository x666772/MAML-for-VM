# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import random
from random import sample
from tslearn.preprocessing import TimeSeriesResampler

# Visualization
import matplotlib.pyplot as plt
from tqdm import tqdm

#Expression
from argparse import  Namespace

# Neural Network
import torch
from torch import nn
from torch.utils.data import DataLoader

import learn2learn as l2l

# Directory
from datetime import datetime
import os
os.chdir('G:/其他電腦/MacBook Pro/Researches/_CODE_')
from preprocess2 import computeStatistics
from preprocess2 import splitDomain_restusture, check_vars
from utils import domain_weightedAVG,  updateLosses_modelCheckpoint, learning_curve
from utils import SeqDataset, batch_to_tensor

from models import LSTM_Regr, TCN_Regr, Transformer_Regr , GRU_Regr
from vm import pureVM_main,  NoamOpt

DB = 'G:/其他電腦/MacBook Pro/PHM Data Challenge 2016 (phm_cmp_removal_rates)'
recordsDB = 'G:/其他電腦/MacBook Pro/Researches/Model Records/'
os.chdir(DB)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%%
usageVars = ['USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER', 'USAGE_OF_POLISHING_TABLE', 
            'USAGE_OF_DRESSER_TABLE', 'USAGE_OF_MEMBRANE', 'USAGE_OF_PRESSURIZED_SHEET']

pressureVars = ['PRESSURIZED_CHAMBER_PRESSURE', 'MAIN_OUTER_AIR_BAG_PRESSURE', 'CENTER_AIR_BAG_PRESSURE', 
                'RETAINER_RING_PRESSURE', 'RIPPLE_AIR_BAG_PRESSURE', 'EDGE_AIR_BAG_PRESSURE']

slurryVars = ['SLURRY_FLOW_LINE_A', 'SLURRY_FLOW_LINE_B', 'SLURRY_FLOW_LINE_C']

rotationVars = ['WAFER_ROTATION', 'STAGE_ROTATION', 'HEAD_ROTATION']

oneHotVars = ['CHAMBER_1.0', 'CHAMBER_2.0', 'CHAMBER_3.0', 'CHAMBER_4.0', 'CHAMBER_5.0', 'CHAMBER_6.0', 
              'MACHINE_DATA_1', 'MACHINE_DATA_2', 'MACHINE_DATA_3', 'MACHINE_DATA_4', 'MACHINE_DATA_5', 'MACHINE_DATA_6']

oneHotVars_v6 = ['CHAMBER_1.0', 'CHAMBER_2.0', 'CHAMBER_3.0', 'CHAMBER_4.0', 'CHAMBER_5.0', 'CHAMBER_6.0']


categoricalVars = ['MACHINE_ID']

binaryVars = [ 'DRESSING_WATER_STATUS', 'STAGE_A']


continuousCols = usageVars + pressureVars + slurryVars + rotationVars

discreteVars = oneHotVars + categoricalVars + binaryVars

allVars = continuousCols + discreteVars

domain_means = [151.24, 73.08, 79.97]
domain_sizes = { 'Train': [364,798,815] , 'Valid': [67,185,172], 'Test': [73,165,186] }

#%%


def generateVirtualDomain(domain1:dict, domain2:dict, meanPair:list, size: int, ratio: float, args):
    
    if args.meta_sampleMode == 'normal':
        percentiles = random.randn(size) # smaple with N(0,1) shape
    elif args.meta_sampleMode == 'uniform':
        percentiles = random.uniform(size=size)
    else:
        raise Exception(f'Unprogrammed meta_sampleMode \'{args.meta_sampleMode}\'!')
        
    percentiles =  percentiles - min(percentiles) + random.uniform(0,0.1)
    percentiles =  percentiles / ( max(percentiles) + random.uniform(0,0.1))
    
    domain_v ={}
    domainMean = {}
    
    for split in ['sup', 'que', 'test']:
        
        d1_split_size, d2_split_size = domain1[split].shape[0], domain2[split].shape[0]
        
        ids1 = np.sort((percentiles * d1_split_size).astype(int))
        ids2 = np.sort((percentiles * d2_split_size).astype(int))
        
        ids1 = range(d1_split_size) if size >= d1_split_size else ids1
        ids2 = range(d2_split_size) if size >= d2_split_size else ids2
        
        svid_list, len_list, mrr_list = [],[],[]
        for k in range(size):
            
            if k >= min(d1_split_size, d2_split_size):
                continue
            
            iddm1, iddm2 = ids1[k], ids2[k]
            ts1 = domain1[split].iloc[iddm1]
            ts2 = domain2[split].iloc[iddm2]
            
            if args.model == 'Transformer':
                avglen = 340
            else:
                avglen = int((ts1['Len'] + ts2['Len'])/2)
                len_transformer = TimeSeriesResampler(sz=avglen)
            
            svid_v = pd.DataFrame()
            
            for col_name in ts1['SVID'].columns:
                
                if args.model == 'Transformer':
                    col_1 = ts1['SVID'][col_name].values      
                    col_2 = ts1['SVID'][col_name].values
                else:
                    col_1 = len_transformer.fit_transform(ts1['SVID'][col_name].values)[0,:,0]        
                    col_2 = len_transformer.fit_transform(ts2['SVID'][col_name].values)[0,:,0]
                
                col_v = []
                for i in range(avglen):
                    
                    value = col_1[i] * ratio + col_2[i] * (1-ratio)
                    value = round(value,0) if col_name in oneHotVars + binaryVars else value
                    col_v += [value]
                svid_v[col_name] = col_v
            
            mrr_v = ts1['AVG_REMOVAL_RATE'] * ratio + ts2['AVG_REMOVAL_RATE'] * (1-ratio)
            
            
            svid_list  += [svid_v]
            len_list  += [avglen]
            mrr_list  += [mrr_v]
            
        domain_v[split] = pd.DataFrame({'SVID': svid_list, 'Len': len_list, 'AVG_REMOVAL_RATE': mrr_list, 'ID':range(len(len_list))})
        if args.metaTrain_initMean == 'kSample':
            domainMean[split] = domain_v[split]['AVG_REMOVAL_RATE'].mean()
        elif args.metaTrain_initMean == 'target':
            domainMean[split] = meanPair[0] * ratio + meanPair[1] * (1-ratio) 

    return domain_v, domainMean
   
def generateMetaBatch( train_domains_DFs: list, orig_domain_means: list , args: Namespace , 
                      config_preprocess2: dict , restructure: bool ):
    
    metaBatch = []
    metaBatchMeans = []
    
    if len(train_domains_DFs) == 2 :
        
        domain_pairs = [train_domains_DFs]
        origMean_pairs = [orig_domain_means]
    
    elif len(train_domains_DFs) == 3 :
        [ domain_1, domain_2 , domain_3 ] = train_domains_DFs
        domain_pairs = [[domain_1,domain_2], [domain_2, domain_3], [domain_3, domain_1]]
        
        [ mean_1, mean_2, mean_3 ] = orig_domain_means
        origMean_pairs = [[mean_1,mean_2], [mean_2, mean_3], [mean_3, mean_1]]
        
    rs, count = random.random(args.numInterp), 1    
    pairCount = -1
    for [d1, d2] in domain_pairs :
        pairCount += 1
        meanPair = origMean_pairs[pairCount]
        
        for i in range(args.numInterp):
            
            print(f'\tGenterating Virtual Domain {count}')
            
            if args.randFreq == 'virtTask': 
                r = random.random()
            elif args.randFreq == 'metaBatch':
                r = rs[i]
            else:
                raise Exception(f'Unprogrammed randFreq \'{args.randFreq}\'!')
            
            domain_v, domainMean = generateVirtualDomain(domain1 = d1, domain2 = d2, meanPair= meanPair, size= args.meta_K, ratio= r, args=args)
            
            
            if args.with_stats:
                [support_v, query_v, test_v] , Stats_Size = computeStatistics(
                    grouped_datasets= domain_v.values() ,  computeCols= config_preprocess2['ComputeCols'] , 
                    stats_list= config_preprocess2['statsList'], TQDM=False)
                
                domain_v = {'sup': support_v, 'que': query_v, 'test': test_v}
            
            
            if restructure:
                
                [support_v, query_v, test_v] = domain_v.values()
                
                Nested_Datasets = [ { 'name': 'train', 'data': support_v}, { 'name': 'val'  , 'data': query_v  }, { 'name': 'test' , 'data': test_v }    ]
                
                GroupBy, Groups = None, None
                
                split_datasets = splitDomain_restusture(Nested_Datasets , GroupBy, Groups, restructure = True, verbose=False)
                
                [support_v_listed, query_v_listed, test_v_listed] = split_datasets.values()
                
                domain_v = {'sup': support_v_listed, 'que': query_v_listed, 'test': test_v_listed}
            
            metaBatch += [domain_v]
            metaBatchMeans += [domainMean]
            count += 1
    
    return metaBatch, metaBatchMeans

def add_orig_domain( metaBatch, metaBatchMeans, val_domains_lists, orig_domain_means, args):
    
    if args.num_addOrigDomains == 0:
        pass
    
    else:
        
        metaBatch += val_domains_lists
        
        for i in range(len(orig_domain_means)):
            metaBatchMeans += [{'sup': orig_domain_means[i]}]
         
        if args.num_addOrigDomains == 1:
            randID = random.randint(2) + args.numInterp
            metaBatch.pop(randID)
            metaBatchMeans.pop(randID)
        
        elif args.num_addOrigDomains == 2:
            pass
        else:
            raise Exception('args.num_addOrigDomains= ', args.num_addOrigDomains, ' not available! ')
    
    return metaBatch, metaBatchMeans

#%%

def meta_train(meta_model, loss_fn, optimizer, metaBatch, metaBatchMeans, args):

    meta_train_loss = []

    for taskID in range(len(metaBatch)):   # for each task in the batch
        print(f'\t( Batch Task {taskID+1} )')
        supset = SeqDataset(data= metaBatch[taskID]['sup'], withStats = args.with_stats)
        queset = SeqDataset(data= metaBatch[taskID]['que'], withStats = args.with_stats)
        suploader = DataLoader(supset, batch_size = args.meta_K, shuffle = False, collate_fn = supset.collate_fn )
        queloader = DataLoader(queset, batch_size = args.meta_K, shuffle = False, collate_fn = queset.collate_fn )
        
        supBatch  = next(iter(suploader))
        queBatch =  next(iter(queloader))
        
        baseLearner = meta_model.clone()
        
        baseLearner.module.out.bias.data.fill_(metaBatchMeans[taskID]['sup'])

        # divide the data into support and query sets
        padded, seq_len, stats, targets = batch_to_tensor(supBatch, args)
        
        # baseLearner adapt to support tasks
        for _ in range(args.base_adaptSteps): # adaptation_steps
        
            #print(padded.is_leaf, seq_len.is_leaf, stats.is_leaf, targets.is_leaf)
            
            mrr = baseLearner(padded, seq_len, stats)
            
            support_loss=loss_fn(mrr, targets)
            
            baseLearner.adapt(support_loss)
            
            print(f'\t\tadapt step {_+1} loss {support_loss:.3f}')
            

        # record loss from query tasks
        padded, seq_len, stats, targets = batch_to_tensor(queBatch, args)
        
        query_preds = baseLearner(padded, seq_len, stats)
        query_loss = loss_fn(query_preds, targets)
        meta_train_loss += [query_loss]
        print(f'\t\tadapted evaluation: {query_loss:.3f}')
    
    #meta_train_loss = domain_weightedAVG( meta_train_loss, 'Valid', verbose= False) if len(meta_train_loss)==3 else sum(meta_train_loss) / len(meta_train_loss)
    meta_train_loss = torch.stack(meta_train_loss).mean()
    
    # Meta Model update
    optimizer.zero_grad()
    meta_train_loss.backward()
    optimizer.step()

    return meta_train_loss.item()      
    
#%%
def meta_val(meta_model, loss_fn, val_domains_lists, orig_domain_means, args):

    meta_val_loss = []
    
    for taskID in range(len(val_domains_lists)):   # for each task in the batch
        print(f'\t( Batch Task {taskID+1} )')
        
        sup_data = val_domains_lists[taskID]['sup']
        que_data = val_domains_lists[taskID]['que'] 
        
        #sup_data = sup_data if args.meta_K == 'all' else sample(sup_data, args.meta_K )
        que_data = que_data if args.meta_Q == 'all' else sample(que_data, min(args.meta_Q, len(que_data) ))
        
        supset = SeqDataset(data= sup_data, withStats = args.with_stats)
        queset = SeqDataset(data= que_data, withStats = args.with_stats)
        suploader = DataLoader(supset, batch_size = args.meta_K, shuffle = True, collate_fn = supset.collate_fn )
        queloader = DataLoader(queset, batch_size = 1, shuffle = True, collate_fn = queset.collate_fn )
        
        supBatch  = next(iter(suploader))
        queiter = iter(queloader)
        
        baseLearner = meta_model.clone()
        
        if args.metaVal_initMean == 'kSample':
            setBias = sum(supBatch['MRR']) / len(supBatch['MRR'])
        elif args.metaVal_initMean == 'target':
            setBias = orig_domain_means[taskID]
        baseLearner.module.out.bias.data.fill_(setBias)

        # divide the data into support and query sets
        padded, seq_len, stats, targets = batch_to_tensor(supBatch, args)
        
        # baseLearner adapt to support tasks
        for _ in range(args.base_adaptSteps): # adaptation_steps
            
            mrr = baseLearner(padded, seq_len, stats)
            support_loss=loss_fn(mrr, targets)
            baseLearner.adapt(support_loss)
            
            print(f'\t\tadapt step {_+1} loss {support_loss:.3f}')
        
        queTaskLoss = 0.0
        for batch in range(len(queloader)):
            queBatch =  next(queiter)
            # record loss from query tasks
            padded, seq_len, stats, targets = batch_to_tensor(queBatch, args)
            query_preds = baseLearner(padded, seq_len, stats)
            batch_loss = loss_fn(query_preds, targets)
            queTaskLoss += batch_loss.item()
        
        queTaskLoss = queTaskLoss / len(queset)
        meta_val_loss += [queTaskLoss]
        print(f'\t\tadapted evaluation: {queTaskLoss:.3f}')

    meta_val_loss = domain_weightedAVG( meta_val_loss, 'Valid', verbose= False) if len(meta_val_loss)==3 else sum(meta_val_loss) / len(meta_val_loss)

    return meta_val_loss      

#%%
def meta_test(meta_model, loss_fn, test_domains_lists, orig_domain_means, args):

    meta_test_loss = []
    
    for taskID in range(len(test_domains_lists)):   # for each task in the batch
        print(f'\t( Batch Task {taskID+1} )')

        supset = SeqDataset(data= test_domains_lists[taskID]['sup'], withStats = args.with_stats)
        testset = SeqDataset(data= test_domains_lists[taskID]['test'], withStats = args.with_stats)
        suploader = DataLoader(supset, batch_size = args.meta_K, shuffle = True, collate_fn = supset.collate_fn )
        testloader = DataLoader(testset, batch_size = 1, shuffle = True, collate_fn = testset.collate_fn )
        
        supBatch  = next(iter(suploader))
        testiter =  iter(testloader)
        
        baseLearner = meta_model.clone()
        
        if args.metaVal_initMean == 'kSample':
            setBias = sum(supBatch['MRR']) / len(supBatch['MRR'])
        elif args.metaVal_initMean == 'target':
            setBias = orig_domain_means[taskID]
        baseLearner.module.out.bias.data.fill_(setBias)

        # divide the data into support and query sets
        padded, seq_len, stats, targets = batch_to_tensor(supBatch, args)
        
        # baseLearner adapt to support tasks
        for step in range(args.base_adaptSteps_test): # adaptation_steps
            
            mrr = baseLearner(padded, seq_len, stats)
            support_loss=loss_fn(mrr, targets)
            baseLearner.adapt(support_loss)
            
            print(f'\t\tadapt step {step+1} loss {support_loss:.3f}')

        testTaskLoss = 0.0
        for batch in range(len(testloader)):
            testBatch =  next(testiter)
            # record loss from query tasks
            padded, seq_len, stats, targets = batch_to_tensor(testBatch, args)
            test_preds = baseLearner(padded, seq_len, stats)
            batch_loss = loss_fn(test_preds, targets)
            testTaskLoss += batch_loss.item()
        
        testTaskLoss = testTaskLoss / len(testset)
        meta_test_loss += [round(testTaskLoss,3)]
        print(f'\t\tadapted evaluation: {testTaskLoss:.3f}')

    avgLoss = domain_weightedAVG(meta_test_loss, 'Test') if len(meta_test_loss)==3 else sum(meta_test_loss) / len(meta_test_loss)

    return avgLoss, meta_test_loss

#%%

def metaDomains_init(orig_datasets, args, verbose= False):
    
    all_datasets_df     = orig_datasets['df']
    all_datasets_list   = orig_datasets['list']
    
    for dataset in all_datasets_df:
        dataset.sort_values(by=['AVG_REMOVAL_RATE'], inplace = True)
        
    var_list, dataset_sizes = check_vars(all_datasets_df, restructured=False, verbose=False)
    
    input_dim = len(var_list)
    
    args.input_dim, args.last_hidden = input_dim, input_dim 
    
    hidden = input_dim
    if args.model in ['LSTM', 'GRU']:
        args.hidden_size = hidden
    elif args.model == 'TCN':
        args.hidden_channels = hidden
    elif args.model == 'Transformer' :
        args.dim_feedforward = hidden
    
    args.stats_size = 0 if not args.with_stats else all_datasets_list[0][0]['Statistics'].shape[0] * all_datasets_list[0][0]['Statistics'].shape[1]
    
    [support_1, query_1, test_1, support_2, query_2, test_2, support_3, query_3, test_3] = all_datasets_df
    [support_1_list, query_1_list, test_1_list, support_2_list, query_2_list, test_2_list, support_3_list, query_3_list, test_3_list] = all_datasets_list
    
    domain_1 = {'sup': support_1, 'que': query_1, 'test': test_1}
    domain_2 = {'sup': support_2, 'que': query_2, 'test': test_2}
    domain_3 = {'sup': support_3, 'que': query_3, 'test': test_3}
    train_domains_DFs = [domain_1, domain_2, domain_3]

    domain_1_list = {'sup': support_1_list, 'que': query_1_list, 'test': test_1_list}
    domain_2_list = {'sup': support_2_list, 'que': query_2_list, 'test': test_2_list}
    domain_3_list = {'sup': support_3_list, 'que': query_3_list, 'test': test_3_list}
    val_domains_lists = [domain_1_list, domain_2_list, domain_3_list]
    test_domains_lists = [domain_1_list, domain_2_list, domain_3_list]
    
    orig_domain_means = domain_means.copy()
    
    if args.no_leak:
        train_domains_DFs.pop( args.noLeak_tuningDomain-1 )
        val_domains_lists.pop( args.noLeak_tuningDomain-1 )
        orig_domain_means.pop( args.noLeak_tuningDomain-1 )
        
    dataset_sizes = []
    for d in range(len(train_domains_DFs)):
        for split in ['sup', 'que', 'test']:
            assert train_domains_DFs[d][split].shape[0] == len(val_domains_lists[d][split])
            dataset_sizes += [train_domains_DFs[d][split].shape[0]]
        
    if verbose:
        print('Input Dim:', input_dim, ', Dataset Sizes:', dataset_sizes)
        
    return train_domains_DFs, val_domains_lists, test_domains_lists, args, orig_domain_means

#%%
def meta_init(args: Namespace):
    # model
    if args.model == 'LSTM':
        base_model_1 = LSTM_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size , 
                             num_layers_lstm = args.num_layers_lstm , dropout = args.dropout , 
                             bidirectional = args.bidirectional , stats_size=args.stats_size,
                             last_hidden = args.last_hidden, act_fn = args.act_fn, 
                             stablize = args.stablize, set_target = args.set_target)
        base_model_2 = LSTM_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size ,
                               num_layers_lstm = args.num_layers_lstm , dropout = args.dropout ,
                               bidirectional = args.bidirectional , stats_size=args.stats_size,
                               last_hidden = args.last_hidden, act_fn = args.act_fn, 
                               stablize = args.stablize, set_target = args.set_target)
    elif args.model == 'TCN':
        base_model_1 = TCN_Regr(input_size= args.input_dim , hidden_size= args.hidden_channels , num_levels_tcn= args.num_levels_tcn ,
                    kernel_size= args.kernel_size, dropout= args.dropout , stats_size= args.stats_size,
                    last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, set_target = args.set_target)
        base_model_2 = TCN_Regr(input_size= args.input_dim , hidden_size= args.hidden_channels , num_levels_tcn= args.num_levels_tcn ,
                    kernel_size= args.kernel_size, dropout= args.dropout , stats_size= args.stats_size,
                    last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, set_target = args.set_target)
    
    elif args.model == 'Transformer':
        base_model_1 = Transformer_Regr(d_model= args.input_dim, num_encoder_layers= args.num_encoder_layers, nhead= args.nhead, 
                                dim_feedforward= args.dim_feedforward, dim_reduction= args.dim_reduction, dropout=args.dropout ,
                                stats_size=args.stats_size, last_hidden = args.last_hidden, 
                                act_fn = args.act_fn, stablize = args.stablize, set_target = args.set_target, pos_encoding = args.pos_encoding)
        
        base_model_2 = Transformer_Regr(d_model= args.input_dim, num_encoder_layers= args.num_encoder_layers, nhead= args.nhead, 
                                dim_feedforward= args.dim_feedforward, dim_reduction= args.dim_reduction, dropout=args.dropout ,
                                stats_size=args.stats_size, last_hidden = args.last_hidden, 
                                act_fn = args.act_fn, stablize = args.stablize, set_target = args.set_target, pos_encoding = args.pos_encoding)
    
    elif args.modle == 'GRU':
        base_model_1 = GRU_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size , 
                         num_layers_gru = args.num_layers_gru , dropout = args.dropout , 
                         bidirectional = args.bidirectional , stats_size=args.stats_size,
                         last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, set_target = args.set_target)
        base_model_2 = GRU_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size ,
                               num_layers_gru = args.num_layers_gru , dropout = args.dropout ,
                               bidirectional = args.bidirectional , stats_size=args.stats_size,
                               last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, set_target = args.set_target)
        
    
    # meta model
    if args.model == 'TCN':
        meta_model_1 = l2l.algorithms.MAML(base_model_1, args.base_lr).to(args.device)
        meta_model_2 = l2l.algorithms.MAML(base_model_2, args.base_lr).to(args.device)
    else:
        kronecker_transform = l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear)
        
        meta_model_1 = l2l.algorithms.GBML( base_model_1, kronecker_transform, args.base_lr).to(args.device)
        meta_model_2 = l2l.algorithms.GBML( base_model_2, kronecker_transform, args.base_lr).to(args.device)
        

    # optimization
    if args.opt_type == 'SGD':
        optimizer = torch.optim.SGD(meta_model_1.parameters(), lr = args.meta_lr, momentum = args.sgd_momentum, weight_decay = args.meta_wd)
    elif args.opt_type == 'Adam':
        optimizer = torch.optim.Adam(meta_model_1.parameters(), lr = args.meta_lr, weight_decay = args.meta_wd)
    elif args.opt_type =='AMSGrad':
        optimizer = torch.optim.Adam(meta_model_1.parameters(), lr = args.meta_lr, weight_decay = args.meta_wd, amsgrad= True)
    elif args.opt_type == 'Noam':
        optimizer = NoamOpt(
            model_size= args.input_dim, factor= args.noam_factor, warmup= args.noam_warmup, 
            optimizer= torch.optim.AdamW(meta_model_1.parameters(), lr= args.adamW_lr, betas=args.adamW_betas, eps=args.adamW_eps, weight_decay= args.meta_wd))
        plt.plot(np.arange(1, args.pureVM_numEpoch), [optimizer.rate(i) for i in range(1, args.meta_numEpoch)])
        plt.legend([f"Dim:{optimizer.model_size}, Factor:{optimizer.factor}, Warmup:{optimizer.warmup}"])
        plt.yticks(fontsize= 10)
        plt.title('Scheduled Learning Rate')
        plt.show()
    else:
        raise Exception('Unprogrammed Optimization Type !')
    
    # loss function
    loss_fn =  nn.MSELoss(reduction= args.meta_lossReduction)

    return meta_model_1, meta_model_2, optimizer, loss_fn


#%%

def metaLrRegulator(optimizer, args, epoch, loss_all, metaFineTune=False, opt_temp=None):
    
    bestValLoss = float("inf") if epoch == 0 else loss_all['best']['valid'][-1]
    
    if metaFineTune and epoch <= args.base_adaptSteps:
        optimizer = opt_temp if epoch == args.base_adaptSteps else optimizer
        
    elif args.lr_regulator and epoch > 0:

        if bestValLoss > 1000.0:
            for g in optimizer.param_groups:
                g['lr'] = args.meta_lr * 1000
                
        elif bestValLoss > 500.0:
            for g in optimizer.param_groups:
                g['lr'] = args.meta_lr * 100
                
        elif bestValLoss > 100.0:
            for g in optimizer.param_groups:
                g['lr'] = args.meta_lr * 10
        
        elif bestValLoss > 50.0:
            for g in optimizer.param_groups:
                g['lr'] = args.meta_lr * 1
        else:
            for g in optimizer.param_groups:
                g['lr'] = args.meta_lr * 0.5
            
    
    return optimizer

#%%

def meta_main(args: Namespace, config_preprocess2, orig_datasets:dict, savingDir: str, verbose=True):
    if not os.path.exists(savingDir):
        os.makedirs(savingDir)
    os.chdir(savingDir)
    startTime = datetime.now().strftime('%H.%M.%S')

    # Initialization
    train_domains_DFs, val_domains_lists, test_domains_lists, args, orig_domain_means = metaDomains_init(
        orig_datasets, args, verbose= True)
    
    meta_model, best_meta_model, opt, loss_fn = meta_init(args)

    loss_all = {'epoch': {'train':[],'valid': []},'best': {'train':[],'valid': []}}
 
    
    pbar = range(args.meta_numEpoch) if verbose else tqdm(range(args.meta_numEpoch))
    for epoch in pbar: # num_tasks/batch_size
        print(f'\n【 Epoch {epoch+1} 】')
        print('   [ Meta Train ]')
            
        metaBatch, metaBatchMeans = generateMetaBatch(train_domains_DFs, orig_domain_means, 
                                                      args, config_preprocess2, restructure= True)
        
        metaBatch, metaBatchMeans = add_orig_domain( metaBatch, metaBatchMeans, val_domains_lists, orig_domain_means, args= args)
        
        opt = metaLrRegulator(opt, args, epoch, loss_all)
        
        meta_train_loss = meta_train(meta_model, loss_fn, opt, metaBatch, metaBatchMeans, args)  
        
        
        print('   [ Meta Validation ]')
        meta_val_loss = meta_val(meta_model, loss_fn, val_domains_lists, orig_domain_means, args)

        loss_all = updateLosses_modelCheckpoint(epoch, meta_train_loss, meta_val_loss, loss_all, meta_model, startTime, None, verbose, pbar)
    
    print('【 Meta Test 】')
    best_meta_model.load_state_dict(torch.load(f'{startTime} Best_valid_loss_state.pt')) 

    avg_testLoss, meta_test_loss = meta_test(best_meta_model, loss_fn, test_domains_lists, domain_means,  args)
    print('Test losses : ', meta_test_loss, ' , AVG : ', round(avg_testLoss,3))

    learning_curve(allLoss= loss_all, finalValLoss = None, testLoss= round(avg_testLoss,3), save= True, startTime= startTime, args= args)
    
    return best_meta_model

#%%

def meta_fineTuneTest(meta_model, args, domain_KQsampler, savingDir):
    
    startTime2 = datetime.now().strftime('%H.%M.%S')
   
    Taskset_1, Taskset_2, Taskset_3 = domain_KQsampler.sampleKQ(args)

    # Main Training
    trained_model = meta_model.module
    torch.save(trained_model.state_dict(), f'{startTime2} fine tune temp.pt')  
    
    # Fine Tune 1
    print("\n----【Fine Tune MAML on Domain 1】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_1, best_fineTuned_model_1,  bestLosses_tune1 = pureVM_main( 
        args, trainData = Taskset_1[0], valData = Taskset_1[1], testData = Taskset_1[2], 
        savingDir = savingDir, fineTune = True, metaFineTune= True, trained_model = trained_model, 
        verbose= False, plotLearningCurve = True, bias_init = domain_means[0] )

    # Fine Tune 2
    print("\n----【Fine Tune MAML on Domain 2】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_2, best_fineTuned_model_2, bestLosses_tune2 = pureVM_main(
        args, trainData = Taskset_2[0], valData = Taskset_2[1], testData = Taskset_2[2],
        savingDir = savingDir, fineTune = True,  metaFineTune= True, trained_model = trained_model, 
        verbose= False, plotLearningCurve = True, bias_init = domain_means[1] )
    
    # Fine Tune 3
    print("\n----【Fine Tune MAML on Domain 3】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_3, best_fineTuned_model_3,  bestLosses_tune3 = pureVM_main(
        args, trainData = Taskset_3[0], valData = Taskset_3[1], testData = Taskset_3[2],
        savingDir = savingDir, fineTune = True,  metaFineTune= True, trained_model = trained_model, 
        verbose= False, plotLearningCurve = True, bias_init = domain_means[2] )
        
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    
    bestLossTrain = bestLosses_tune1[0], bestLosses_tune2[0], bestLosses_tune3[0]
    bestLossValid = bestLosses_tune1[1], bestLosses_tune2[1], bestLosses_tune3[1]
    bestLossTest  = bestLosses_tune1[2], bestLosses_tune2[2], bestLosses_tune3[2]
    
    print("\n- - - - - -【Dataset Total Loss】- - - - - -")
    lossTrain = domain_weightedAVG( bestLossTrain, 'Train')
    lossValid = domain_weightedAVG( bestLossValid, 'Valid')
    lossTest = domain_weightedAVG( bestLossTest, 'Test' )
    
    if args.no_leak:
        return [bestLosses_tune1, bestLosses_tune2, bestLosses_tune3][args.noLeak_tuningDomain-1]
    else:
        return [lossTrain, lossValid, lossTest]
