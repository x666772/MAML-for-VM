# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import random
from typing import List, Dict
from random import sample

# Visualization
import matplotlib.pyplot as plt

# Neural Network
import torch
from torch.nn.utils import rnn
from torch.utils.data import Dataset

#%%


domain_means = [151.24, 73.08, 79.97]
domain_sizes = { 'Train': [364,798,815] , 'Valid': [67,185,172], 'Test': [73,165,186] }

# Dataset Total Loss
def domain_weightedAVG(loss_list: list, orig_split: str, verbose=True) -> float :

    avgLoss = np.average(a= loss_list, weights= domain_sizes[orig_split] )

    if verbose:
        print(f'Weighted-Averaged {orig_split}  Loss: {avgLoss:.3f}')
    return avgLoss

# Adapting all loss reductions
def comupte_epoch_loss(unreduced_losses_list, dataset_size, loss_reduction ):

    epoch_loss = torch.cat(unreduced_losses_list)

    if loss_reduction == 'mean':
        epoch_loss = np.average(a= epoch_loss.tolist(), weights= dataset_size)
    elif loss_reduction == 'sum':
        epoch_loss =  epoch_loss.sum().item() / sum(dataset_size)
    elif loss_reduction == 'none':
        epoch_loss =  epoch_loss.mean().item()
    else:
        raise Exception('Unprogrammed loss reduction:', loss_reduction)
    
    return epoch_loss

# Record epoch loss, best loss, save best model, show loss status
def updateLosses_modelCheckpoint(epoch, trainLoss, validLoss, allLoss, model, startTime, summaryWriter, verbose, pbar= None):
    
    trainLoss = 1e6 if np.isnan(trainLoss) or np.isinf(trainLoss) else trainLoss
    validLoss = 1e6 if np.isnan(validLoss) or np.isinf(validLoss) else validLoss
    
    # Epoch Loss
    allLoss['epoch']['train'].append(trainLoss)
    allLoss['epoch']['valid'].append(validLoss)
    
    # Best Loss & Model Checkpoint
    if epoch == 0:
        allLoss['best'] = {'train':[trainLoss], 'valid': [validLoss]}
        torch.save(model.state_dict(), f'{startTime} Best_train_loss_state.pt')
        torch.save(model.state_dict(), f'{startTime} Best_valid_loss_state.pt')
    else:
        if trainLoss <= allLoss['best']['train'][-1]:
            allLoss['best']['train'].append(trainLoss)
            torch.save(model.state_dict(), f'{startTime} Best_train_loss_state.pt')
        else:
            allLoss['best']['train'].append( allLoss['best']['train'][-1] )
        
        if validLoss <= allLoss['best']['valid'][-1]:
            allLoss['best']['valid'].append(validLoss)
            torch.save(model.state_dict(), f'{startTime} Best_valid_loss_state.pt')
        else:
            allLoss['best']['valid'].append(allLoss['best']['valid'][-1])
    
    # Summary Writer
    if summaryWriter != None:
        summaryWriter.add_scalars('Loss/Epoch', {'train': trainLoss                   , 'valid': validLoss                   }, epoch)
        summaryWriter.add_scalars('Loss/Best' , {'train': allLoss['best']['train'][-1], 'valid': allLoss['best']['valid'][-1]}, epoch)
    
    # Show Loss
    if verbose:
        print("【Epoch %d】" % (epoch + 1))
        best_trainLoss, best_validLoss = allLoss['best']['train'][-1], allLoss['best']['valid'][-1]
        print(f'  Train Loss: {trainLoss:.3f}\t|  Best: {best_trainLoss:.3f}')
        print(f'  Valid Loss: {validLoss:.3f}\t|  Best: {best_validLoss:.3f}')
    else:
        pbar.set_postfix({'Train': round(trainLoss,3), 'bestTrain':round(allLoss['best']['train'][-1],3), 'Val': round(validLoss,3), 'bestVal': round(allLoss['best']['valid'][-1],3)}) 
    
    return allLoss

# Plot and save learning curve
def learning_curve(allLoss:Dict, finalValLoss, testLoss, save: bool, startTime, args):
    plt.rcParams.update({'font.size': 5})
    fig = plt.figure(figsize= (5.3 , 4.3), dpi=128)
    
    idx = 1
    for EoB in ['epoch', 'best']:
        
        sub = fig.add_subplot(2,2,idx)
        ymax = max(np.percentile(allLoss[EoB]['train'], 80), np.percentile(allLoss[EoB]['valid'], 80)) + 1
        ypbot1, ypbot2 = min(allLoss[EoB]['train']), min(allLoss[EoB]['valid'])
        if ymax - max(ypbot1, ypbot2) > 3000:
            ymax = max(ypbot1, ypbot2) + 3000
        ymin = min(ypbot1, ypbot2)
        ymin = ymin- (ymax-ymin)*0.05

        sub.plot(allLoss[EoB]['train'])
        sub.plot(allLoss[EoB]['valid'])
        sub.axhline(y=ypbot1, color='silver', linewidth = 0.5)
        sub.axhline(y=ypbot2, color='silver', linewidth = 0.5)
        sub.set_title(f'{EoB} loss')
        sub.set_ylim([ymin, ymax])
        idx +=2
    
    sub = fig.add_subplot(1,2,2)
    sub.axis('off')
    best_trainLoss, best_validLoss = allLoss['best']['train'][-1], allLoss['best']['valid'][-1]
    
    pd.set_option("display.precision", 10)
    hyperpara = pd.DataFrame(vars(args), index = [0]).T
    hyperpara = hyperpara.drop(['model', 'experiment','device'])
    
    if 'reload_datetime' in hyperpara.index:
        hyperpara.loc['reload_datetime'] = args.reload_datetime[5:]
    
    txt = f'Model: {args.model}\nExperiment: {args.experiment}\nStartTime: {startTime}\nBest Training Loss: {best_trainLoss:.3f}\nBest Validation Loss: {best_validLoss:.3f}\nTest Loss: {testLoss:.3f}\nSettings:{hyperpara}'
    sub.text(0.03, 0.0, txt, fontsize = 5)
    
    fig.tight_layout()
    if save:
        fig.savefig(f'{startTime} Learning Curve.jpg')

    plt.show()

#%%
class SeqDataset(Dataset): # 繼承自 Dataset
    def __init__(
        self, 
        data: List[Dict], 
        withStats = False ):
    
        self.data = data
        self.withStats = withStats

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:

        time_series, targets, idx, Len, Stats = [], [], [], [], []
        isTestSet = True if 'AVG_REMOVAL_RATE' not in samples[0].keys() else False

        # Gather in lists, and encode labels as indices
        for aSample in samples:
            idx += [aSample['ID']]
            time_series += [torch.FloatTensor(aSample['SVID'].values)]
            Len += [int(aSample['Len'])]
            
            if self.withStats:
                Stats += [torch.FloatTensor(aSample['Statistics'].values)]
            if isTestSet == False :
                targets += [aSample['AVG_REMOVAL_RATE']]
            
        # Group the list of tensors into a batched tensor
        # tensors = pad_to_len(tensors, self.max_len, 0)
        
        collated = {'SVID': time_series, 'Len': Len , 'Stats': Stats, 'ID':idx }
        if isTestSet == False:
            collated['MRR'] = targets
        if self.withStats:
            collated['Stats'] = Stats
        
        return collated
        
        
#%%
def batch_to_tensor(batch, args):
    
    batch_first = False if args.model == 'Transformer' else True
    
    padded  = rnn.pad_sequence(batch['SVID'], batch_first=batch_first)
     
    seq_len = torch.tensor(batch['Len'])
    
    targets = torch.FloatTensor(batch['MRR'])
    targets = torch.reshape(targets, (targets.size()[0],1))

    padded, targets = padded.to(args.device), targets.to(args.device)

    stats = torch.stack(batch['Stats']).to(args.device) if args.with_stats else None
    
    return padded, seq_len, stats, targets
    
#%% 

class domain_KQsampler:
    
    def __init__(self, orig_datasets):
        

        self.all_datasets_df = orig_datasets['df']
        self.all_datasets_list = orig_datasets['list']
        
    def sampleKQ(self, args):
    
        [support_1, query_1, test_1, 
         support_2, query_2, test_2, 
         support_3, query_3, test_3] = self.all_datasets_df
        
        [support_1_list, query_1_list, test_1_list, 
         support_2_list, query_2_list, test_2_list, 
         support_3_list, query_3_list, test_3_list] = self.all_datasets_list
        
        Ks, Qs= [], []
        
        if args.vm_K != 'all':
            for i in range(3):
                Ks += [ args.vm_K if domain_sizes['Train'][i] > args.vm_K else domain_sizes['Train'][i] ]
                
        if args.vm_Q != 'all':
            for i in range(3):
                Qs += [ args.vm_Q if domain_sizes['Valid'][i] > args.vm_Q else domain_sizes['Valid'][i] ]
        
        if args.vm_sampleMode == 'uniform':
            
            support_1_sampled = support_1_list if args.vm_K == 'all' else sample(support_1_list, Ks[0])
            support_2_sampled = support_2_list if args.vm_K == 'all' else sample(support_2_list, Ks[1])
            support_3_sampled = support_3_list if args.vm_K == 'all' else sample(support_3_list, Ks[2])
            
            query_1_sampled = query_1_list if args.vm_Q == 'all' else sample(query_1_list, Qs[0])
            query_2_sampled = query_2_list if args.vm_Q == 'all' else sample(query_2_list, Qs[1])
            query_3_sampled = query_3_list if args.vm_Q == 'all' else sample(query_3_list, Qs[2])
            
            Taskset_1 = [support_1_sampled, query_1_sampled, test_1_list] 
            Taskset_2 = [support_2_sampled, query_2_sampled, test_2_list] 
            Taskset_3 = [support_3_sampled, query_3_sampled, test_3_list]
            
        elif args.vm_sampleMode == 'normal':
            
            
            all_datasets = [support_1, query_1, test_1, support_2, query_2, test_2, support_3, query_3, test_3]
            for dataset in all_datasets:
                dataset.sort_values(by=['AVG_REMOVAL_RATE'], inplace = True)
                
            domain_1 = {'sup': support_1, 'que': query_1, 'test': test_1}
            domain_2 = {'sup': support_2, 'que': query_2, 'test': test_2}
            domain_3 = {'sup': support_3, 'que': query_3, 'test': test_3}
            domains_DFs = [domain_1, domain_2, domain_3]
            
            domain_lists = []
            for d in range(3):
            
                percentiles_k = random.randn(Ks[d]) # smaple with N(0,1) shape
                percentiles_q = random.randn(Qs[d]) # smaple with N(0,1) shape
                
                percentiles_k =  percentiles_k - min(percentiles_k) + random.uniform(0,0.1)
                percentiles_q =  percentiles_q - min(percentiles_q) + random.uniform(0,0.1)
                
                percentiles_k =  percentiles_k / ( max(percentiles_k) + random.uniform(0,0.1))
                percentiles_q =  percentiles_q / ( max(percentiles_q) + random.uniform(0,0.1))
                
                ids_k = np.sort((percentiles_k * domain_sizes['Train'][d]).astype(int)) 
                ids_q = np.sort((percentiles_q * domain_sizes['Valid'][d]).astype(int)) 
                
                ids_k = range(domain_sizes['Train'][d]) if Ks[d] >= domain_sizes['Train'][d] else ids_k
                ids_q = range(domain_sizes['Valid'][d]) if Qs[d] >= domain_sizes['Valid'][d] else ids_q
                
                domain_sup, domain_que = [], []
                for k in range(Ks[d]):
                    
                    ts_k = domains_DFs[d]['sup'].iloc[ids_k[k]].to_dict()
                    domain_sup += [ts_k]
                    
                for q in range(Qs[d]):
                    
                    ts_q = domains_DFs[d]['que'].iloc[ids_q[q]].to_dict()
                    domain_que += [ts_q]
                
                domain_lists += [[domain_sup, domain_que]]
            
            Taskset_1 = [domain_lists[0][0], domain_lists[0][1], test_1_list] 
            Taskset_2 = [domain_lists[1][0], domain_lists[1][1], test_2_list] 
            Taskset_3 = [domain_lists[2][0], domain_lists[2][1], test_3_list]
            
        else:
            raise Exception(f'Unprogrammed vm_sampleMode \'{args.vm_sampleMode}\'!')    
        
        return Taskset_1, Taskset_2, Taskset_3



#%%

def avg(a,b,c):
    aList = [a,b,c]
    wavg = round(domain_weightedAVG(aList,'Test',verbose=False),3)
    avg = round(np.mean(aList),3)
    print(wavg, '\t', avg)
    
#%%

def remove_outlier_IQR(exp_losses):
    
    loss_df = exp_losses.copy()
    
    q75,q25 = np.percentile(loss_df['test'],[75,25])
    IQR = q75-q25
 
    max = q75+(1.5*IQR)
    min = q25-(1.5*IQR)
    
    for col in ['train', 'valid', 'test']:
        loss_df.loc[ loss_df['test'] < min, col ] = np.nan
        loss_df.loc[ loss_df['test'] > max, col ] = np.nan
    
    loss_desc = loss_df.describe()
    
    print('Num outliers: ', loss_df['test'].isnull().sum())
    print(loss_desc)
    
    return loss_df, loss_desc

    