# -*- coding: utf-8 -*-

import numpy as np

# Visualization
import matplotlib.pyplot as plt
from tqdm import tqdm

#Expression
from argparse import  Namespace

# Neural Network
import torch
from torch import nn

from torch.utils.data import  DataLoader
from torch.utils.tensorboard import SummaryWriter

# Directory
import os
os.chdir('G:/其他電腦/MacBook Pro/Researches/_CODE_')

from preprocess2 import check_vars
from models import LSTM_Regr, TCN_Regr, Transformer_Regr, GRU_Regr

from utils import comupte_epoch_loss, updateLosses_modelCheckpoint, learning_curve
from utils import SeqDataset, batch_to_tensor

from datetime import datetime

#%%

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

#%%
def train(model, iterator, optimizer, criterion, args):
    model.train()   # set model in training mode

    #initialize every epoch
    batch_losses, dataset_size = [], []

    for batch in iterator:
        dataset_size.append( len(batch['ID']) ) # Record Batch Size
        
        optimizer.zero_grad()           # reset the gradients after every batch
        
        padded, seq_len, stats, targets = batch_to_tensor(batch, args)
        
        mrr = model(padded, seq_len, stats)
        
        loss = criterion(mrr, targets).flatten()  # compute batch loss
        batch_losses.append(loss)         # epoch_loss -> list of tensors
        loss = loss.mean() if args.pureVM_lossReduction == 'none' else loss
        loss.backward()                 # backpropage the loss and compute gradients  
        
        if args.clip :
            torch.nn.utils.clip_grad_norm_(parameters= model.parameters(), max_norm= 0.5, norm_type=2)
            
        optimizer.step()                # update the weights

    epoch_loss = comupte_epoch_loss(batch_losses, dataset_size, args.pureVM_lossReduction )
    return epoch_loss

def evaluate(model, iterator, criterion, args):
    model.eval()  # set model in evluation mode

    #initialize every epoch
    batch_losses, dataset_size = [], []

    with torch.no_grad():               # deactivates autograd
        for batch in iterator:
            dataset_size.append( len(batch['ID']) ) # Record Batch Size
            
            padded, seq_len, stats, targets = batch_to_tensor(batch, args )
            
            mrr = model(padded, seq_len, stats)
            loss = criterion(mrr, targets).flatten()  # compute batch loss

            batch_losses.append(loss)   
         
    epoch_loss = comupte_epoch_loss(batch_losses, dataset_size, args.pureVM_lossReduction )
    return epoch_loss

#%%
def dataloader_init(train_data, val_data, test_data, args, verbose= False):
    
    var_list, dataset_sizes = check_vars([train_data, val_data, test_data], restructured=True, verbose=False)
    input_dim = len(var_list)
    
    args.input_dim, args.last_hidden = input_dim, input_dim 
    
    hidden = input_dim
    if args.model in ['LSTM', 'GRU']:
        args.hidden_size = hidden
    elif args.model == 'TCN':
        args.hidden_channels = hidden
    elif args.model == 'Transformer' :
        args.dim_feedforward = hidden
    
    
    args.stats_size = 0 if not args.with_stats else train_data[0]['Statistics'].shape[0] * train_data[0]['Statistics'].shape[1]
    
    if verbose:
        print('Input Dim:', input_dim, ', Dataset Sizes:', dataset_sizes)
    
    datasets: dict[str, SeqDataset] = {
        TRAIN : SeqDataset(data= train_data, withStats = args.with_stats),
        DEV   : SeqDataset(data= val_data  , withStats = args.with_stats),
        TEST  : SeqDataset(data= test_data , withStats = args.with_stats)   }

    train_loader = DataLoader(datasets['train'], batch_size = args.batch_size, shuffle = True, 
                              collate_fn = datasets['train'].collate_fn )
    dev_loader =   DataLoader(datasets['eval'], batch_size = 1, shuffle = False, collate_fn = datasets['eval'].collate_fn )
    test_loader =  DataLoader(datasets['test'], batch_size = 1, shuffle = False, collate_fn = datasets['test'].collate_fn)
    
    return train_loader, dev_loader, test_loader, args

def initialization(args: Namespace, plot_scheduler= True, trained_model= None, bias_init= 90):
    # model
    
    if args.model == 'LSTM':
        model = LSTM_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size , 
                             num_layers_lstm = args.num_layers_lstm , dropout = args.dropout , 
                             bidirectional = args.bidirectional , stats_size=args.stats_size,
                             last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, 
                             bias_init=bias_init, set_target = args.set_target)
        
        model_2 = LSTM_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size ,
                               num_layers_lstm = args.num_layers_lstm , dropout = args.dropout ,
                               bidirectional = args.bidirectional , stats_size=args.stats_size,
                               last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, 
                               bias_init=bias_init , set_target = args.set_target)
    elif args.model == 'TCN':
        model = TCN_Regr(input_size= args.input_dim , hidden_size= args.hidden_channels , num_levels_tcn= args.num_levels_tcn ,
                    kernel_size= args.kernel_size, dropout= args.dropout , stats_size= args.stats_size,
                    last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, 
                    bias_init=bias_init, set_target = args.set_target)
        
        model_2 = TCN_Regr(input_size= args.input_dim , hidden_size= args.hidden_channels , num_levels_tcn= args.num_levels_tcn ,
                    kernel_size= args.kernel_size, dropout= args.dropout , stats_size= args.stats_size,
                    last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, 
                    bias_init=bias_init, set_target = args.set_target)
    
    
    elif args.model == 'Transformer':
        model = Transformer_Regr(d_model= args.input_dim, num_encoder_layers= args.num_encoder_layers, nhead= args.nhead, 
                                dim_feedforward= args.dim_feedforward, dim_reduction= args.dim_reduction, dropout=args.dropout ,
                                stats_size=args.stats_size, last_hidden = args.last_hidden, 
                                act_fn = args.act_fn, stablize = args.stablize, 
                                bias_init=bias_init, set_target = args.set_target, pos_encoding = args.pos_encoding)
        
        model_2 = Transformer_Regr(d_model= args.input_dim, num_encoder_layers= args.num_encoder_layers, nhead= args.nhead, 
                                dim_feedforward= args.dim_feedforward, dim_reduction= args.dim_reduction, dropout=args.dropout ,
                                stats_size=args.stats_size, last_hidden = args.last_hidden, 
                                act_fn = args.act_fn, stablize = args.stablize, 
                                bias_init=bias_init, set_target = args.set_target, pos_encoding = args.pos_encoding)
    elif args.model == 'GRU':
        model = GRU_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size , 
                         num_layers_gru = args.num_layers_gru , dropout = args.dropout , 
                         bidirectional = args.bidirectional , stats_size=args.stats_size,
                         last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, 
                         bias_init=bias_init, set_target = args.set_target)
        
        model_2 = GRU_Regr(input_dim = args.input_dim , hidden_size = args.hidden_size ,
                               num_layers_gru = args.num_layers_gru , dropout = args.dropout ,
                               bidirectional = args.bidirectional , stats_size=args.stats_size,
                               last_hidden = args.last_hidden, act_fn = args.act_fn, stablize = args.stablize, 
                               bias_init=bias_init, set_target = args.set_target)
        
    if trained_model != None:
        model = trained_model
        if args.fineTune_targetBias:
            model.out.bias.data.fill_(bias_init)
        
    model, model_2 = model.to(args.device), model_2.to(args.device)
    
    # optimization
    if args.opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.pureVM_lr, momentum = args.sgd_momentum, weight_decay = args.weight_decay)
    elif args.opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.pureVM_lr, weight_decay = args.weight_decay)
    elif args.opt_type =='AMSGrad':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.pureVM_lr, weight_decay = args.weight_decay, amsgrad= True)
    elif args.opt_type == 'Noam':
        baseOpt = torch.optim.AdamW(model.parameters(), lr= args.adamW_lr, betas=(args.adamW_beta1, args.adamW_beta2), eps=args.adamW_eps, weight_decay= args.weight_decay)
        optimizer = NoamOpt(model_size= args.input_dim, factor= args.noam_factor, warmup= args.noam_warmup, optimizer= baseOpt)
        plt.plot(np.arange(1, args.pureVM_numEpoch), [optimizer.rate(i) for i in range(1, args.pureVM_numEpoch)])
        plt.legend([f"Dim:{optimizer.model_size}, Factor:{optimizer.factor}, Warmup:{optimizer.warmup}"])
        plt.yticks(fontsize= 10)
        plt.title('Scheduled Learning Rate')
        plt.show()
    elif args.opt_type == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr = args.pureVM_lr, weight_decay = args.weight_decay)
    else:
        raise Exception('Unprogrammed Optimization Type !')
    
    # loss function
    criterion =  nn.MSELoss(reduction= args.pureVM_lossReduction)

    return model, model_2, optimizer, criterion

#%%

def lrRegulator(optimizer, args, epoch, loss_all, metaFineTune, opt_temp):
    
    bestValLoss = None if epoch == 0 else loss_all['best']['valid'][-1]
    
    if metaFineTune and epoch <= args.base_adaptSteps:
        optimizer = opt_temp if epoch == args.base_adaptSteps else optimizer
    
    elif args.lr_regulator and epoch > 0:
        if bestValLoss > 5000.0:
            for g in optimizer.param_groups:
                g['lr'] = args.pureVM_lr * 500
        elif bestValLoss > 500.0:
            for g in optimizer.param_groups:
                g['lr'] = args.pureVM_lr * 100
        elif bestValLoss > 100.0:
            for g in optimizer.param_groups:
                g['lr'] = args.pureVM_lr * 10
        else:
            for g in optimizer.param_groups:
                g['lr'] = args.pureVM_lr * 1
            
    return optimizer
    
#%%
TRAIN, DEV, TEST = "train", 'eval', 'test'
SPLITS = [TRAIN, DEV, TEST]

def pureVM_main(args, trainData: list , valData: list, testData: list , savingDir: str, 
                Inference = False, fineTune = False, metaFineTune = False, trained_model = None, 
                verbose= True, plotLearningCurve= True, bias_init= 90):
    if not os.path.exists(savingDir):
        os.makedirs(savingDir)
    os.chdir(savingDir)
    startTime = datetime.now().strftime('%H.%M.%S')
    
    # Initialization

    train_loader, dev_loader, test_loader, args = dataloader_init(trainData, valData, testData, args, verbose=True)
    
    model, best_val_model, optimizer, criterion = initialization(args, plot_scheduler = plotLearningCurve, trained_model=trained_model, bias_init=bias_init)
    
    opt_temp = optimizer if metaFineTune else None
    optimizer = torch.optim.SGD(model.parameters(), lr = args.base_lr) if metaFineTune else optimizer

    writer = SummaryWriter() if plotLearningCurve else None

    loss_all = {'epoch': {'train':[],'valid': []},'best': {'train':[],'valid': []}}
    
    numEpoch = args.pureVM_fineTune_numEpoch if fineTune else args.pureVM_numEpoch
    pbar = range(numEpoch) if verbose else tqdm(range(numEpoch))
    for epoch in pbar:
        
        optimizer = lrRegulator(optimizer, args, epoch, loss_all, metaFineTune, opt_temp)
            
        train_iterator, dev_iterator = iter(train_loader), iter(dev_loader)
        
        train_loss = train(model, train_iterator, optimizer, criterion, args )
        valid_loss = evaluate(model, dev_iterator, criterion, args )
        
        loss_all = updateLosses_modelCheckpoint(epoch, train_loss, valid_loss, loss_all, model, startTime, writer, verbose, pbar)

    # Valid best loss model
    print('【Best Val.Loss Model】')
    best_val_model.load_state_dict(torch.load(f'{startTime} Best_valid_loss_state.pt'))

    best_train_loss = loss_all['best']['train'][-1]
    dev_iterator = iter(dev_loader)
    best_valid_loss = evaluate(best_val_model, dev_iterator, criterion, args )

    if Inference == False:
        test_iterator = iter(test_loader)
        test_loss = evaluate(best_val_model, test_iterator, criterion, args )
        
    # Record & Show Final Results
    print(f' Val Loss = {best_valid_loss:.3f}  |  Test Loss = {test_loss:.3f}')

    if plotLearningCurve:
        learning_curve(allLoss= loss_all, finalValLoss = best_valid_loss, testLoss= test_loss, save= True, startTime= startTime, args= args)
        writer.close()

    txtFile= open("Model Loss Records.txt",'a')
    txt = f'【Start Time: {startTime}】\n Best Training Loss: {best_train_loss:.3f}   |   Best Validation Loss: {best_valid_loss:.3f}\n Testing Loss: {test_loss:.3f}\n Setting:\n{vars(args)}\n\n'    
    txtFile.write(txt)
    txtFile.close()

    bestLosses = [best_train_loss, best_valid_loss, test_loss]

    return model, best_val_model, bestLosses

#%%

def mismatch_test(args, model, testDatasets, testset_names ):
    
    for setID in range(len(testDatasets)):
        datasets: dict[str, SeqDataset] = {
            TEST   : SeqDataset(data= testDatasets[setID], withStats = args.with_stats)}
        test_loader =  DataLoader(datasets['test'], batch_size = len(datasets['test']) , shuffle = False, 
                                  collate_fn = datasets['test'].collate_fn )
        test_iterator = iter(test_loader)
        model = model.to(args.device)
        criterion = nn.MSELoss(reduction= args.pureVM_lossReduction)
        test_loss = evaluate(model, test_iterator, criterion, args.device, args.pureVM_lossReduction)
        print(f'{testset_names[setID]} testing loss: {test_loss:.3f}')