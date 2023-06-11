# -*- coding: utf-8 -*-

import pandas as pd

# Neural Network
import torch

import os
os.chdir('G:/其他電腦/MacBook Pro/Researches/_CODE_')
from vm import pureVM_main
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

domain_means = [151.24, 73.08, 79.97]
domain_sizes = { 'Train': [364,798,815] , 'Valid': [67,185,172], 'Test': [73,165,186] }

#%%

def fineTune_test(ordered_tasks: dict, domains_order: list, args, savingDir):
    
    startTime2 = datetime.now().strftime('%H.%M.%S')
    selectedTask1, selectedTask2, selectedTask3 = ordered_tasks['data']
    name1, name2, name3 = ordered_tasks['name']
    
    if args.fineTune_targetBias:
        initBias = [domain_means[domains_order[0]-1], domain_means[domains_order[1]-1], domain_means[domains_order[2]-1]]
    else:
        initBias = [90,90,90]
        
    # Main Training
    print(f"--------------【Train {name1}】--------------")
    model, trained_model, bestLosses_main = pureVM_main(
        args, trainData = selectedTask1[0], valData = selectedTask1[1], testData = selectedTask1[2],
        savingDir = savingDir, fineTune = False, trained_model = None, verbose= False, plotLearningCurve = True, bias_init= initBias[0])
    torch.save(trained_model.state_dict(), f'{startTime2} fine tune temp.pt')

    # Fine Tune 1
    print(f"----【Fine Tune {name1} Model on {name2}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_1, best_fineTuned_model_1, bestLosses_tune1 = pureVM_main(
        args, trainData = selectedTask2[0], valData = selectedTask2[1], testData = selectedTask2[2], 
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= True, plotLearningCurve = False, bias_init= initBias[1] )

    # Fine Tune 2
    print(f"----【Fine Tune {name1} Model on {name3}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_2, best_fineTuned_model_2, bestLosses_tune2 = pureVM_main(
        args, trainData = selectedTask3[0], valData = selectedTask3[1], testData = selectedTask3[2],
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= True, plotLearningCurve = False, bias_init= initBias[2] )
    
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    
    return trained_model, best_fineTuned_model_1, best_fineTuned_model_2, bestLosses_main, bestLosses_tune1, bestLosses_tune2


#%%

def transferLearning(ordered_tasks: dict, domains_order: list, args, savingDir):
    
    startTime2 = datetime.now().strftime('%H.%M.%S')
    selectedTask1, selectedTask2, selectedTask3 = ordered_tasks['data']
    name1, name2, name3 = ordered_tasks['name']
    
    if args.fineTune_targetBias:
        initBias = [domain_means[domains_order[0]-1], domain_means[domains_order[1]-1], domain_means[domains_order[2]-1]]
    else:
        initBias = [90,90,90]

    # Main Training
    print(f"--------------【Train {name1}】--------------")
    model, trained_model, bestLosses_main = pureVM_main(
        args, trainData = selectedTask1[0], valData = selectedTask1[1], testData = selectedTask1[2],
        savingDir = savingDir, fineTune = False, trained_model = None, verbose= False, plotLearningCurve = True, bias_init= initBias[0])
    torch.save(trained_model.state_dict(), f'{startTime2} fine tune temp.pt')

    # Fine Tune 1
    print(f"----【Fine Tune {name1} Model on {name2}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_1, best_fineTuned_model_1, bestLosses_tune1 = pureVM_main(
        args, trainData = selectedTask2[0], valData = selectedTask2[1], testData = selectedTask2[2], 
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= False, plotLearningCurve = True, bias_init= initBias[1] )

    # Fine Tune 2
    print(f"----【Fine Tune {name1} Model on {name3}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_2, best_fineTuned_model_2, bestLosses_tune2 = pureVM_main(
        args, trainData = selectedTask3[0], valData = selectedTask3[1], testData = selectedTask3[2],
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= False, plotLearningCurve = True, bias_init= initBias[2] )
    
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    
    return trained_model, best_fineTuned_model_1, best_fineTuned_model_2, bestLosses_main, bestLosses_tune1, bestLosses_tune2

#%%
domain_means = [151.24, 73.08, 79.97]

def multitask_fineTune(ordered_tasks: dict, args, savingDir):
    
    startTime2 = datetime.now().strftime('%H.%M.%S')
    pooledTasks, task1, task2, task3 = ordered_tasks['data']
    name0, name1, name2, name3 = ordered_tasks['name']

    # Main Training
    print(f"\n--------------【Train {name0}】--------------")
    last_model, trained_model, bestLosses_main = pureVM_main(
        args, trainData = pooledTasks[0], valData = pooledTasks[1], testData = pooledTasks[2],
        savingDir = savingDir, fineTune = False, trained_model = None, verbose= False, plotLearningCurve = True)
    torch.save(trained_model.state_dict(), f'{startTime2} fine tune temp.pt')

    # Fine Tune 1
    print(f"\n----【Fine Tune {name0} Model on {name1}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_1, best_fineTuned_model_1,  bestLosses_tune1 = pureVM_main( 
        args, trainData = task1[0], valData = task1[1], testData = task1[2], 
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= False, plotLearningCurve = True, bias_init = domain_means[0] )

    # Fine Tune 2
    print(f"\n----【Fine Tune {name0} Model on {name2}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_2, best_fineTuned_model_2, bestLosses_tune2 = pureVM_main(
        args, trainData = task2[0], valData = task2[1], testData = task2[2],
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= False, plotLearningCurve = True, bias_init = domain_means[1] )
    
    # Fine Tune 3
    print(f"\n----【Fine Tune {name0} Model on {name3}】----")
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    fineTuned_model_3, best_fineTuned_model_3,  bestLosses_tune3 = pureVM_main(
        args, trainData = task3[0], valData = task3[1], testData = task3[2],
        savingDir = savingDir, fineTune = True, trained_model = trained_model, verbose= False, plotLearningCurve = True, bias_init = domain_means[2] )
    
    trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
    
    return trained_model, best_fineTuned_model_1, best_fineTuned_model_2, best_fineTuned_model_3, bestLosses_main, bestLosses_tune1, bestLosses_tune2, bestLosses_tune3

#%%

def KQ_fineTuneTest(trained_model, args, domain_KQsampler, savingDir):
    
    print(f"\n----【Fine Tune on Domain {args.noLeak_tuningDomain}】----")
    
    startTime2 = datetime.now().strftime('%H.%M.%S')
    torch.save(trained_model.state_dict(), f'{startTime2} fine tune temp.pt') 
    
    mtftt = True if args.experiment in ['metaTune', 'meta_multiExpTune'] else False
    
    if args.no_leak:
        
        trainLosses, valLosses, testLosses = [], [], []
        
        for expID in range(args.fineTune_numExps):
            
            print(f"\n- - - - - -【Exp no.{expID}】- - - - - -")
            
            TestTask = domain_KQsampler.sampleKQ(args)[args.noLeak_tuningDomain-1]
    
            trained_model.load_state_dict(torch.load(f'{startTime2} fine tune temp.pt'))
            
            fineTuned_model, best_fineTuned_model,  bestLosses_tune = pureVM_main( 
                args, trainData = TestTask[0], valData = TestTask[1], testData = TestTask[2], 
                savingDir = savingDir, fineTune = True, metaFineTune= mtftt, trained_model = trained_model, 
                verbose= False, plotLearningCurve = True, bias_init = domain_means[args.noLeak_tuningDomain-1] )
            
            trainLosses += [round(bestLosses_tune[0],3)]
            valLosses   += [round(bestLosses_tune[1],3)]
            testLosses  += [round(bestLosses_tune[2],3)]
            
        exp_losses = pd.DataFrame({'train':trainLosses, 'valid': valLosses, 'test': testLosses})
        
        exp_losses_desc = exp_losses.describe()
        print(exp_losses_desc)
        
        return exp_losses, exp_losses_desc