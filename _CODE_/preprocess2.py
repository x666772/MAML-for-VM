# -*- coding: utf-8 -*-

import pandas as pd
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.preprocessing import scale, normalize

# Visualization
from tqdm import tqdm, trange

# Directory
import glob


#%% Read Data

def read_files(X_pattern , y_path):
    csv_list = glob.glob(X_pattern)
    df = pd.DataFrame()
    pbar = tqdm(csv_list)
    for f in pbar:
        file = pd.read_csv(f)
        df = pd.concat([df, file], ignore_index= True)

        pbar.set_postfix({'file': f.split("/")[-1], 'shape': file.shape})
    y = pd.read_csv(y_path)
    print('Shapes : ', df.shape, y.shape)
    return df, y


#%% Load Preprocessed

def read_nested_DF(path, parent_name, nested_col, encoded_cols):
    print( f'Loading {parent_name} ...')
    SVID_list = []
    grouped_DF = pd.read_csv(f'{path}{parent_name}.csv', index_col = 0)
    
    for i in trange(grouped_DF.shape[0]):
        SVID = pd.read_csv(f'{path}{parent_name}_{nested_col}/{parent_name}_{nested_col}_{i}.csv', index_col = 0)
        SVID[encoded_cols] = SVID[encoded_cols].astype('uint8')
        SVID_list.append(SVID)
    
    grouped_DF.insert(3, 'SVID', SVID_list)
    return grouped_DF



#%% Preprocess phase2
def removeOutlier(dataset, outlier_MRR_threshold):
    outlier_removed_df = pd.DataFrame()
    for i in range(dataset.shape[0]):
        if dataset.iloc[i]['AVG_REMOVAL_RATE'] < outlier_MRR_threshold:
            outlier_removed_df = pd.concat( [ outlier_removed_df, dataset.iloc[i] ] , axis = 'columns' )
        else:
            print('Outlier Removed: waferID=', dataset.iloc[i]['WAFER_ID'], 'stage=', dataset.iloc[i]['STAGE'])
    return outlier_removed_df.T

def resample_dataset_SVIDs(dataset, mode: str, toLen: int ): # mode: 'trim', 'uniform'
    
    if mode == 'skip':
        return dataset
    
    series_resampler = TimeSeriesResampler(sz=toLen)
    resampledSVIDs = [] 
    
    if mode == 'trim':
        for i in tqdm(dataset.index):
            if dataset.loc[i]['Len'] > toLen:  
                resampledSVID = pd.DataFrame(index = range(dataset.loc[i]['SVID'].index[0], dataset.loc[i]['SVID'].index[0]+toLen)) 
                for col_name in dataset.loc[i]['SVID'].columns:
                    oneCol = series_resampler.fit_transform(dataset.loc[i]['SVID'][col_name].values)[0,:,0]
                    resampledSVID[col_name] = oneCol
            else:
                resampledSVID = dataset.loc[i]['SVID']
            resampledSVIDs += [resampledSVID]
    
    elif mode == 'uniform':

        for i in tqdm(dataset.index):
            
            resampledSVID = pd.DataFrame(index = range(dataset.loc[i]['SVID'].index[0], dataset.loc[i]['SVID'].index[0]+toLen)) 
            for col_name in dataset.loc[i]['SVID'].columns:
                oneCol = series_resampler.fit_transform(dataset.loc[i]['SVID'][col_name].values)[0,:,0]
                resampledSVID[col_name] = oneCol
            
            resampledSVIDs += [resampledSVID]
    
    else: 
        raise Exception("Unprogrammed mode !")
    
    dataset['SVID'] = resampledSVIDs
    return dataset

def pad_grouped_seq(grouped_list, maxLen:int, toMax=False):
    if toMax:
        maxLen = []
        for grouped in  grouped_list:
            maxLen += [max(grouped['Len'])]
        maxLen = max(maxLen)
        print(maxLen)

    for grouped in grouped_list:
        for i in tqdm(range(grouped.shape[0])):
            seq = grouped.iloc[i]
            svid = seq['SVID']
            padLen = maxLen - svid.shape[0]
            zeros = pd.DataFrame(0, index=range(padLen), columns=svid.columns)
            svid = svid.append(zeros, ignore_index=True)
            seq['SVID'] = svid
            grouped.iloc[i] = seq

    return grouped_list

def dropSVIDCol(grouped_datasets: list, drop_col: list):
    if len(drop_col) > 0:
        for dataset in grouped_datasets:
            for i in range(dataset.shape[0]):
                dataset.iloc[i]['SVID'].drop( columns= drop_col , inplace= True)
    return grouped_datasets

def computeStatistics(grouped_datasets: list, computeCols: list, stats_list: list, TQDM=True):
    
    for dataset in grouped_datasets:
        statistics_list = []
        pbar = tqdm(range(dataset.shape[0])) if TQDM else range(dataset.shape[0])
        for i in pbar:
            statistics = dataset.iloc[i]['SVID'][computeCols].describe()
            statistics = statistics.loc[stats_list]
            statistics_list += [statistics]
        dataset['Statistics'] = statistics_list

    stats_size = len(computeCols) * len(stats_list)
    return grouped_datasets, stats_size

#%%
def splitDomain_restusture(nested_datasets, groupBy, groups, restructure=True, verbose=True):
    # restructure: DF[DF] -> List[Dict]
    split_datasets = {}
    for dataset in nested_datasets:
        one_orig_split = []

        if groupBy == None:
            if restructure:
                for i in range(dataset['data'].shape[0]):
                    one_orig_split.append(dataset['data'].iloc[i].to_dict())
                    
            else:
                one_orig_split = dataset['data']
        else:
            for group in groups:
                one_new_split = dataset['data'].groupby(groupBy).get_group(group)
                if restructure:
                    one_listed_new_split = []
                    for i in range(one_new_split.shape[0]):
                        one_listed_new_split.append(one_new_split.iloc[i].to_dict())
                else:
                    one_listed_new_split = one_new_split
                one_orig_split.append(one_listed_new_split)
                
        split_datasets[dataset['name']] = one_orig_split
    
    if verbose:
        print('Split Domain and Convert DF[DF] to List[Dict] complete')
    return split_datasets

#%%

def normalize_DFs(DF_list,    # normalize as globaly as possible 
                  normalizeCols,
                  norm = str  ,    # 'l1'-> abs.values add up to 1 ; 'l2'-> squared.values add up to 1 ; 'max' = max=1
                  axis = 0      ):   # 0 -> feature-wise ; 1 -> sample-wise
   
    normalized_list, norms_list = [], []
    
    currentCols = DF_list[0].columns
    
    otherCols = [i for i in currentCols if i not in normalizeCols]

    for DF in DF_list:
        
        toNormDF = DF[normalizeCols]
        restDF = DF[otherCols]
        index = DF.index
        columns = toNormDF.columns
        
        if norm == 'standardize':
            normalized = scale( X= toNormDF, axis= axis)
            norms = None
        else:
            normalized, norms = normalize(X= toNormDF, norm = norm , axis=axis, return_norm= True)
         
        normalized = pd.DataFrame(data= normalized, index=index, columns=columns)

        normalized = pd.concat( [normalized, restDF], axis='columns')

        normalized_list += [normalized]
        norms_list += [norms]

    return [normalized_list, norms_list]

#%% Check variables of lists are the same
def check_vars(check_list: list, restructured= True, verbose=True):
    equal = True
    vars = []
    dataset_sizes = []
    for alist in check_list:
        
        if restructured:
            cols = list(alist[0]['SVID'].columns.values)
            size = len(alist)
        else:
            cols = list(alist.iloc[0]['SVID'].columns.values)
            size = alist.shape[0]
        vars.append(cols)
        dataset_sizes.append(size)
    for v in vars[1:]:
        if v != vars[0]:
            print('Unequal Varirables!')
            equal = False
    if equal:
        if verbose:
            print('variables are the same!')
        return vars[0], dataset_sizes
    else:
        for v in vars:
            print(v)
        raise Exception('Unequal Varirables!')

#%%
DB = 'G:/其他電腦/MacBook Pro/PHM Data Challenge 2016 (phm_cmp_removal_rates)'

train_X_pattern = DB +'/2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data/training/*.csv'
train_y_path    = DB +'/2016 PHM DATA CHALLENGE CMP DATA SET/CMP-training-removalrate.csv'

val_X_pattern   = DB +'/2016 PHM DATA CHALLENGE CMP VALIDATION DATA SET/validation/*.csv'
val_y_path      = DB +'/PHM16TestValidationAnswers/orig_CMP-validation-removalrate.csv'

test_X_pattern  = DB +'/2016 PHM DATA CHALLENGE CMP DATA SET/CMP-data/test/*.csv'
test_y_path     = DB +'/PHM16TestValidationAnswers/orig_CMP-test-removalrate.csv'

'''
print('\nreading files...')
train, y_train = read_files(train_X_pattern , train_y_path)
val  , y_val   = read_files(val_X_pattern   , val_y_path)
test , y_test  = read_files(test_X_pattern   , test_y_path)
'''

#%%

def execute_preprocess_phase2(args, config_preprocess2):
    
    Path = f'Preprocessed Data {args.preprocessedV}/'
    Nested_col = 'SVID'
    Encoded_Cols = ["STAGE_A","CHAMBER_1.0","CHAMBER_2.0","CHAMBER_3.0","CHAMBER_4.0","CHAMBER_5.0","CHAMBER_6.0"]

    print('\nloading preprocessed data...')
    train_grouped = read_nested_DF(path= Path, parent_name= 'train_grouped', nested_col= Nested_col , encoded_cols= Encoded_Cols)
    val_grouped   = read_nested_DF(path= Path, parent_name= 'val_grouped'  , nested_col= Nested_col , encoded_cols= Encoded_Cols)
    test_grouped  = read_nested_DF(path= Path, parent_name= 'test_grouped' , nested_col= Nested_col , encoded_cols= Encoded_Cols)
    
    Outlier_MRR_Threshold   = config_preprocess2['Outlier_MRR_Threshold']
    resampleMode            = config_preprocess2['resampleMode']
    ToLen                   = config_preprocess2['ToLen']
    Drop_Col                = config_preprocess2['Drop_Col']
    statsList               = config_preprocess2['statsList']
    ComputeCols             = config_preprocess2['ComputeCols']
    
    print('\nremoving ouliers...')
    train_grouped = removeOutlier(train_grouped , Outlier_MRR_Threshold)
    val_grouped   = removeOutlier(val_grouped   , Outlier_MRR_Threshold)
    test_grouped  = removeOutlier(test_grouped  , Outlier_MRR_Threshold)


    print('\nresampling SVIDs...')
    train_grouped = resample_dataset_SVIDs(dataset = train_grouped , mode = resampleMode , toLen = ToLen)
    val_grouped   = resample_dataset_SVIDs(dataset = val_grouped   , mode = resampleMode , toLen = ToLen)
    test_grouped  = resample_dataset_SVIDs(dataset = test_grouped  , mode = resampleMode , toLen = ToLen)
    
    '''
    print('padding SVIDs...')
    [train_grouped, val_grouped, test_grouped] = pad_grouped_seq([train_grouped, val_grouped, test_grouped], maxLen= MaxLen)
    '''

    print('dropping columns...')
    [train_grouped, val_grouped, test_grouped] = dropSVIDCol( grouped_datasets= [train_grouped, val_grouped, test_grouped] , drop_col= Drop_Col)
    
    
    if args.with_stats:
        print('\ncomputing statistical features...')
        [train_grouped, val_grouped, test_grouped] , Stats_Size, = computeStatistics( grouped_datasets= [train_grouped, val_grouped, test_grouped] ,  
                                                                       computeCols= ComputeCols , stats_list= statsList)
        
    
    Nested_Datasets = [ { 'name': 'train', 'data': train_grouped}, 
                        { 'name': 'val'  , 'data': val_grouped  }, 
                        { 'name': 'test' , 'data': test_grouped }    ]
    
    # No split
    GroupBy = None
    Groups  = None
    split_datasets = splitDomain_restusture(Nested_Datasets , GroupBy, Groups, restructure = True)
    
    [train_listed, val_listed, test_listed] = split_datasets.values()
    
    all_datasets = [train_listed, val_listed, test_listed]
    
    
    # Split Stage & Chambers Group
    GroupBy = ['STAGE', 'CHAMBERS_GROUP']
    Groups  = [('A',0), ('A',1), ('B',1)]
    
    split_datasets = splitDomain_restusture(Nested_Datasets , GroupBy, Groups, restructure = False)
    
    [[support_1, support_2, support_3], 
     [query_1  , query_2  , query_3  ],
     [test_1   , test_2   , test_3   ]] = split_datasets.values()
    
    all_datasets_df = [support_1, query_1, test_1, support_2, query_2, test_2, support_3, query_3, test_3]
    
    var_list, dataset_sizes = check_vars(all_datasets_df, restructured=False)
    
    
    # Split Stage & Chambers Group
    GroupBy = ['STAGE', 'CHAMBERS_GROUP']
    Groups  = [('A',0), ('A',1), ('B',1)]
    
    split_datasets = splitDomain_restusture(Nested_Datasets , GroupBy, Groups, restructure=True)
    
    [[support_1_list, support_2_list, support_3_list], 
     [query_1_list  , query_2_list  , query_3_list  ],
     [test_1_list   , test_2_list   , test_3_list   ]] = split_datasets.values()
    
    all_datasets_list = [support_1_list, query_1_list, test_1_list, support_2_list, query_2_list, test_2_list, support_3_list, query_3_list, test_3_list]
    
    var_list, dataset_sizes = check_vars(all_datasets_list, restructured=True)
    
    return all_datasets, all_datasets_df, all_datasets_list 

