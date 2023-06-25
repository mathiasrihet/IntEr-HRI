import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from collections import OrderedDict

# Scikit-learn and Pyriemann ML functionalities
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, roc_auc_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP

from itertools import combinations
import random

def inter_run_KFold(n_run, n_fold, seed=None):
    random.seed(seed)
    test_idx_list = np.array(random.sample(list(combinations(range(n_run), 2)), n_fold))
    while np.all(np.unique(test_idx_list) != np.arange(n_run)):
        print('retry')
        test_idx_list = np.array(random.sample(list(combinations(range(n_run), 2)), n_fold))

    cross_val_gen = []
    for test_idx in test_idx_list :
        cross_val_gen.append([[train_idx for train_idx in range(n_run) if train_idx not in test_idx], test_idx.tolist()])

    return cross_val_gen

def get_epoch_params(w_timelength, ch_modality, preprocessing_modality, w_step=1):
    return f'l{int(w_timelength*1000)}_s{w_step}_{ch_modality}_{preprocessing_modality}'


def compare_classif(subject, epoch_in, epoch_out, w_length, subject_list, datadir, w_timelength, ch_modality, preprocessing_modality, w_step, undersample, normalization):
    #Define some params
    if ch_modality == 'all':
        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 
                    'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 
                    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 
                    'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 
                    'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 
                    'PO3', 'POz', 'PO4', 'PO8']

        
    elif ch_modality == '32':
        ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2',
                    'FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6',
                    'TP10','P7','P3','Pz','P4','P8','PO9', 'PO10', 'O1', 'Oz', 'O2']
        
    elif ch_modality == 'best':
        ch_names = ['CP1', 'CPz', 'CP2', 'Cz', 'C1', 'Pz', 'C3', 'CP3']

    n_channels = len(ch_names)

    epoch_params_in = get_epoch_params(w_timelength, ch_modality, preprocessing_modality, w_step=w_step)
    array_in = f'{datadir}\\{epoch_in}_{epoch_params_in}\\{subject_list[subject-1]}'


    epoch_params_out = get_epoch_params(w_timelength, ch_modality, preprocessing_modality, w_step=w_step)
    array_out = f'{datadir}\\{epoch_out}_{epoch_params_out}\\{subject_list[subject-1]}'

    print('training on', array_in)
    print('testing on', array_out)

    X_train_all = np.load(f'{array_in}\\X_all.npy')
    print('X_train_all', X_train_all.shape)
    y_train_all = np.load(f'{array_in}\\y_all.npy')
    print('y_train_all', y_train_all.shape)
    X_test_all = np.load(f'{array_out}\\X_all.npy')
    print('X_test_all', X_test_all.shape)
    y_test_all = np.load(f'{array_out}\\y_all.npy')
    print('y_test_all', y_test_all.shape)

    # print('X_train_all', X_train_all.shape)
    # print('y_train_all', y_train_all.shape)
    # print('X_test_all', X_test_all.shape)
    # print('y_test_all', y_test_all.shape)


    clfs = OrderedDict()
    cv = inter_run_KFold(8, 10, seed=42)

    clfs['Cov + CSP + TS + SVM'] = make_pipeline(Covariances(estimator='oas'), CSP(nfilter=12, log=False), TangentSpace(), SVC(kernel="rbf"))

    result_list = []
    for fold_n, (train_index, test_index) in enumerate(cv):
        
        # Train test split
        X_train = np.concatenate(X_train_all[train_index])
        X_test = np.concatenate(X_test_all[test_index])
        y_train = np.concatenate(y_train_all[train_index])
        y_test = np.concatenate(y_test_all[test_index])
        
        if undersample == True:
            # Balance classes
            rus = RandomUnderSampler()
            counter=np.array(range(0,len(y_train))).reshape(-1,1)
            index,_ = rus.fit_resample(counter,y_train)
            index = index.squeeze()

            X_train = X_train[index]
            y_train = y_train[index]
        
        # Normalization
        if normalization == True :
            X_std = X_train.std(axis=0)
            X_train /= X_std + 1e-8
            
            X_test /= X_std + 1e-8
            
        y_train_short = y_train[:,0]
        y_test_short = y_test[:,0]
        
        #Iterate over classifiers
        print('Fold', fold_n)
        for m in clfs:
            print(m)

            clfs[m].fit(X_train, y_train_short)  
            y_pred = clfs[m].predict(X_test)
            
            #Metrics
            (tn, fp), (fn, tp) = confusion_matrix(y_test_short, y_pred)

            results = {
            #Window parameters
            'w_length':w_length,
            'w_step':w_step,
            'w_train_n':len(y_train),
            'w_test_n':len(y_test),
            'epoch_in':epoch_in,
            'epoch_out':epoch_out,
            'preprocessing_modality':preprocessing_modality,
            'undersample':undersample,
            'normalization': normalization,

            #Classification Scores
            'Balanced accuracy':balanced_accuracy_score(y_test_short, y_pred),
            'Accuracy':accuracy_score(y_test_short, y_pred),
            'Method':m,
            'AUC':roc_auc_score(y_test_short, y_pred),

            'tn':tn,
            'fp':fp,
            'fn':fn,
            'tp':tp,
                
            'test_idxs' : test_index,
            }

            result_list.append(results)
        
    result_df = pd.DataFrame(result_list)  

    return result_df



