import os
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

from collections import OrderedDict

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP

def get_epoch_params(w_timelength, ch_modality, preprocessing_modality, w_step=1):
    return f'l{int(w_timelength*1000)}_s{w_step}_{ch_modality}_{preprocessing_modality}'

def pred_from_subdf(subdf):
    preds = []
    tests = []
    for i in range(len(subdf)):
        s = subdf[['test_idxs', 'y_pred', 'y_test']].iloc[i]
        idxs = s['test_idxs']
        y_preds = s['y_pred'].reshape(2, -1)
        y_tests = s['y_test'][:, 0].reshape(2, -1)

        for j in range(2):
            preds.append(pd.Series(y_preds[j], name = idxs[j]))
            tests.append(pd.Series(y_tests[j], name = idxs[j]))

    pred_df = pd.DataFrame(preds).reset_index().groupby(by='index').mean()
    test_df = pd.DataFrame(tests).reset_index().groupby(by='index').mean()
    
    return pred_df, test_df

def get_centroid(a):
    centroid_idx = len(a)/2
    if int(centroid_idx)==centroid_idx:
        out = a[int(centroid_idx)]
    else:
        out = a[int(centroid_idx)]
        
    return out

def cluster_from_pred(a):
    pred_idxs = np.where(a==1)[0]
    cluster_idxs = np.where(np.diff(pred_idxs)!=1)[0]
    clusters = [pred_idxs[cluster_idxs[i]+1:cluster_idxs[i+1]+1] for i in range(len(cluster_idxs)-1)]
    
    return clusters 

def test_classif(subject, epoch_in, epoch_out, w_length, subject_list, datadir, w_timelength, ch_modality, preprocessing_modality, w_step, undersample, normalization):
    detect_peak_on = 'Cz'
    test_index = [elem.split('.')[0][-1] for elem in os.listdir(f'{datadir}\\EEG\\test data\\{subject_list[subject]}\\data') if elem.split('.')[-1]=='vhdr']

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
    channels_to_idx = {ch:i for i, ch in enumerate(ch_names)}

    epoch_params_in = get_epoch_params(w_timelength=w_timelength, ch_modality=ch_modality, preprocessing_modality=preprocessing_modality, w_step=w_step)
    array_in = f'{datadir}\\{epoch_in}_{epoch_params_in}\\{subject_list[subject-1]}'


    epoch_params_out = get_epoch_params(w_timelength=w_timelength, ch_modality=ch_modality, preprocessing_modality=preprocessing_modality, w_step=w_step)
    array_out = f'{datadir}\\{epoch_out}_{epoch_params_out}\\{subject_list[subject-1]}'

    print('training on', array_in)
    print('testing on', array_out)

    X_train_all = np.load(f'{array_in}\\X_all.npy')
    print('X_train_all', X_train_all.shape)
    y_train_all = np.load(f'{array_in}\\y_all.npy')
    print('y_train_all', y_train_all.shape)
    X_test_all = np.load(f'{array_out}\\X_all.npy')
    print('X_test_all', X_test_all.shape)
    # y_test_all = np.load(f'{array_out}\\y_all.npy')
    # print('y_test_all', y_test_all.shape)

    # print('X_train_all', X_train_all.shape)
    # print('y_train_all', y_train_all.shape)
    # print('X_test_all', X_test_all.shape)
    # print('y_test_all', y_test_all.shape)

    clfs = OrderedDict()

    clfs['Cov + CSP + TS + SVM'] = make_pipeline(Covariances(estimator='oas'), CSP(nfilter=12, log=False), TangentSpace(), SVC(kernel="rbf"))

    result_list = []
        
    X_train = np.concatenate(X_train_all)
    X_test = np.concatenate(X_test_all)
    y_train = np.concatenate(y_train_all)
    # y_test = np.concatenate(y_test_all)

    # print(X_train.shape)
    # print(X_test.shape)
    
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
    # y_test_short = y_test[:,0]
    
    #Iterate over classifiers
    for m in clfs:
        print(m)
 
        clfs[m].fit(X_train, y_train_short)
        y_pred = clfs[m].predict(X_test)

        results = {
        #Window parameters
        'w_length':w_length,
        'w_step':w_step,
        # 'w_train_n':len(y_train),
        # 'w_test_n':len(y_test),
        'epoch_in':epoch_in,
        'epoch_out':epoch_out,
        'preprocessing_modality':preprocessing_modality,
        'undersample':undersample,
        'normalization':normalization,
        'Method':m,
        'test_idxs' : test_index,
        }
        
        y_pred = y_pred.reshape(2, -1)
    
    
        for j in range(2):
            test_clusters = cluster_from_pred(y_pred[j])
            cluster_min_size = 1
            next_selected_clusters = [get_centroid(c) for c in test_clusters if len(c) > cluster_min_size]
            while len(next_selected_clusters)>6:
                cluster_min_size+=1
                next_selected_clusters = [get_centroid(c) for c in test_clusters if len(c) > cluster_min_size]
            
            one_channel_train_run = X_test_all[j , :, channels_to_idx[detect_peak_on],:]

            selected_windows = [get_centroid(c) for c in test_clusters if len(c) >= cluster_min_size]
            
            while len(selected_windows) > 6:
                del selected_windows[np.diff(selected_windows).argmin()]
            
            results[f'selected_idxs {j}'] = [w*50 + one_channel_train_run[w].argmax() for w in selected_windows]

        result_list.append(results)
        
    result_df = pd.DataFrame(result_list) 

    return result_df