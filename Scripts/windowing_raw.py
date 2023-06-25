import mne
import os
import numpy as np


def correct_channel_label(ch):
    montage = mne.channels.make_standard_montage('standard_1020').ch_names
    if ch == 'CPZ':
        return 'CPz'
    elif ch == 'POZ':
        return 'POz'
    elif ch not in montage:
        return ch.capitalize()
    else:
        return ch

def data_from_raw(raw):
    sfreq = raw.info['sfreq']
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    error_idx = events[events[:,2]==96][:,0]
    # Move onset from event to expected response (P3b)
    error_idx += int(0.4*sfreq)
    
    data = raw.get_data(picks='eeg')
    labels = np.zeros(data.shape[1])
    labels[error_idx] = 1
    
    return data, labels

def short_preprocessing(w, sfreq):
        #Reref CAR
        w = w - np.stack([w.mean(axis=0)]*w.shape[0], axis=0)

        #Lowpass Filter
        w = mne.filter.filter_data(data=w, sfreq=sfreq, l_freq=None, h_freq=15, verbose=False)

        return w


def windowing(subject, w_length, w_step, subject_list, datadir, dataset, preprocessing = None, channel_to_pick = None):
    fname = f'{datadir}\\EEG\\{dataset}\\{subject_list[subject-1]}\\data\\'
    
    if not os.path.exists(fname):
        print(f'subject {subject_list[subject-1]} not in {datadir}')
    else :
        print('subject', subject_list[subject-1])

    runs = [elem for elem in os.listdir(fname) if elem.split('.')[-1] == 'vhdr']

    # if len(runs) !=8:
    #     print('wrong run number')

    # event_list = []
    run_list = []
    for run in runs:
        raw = mne.io.read_raw(f'{fname}\\{run}', verbose=False)
        mne.rename_channels(raw.info, correct_channel_label, allow_duplicates=False, verbose=None)

        if preprocessing == short_preprocessing:
            preprocessing_modality = 'preprocessed'

        elif preprocessing != None:
            preprocessing_modality = '?'

        else:
            preprocessing_modality = 'raw'
        
        # print(preprocessing_modality)
        
        #channel selection
        if channel_to_pick == '32':
            ch_modality = '32'
            raw.pick_channels(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2',
                    'FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6',
                    'TP10','P7','P3','Pz','P4','P8','PO9', 'PO10', 'O1', 'Oz', 'O2'])
            
        elif channel_to_pick == 'best':
            ch_modality = 'best'
            raw.pick_channels(['CP1', 'CPz', 'CP2', 'Cz', 'C1', 'Pz', 'C3', 'CP3'])
            
        elif type(channel_to_pick) == list :
            ch_modality = 'custom'
            raw.pick_channels(channel_to_pick)
        else:
            ch_modality = 'all'
            print('use all given channels')
        
        run_list.append(raw)

    sfreq = int(run_list[0].info['sfreq'])
    w_timelength = w_length/sfreq
    epoch_dir = f'realwindow_{dataset}_l{int(w_timelength*1000)}_s{w_step}_{ch_modality}_{preprocessing_modality}'

    output_directory = f'{datadir}\\{epoch_dir}\\{subject_list[subject-1]}'
    os.makedirs(output_directory, exist_ok = True)

    lengths = []
    for i, run in enumerate(run_list):
        name = runs[i].split('_')[-1].split('.')[0]
        print(name)
        
        data, labels = data_from_raw(run)
        
        n_channels = data.shape[0]
        n_points = data.shape[1]
        
        X = []
        y = []
        
        # A window smaller than 900ms do not allow 15Hz low-pass
        start = max(w_length, 450) 
        for i in range(start, n_points, w_step):
            w = data[:,i-start:i]
            w_label = labels[i-start:i]

            if preprocessing != None:
                w = preprocessing(w, sfreq)

            #Cropping
            w = w[:,:w_length]
            w_label = w_label[:w_length]

            X.append(w)
            y.append(w_label.sum())
        
        X = np.stack(X)
        y = np.array(y, dtype=int)
        y = np.vstack((y,np.abs(1-y))).T

        lengths.append(len(X))

        np.save(f'{output_directory}\\X_{name}.npy', X)
        np.save(f'{output_directory}\\y_{name}.npy', y)

    return output_directory, np.array(lengths, dtype=int)

def stack_windows(epoch_dir, lengths):
    smallest = lengths.min()
    print(f'crop maximum {lengths.max() - smallest} points')

    print('Saving X_all')
    np.save(f'{epoch_dir}\\X_all.npy', np.stack([np.load(f'{epoch_dir}\\{elem}')[:smallest] for elem in os.listdir(epoch_dir) if elem[0:5]=='X_set']))
    print('Saving y_all')
    np.save(f'{epoch_dir}\\y_all.npy',np.stack([np.load(f'{epoch_dir}\\{elem}')[:smallest] for elem in os.listdir(epoch_dir) if elem[0:5]=='y_set']))



