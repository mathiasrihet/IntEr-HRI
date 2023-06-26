import os
import pandas as pd
import datetime

from windowing_raw import windowing, stack_windows, short_preprocessing
from single_subject_validation import compare_classif
from single_subject_prediction import test_classif

today = datetime.datetime.now().strftime("%d%m%y")

cdir = os.getcwd()
print(cdir)
root = '\\'.join(cdir.split('\\')[:-1])
datadir = f'{root}\\Data'
datasets = ['training data', 'test data']

subject_list = ['AA56D', 'AC17D', 'AJ05D', 'AQ59D', 'AW59D', 'AY63D', 'BS34D', 'BY74D']

ch_modality = '32'
sfreq = 500
w_length = 250
w_step = 50
w_timelength = w_length/sfreq

preprocessing = short_preprocessing
preprocessing_modality = 'preprocessed'

undersample = True
normalization = True

train_test_dict = {
    'cross-validation':('realwindow_training data', 'realwindow_training data'),
    'prediction':('realwindow_training data', 'realwindow_test data'),
}

first_run = True
name = 'test'

if first_run:
    for dataset in datasets :
        for i in range(len(subject_list)):
            epoch_dir, lengths = windowing(subject=i, w_length=w_length, w_step=w_step, 
                                            subject_list=subject_list, datadir=datadir, dataset=dataset, 
                                            preprocessing = preprocessing, channel_to_pick = ch_modality)
            stack_windows(epoch_dir, lengths)


df_list = []
epoch_in, epoch_out = train_test_dict['cross-validation']

for trial in range(10):
    for subject in range(len(subject_list)):
        print('trial', trial, 'subject', subject_list[subject-1])
        df = compare_classif(subject=subject, epoch_in=epoch_in, epoch_out=epoch_out, w_length=w_length, subject_list=subject_list, datadir=datadir, w_timelength=w_timelength, ch_modality=ch_modality, preprocessing_modality=preprocessing_modality, w_step=w_step, undersample=undersample, normalization=normalization)
        df['subject'] = subject_list[subject-1]
        df['trial'] = trial
        df_list.append(df)

cross_val_df = pd.concat(df_list)
cross_val_df.to_csv(f'{root}\\Results\\validation_{ch_modality}_{today}_{epoch_in}_{name}.csv')


df_list = []
epoch_in, epoch_out = train_test_dict['prediction']

for subject in range(len(subject_list)):
    print(subject_list[subject-1])
    df = test_classif(subject=subject, epoch_in=epoch_in, epoch_out=epoch_out, w_length=w_length, subject_list=subject_list, datadir=datadir, w_timelength=w_timelength, ch_modality=ch_modality, preprocessing_modality=preprocessing_modality, w_step=w_step, undersample=undersample, normalization=normalization)
    df['subject'] = subject_list[subject-1]
    df_list.append(df)

pred_df = pd.concat(df_list)
pred_df.to_csv(f'{root}\\Results\\prediction_{ch_modality}_{today}_{epoch_in}_{name}.csv')
        