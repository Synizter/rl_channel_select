import numpy as np
from processor import *


def get(subj = None, runs = None, smote=False):
    exclude = [38, 88, 89, 92, 100, 104]
    if subj is None:
        subjs = [s for s in range(1,109 + 1) if s not in exclude]
    else:
        subjs = [s for s in range(subj[0], subj[1] + 1) if s not in exclude]
    #Task 1 and 2, MI vs MM of left and righht fist
    if runs == None:
        runs = [i for i in range(3, 14 + 1)]


    # ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']
    chs = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'OZ', 'P3', 'P4', 'PZ']
    raws = load_data(subjs, runs, max_duration = 120, chs = chs )
    x,y = epochs(raws)
    y = to_one_hot(y[:,-1])


    #global normalize
    x = normallize(x)

    if smote:
        x_reshaped = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        # apply smote to train data
        # smote
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        x_train_smote_raw, y_train = sm.fit_resample(x_reshaped, y)
        print('classes count')
        print ('before oversampling = {}'.format(y.sum(axis=0)))
        print ('after oversampling = {}'.format(y_train.sum(axis=0)))
        x_train_smote_raw = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/len(chs)), len(chs)).astype(np.float64)
        x = x_train_smote_raw
        y = y_train

    return np.expand_dims(x, axis = 1), y    

if __name__ =="__main__":
    x, y = get(smote=True, runs = [3,4, 7,8, 13,14])
    # X, y = get()

    print(x.shape, y.shape)
    