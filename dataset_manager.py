#@title Data Manager
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple
from tqdm import tqdm

class DataManager:
    def __init__(self, subjs_id:dict, data_path:str = 'gdrive/MyDrive/\u5B9F\u9A13\u30C7\u30FC\u30BF/'):
        self.data_path = data_path
        self.subjs_id = subjs_id
        self.sampling_freq = 500
        self.duration = 5
        self.data_point = self.sampling_freq * self.duration
        self.class_num = {'pen':1, 'book':2, 'bottle':3}
        self.chs = {  'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 
            'F7': 7, 'T3': 8, 'T5': 9, 'Fp1': 10, 'Fp2': 11, 'T4': 12, 
            'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18 }
    def normalize(self, x:np.ndarray):
        scaler = MinMaxScaler()
        norm = scaler.fit_transform(x)
        return norm

    #array中にクラスのデータ（グループの１，２，３）のstart場所を探す関数
    def find_event_onset_index(self, data, search_val, correct_length = True):
        prev = 0
        op = []
        ed = []
        for index, d in enumerate(data):
            if prev == 0 and d == search_val:
                op.append(index)
                # print("start", index)
                prev = search_val
            elif prev == search_val and d == 0:
                prev = 0
                ed.append(index)
                # print("end", index)
            
        if correct_length: #correct data length due to chunk recieving delay, either exceeding or below 2500dp
            pass
            for i, (s, e) in enumerate(zip(op, ed)):
                curr = e - s
                if curr < 2500:
                    ed[i] += np.abs(2500 - curr)
                else:
                    ed[i] -= np.abs(2500 - curr)
        else:
            for i, (s, e) in enumerate(zip(op, ed)):
                ed[i] = s + 2400
        return op, ed

        #start場所で生データからクラスデータを引き出す。sample　freqは500Hzですが、データ数は2500ではないなので（2514とか2530とか2478とか）
        #２４００数のデータを引き出す関数
    def extract_target_signal(self, data, onset_evt_index:tuple, div_size = 2500):
        s = int(onset_evt_index[0])
        e = int(onset_evt_index[1])

        if ((e - s) % div_size) != 0:
            raise ValueError('Cannot reshape target array using {} as divide size, (data length = {} | {}-{})'.format(div_size, e - s, s, e))
        norm_data = self.normalize(data[s:e,:])
        return self.div_data_tensor(norm_data, div_size)

        #モデルの入力は3次元であり、2次元を3次元にする関数
    def div_data_tensor(self, ndarray, div_size):
        return np.reshape(ndarray, (ndarray.shape[0]//div_size, div_size, ndarray.shape[1]))

    def gen_ohe(self, shape, target_class):
        none_class = np.zeros((shape[0], 1), dtype=float)
        tar_class = np.ones((shape[0], 1), dtype=float)
        ohe = np.array([], dtype=float).reshape(shape[0],0)
        for i in range(0, len(self.class_num)):
            if i + 1 == target_class:
                ohe = np.hstack([ohe, tar_class])
            else:
                ohe = np.hstack([ohe, none_class])
        return ohe

    def load_dataset(self, subjs = None, excluded_ch:list = None, trial = None, combined = False)->List[np.array]:
        # if combine is true, return data as long single ndarray
        # defualt, load only channel that least affect by blinking
        #load all aubject
        if subjs == None:
            target_path = [self.data_path + '%s/%s' % (key, self.subjs_id[key]) for key in self.subjs_id.keys()]
        else:
            target_path = [self.data_path + '%s/%s' % (key, self.subjs_id[key]) for key in self.subjs_id.keys() if key in subjs]

        #load all trial
        if trial == None:
            rf = ["T%d_MO.txt" % i for i in range(1,6 + 1)]
            lf =["T%d_MO_LABEL.txt" % i for i in range(1,6 + 1)]
        else:
            rf = ["T%d_MO.txt" % i for i in trial]
            lf =["T%d_MO_LABEL.txt" % i for i in trial]
        
        eeg_data_files = []
        label_files = []
        for p in target_path:
            for r, l in zip(rf,lf):
                eeg_data_files.append(p + r)
                label_files.append(p + l)

        # raws = np.vstack([self.normalize(np.loadtxt(f)) for f in eeg_data_files])
        # labels = np.vstack([np.expand_dims(np.loadtxt(f), axis = 1) for f in label_files])
        raws = []
        labels = []
        
        # for i, line in enumerate(tqdm(f)):
        for i, (raw_file, label_file) in enumerate(tqdm(zip(eeg_data_files, label_files), 
                                                        total = len(eeg_data_files),
                                                        desc = 'File loading progress:')):
            t = np.loadtxt(raw_file)
            if excluded_ch != None:
                remove_index = [self.chs[ch] for ch in excluded_ch]
                print("Remove indices:", remove_index)
                t = np.delete(t, remove_index, axis = 1)
            #remove timestamp
            t = np.delete(t, 0, axis = 1)
            t = self.normalize(t)
            raws.append(t)
            labels.append(np.expand_dims(np.loadtxt(label_file), axis = 1))

        if combined:
            return  np.vstack([raw for raw in raws]), np.vstack([label for label in labels])
        else:
            return raws, labels

    #Return list of 3D tensor array for both data and label, number of data depend on the div_size paramters
    #The order of data in the returned list is PEN, BOOK and BOTTLE repectively
    def numpy(self, raws:List[np.ndarray], labels:List[np.ndarray], div_size = 2500, correct_length = True)->Tuple[np.ndarray, np.ndarray]:
        pen_data = np.array([], dtype=float).reshape(0,div_size,raws.shape[1])
        book_data = np.array([], dtype=float).reshape(0,div_size,raws.shape[1])
        bottle_data = np.array([], dtype=float).reshape(0,div_size,raws.shape[1])

        for key in tqdm(self.class_num.keys(), total = len(list(self.class_num.keys())), desc = "converting class"):
            onset_start, onset_end = self.find_event_onset_index(labels.transpose()[0], self.class_num[key], correct_length=correct_length)
            
            for start, end in zip(onset_start, onset_end):
                d = self.extract_target_signal(raws, (start, end), div_size = div_size) #NOTE: if no length correction apply, please be aware of shape to be converted
                if key == 'pen':
                    pen_data = np.vstack([pen_data, d])
                elif key == 'book':
                    book_data = np.vstack([book_data,d])
                elif key == 'bottle':
                    bottle_data = np.vstack([bottle_data,d])
        
        pen_label = self.gen_ohe(pen_data.shape, self.class_num['pen'])
        book_label = self.gen_ohe(book_data.shape, self.class_num['book'])
        bottle_label = self.gen_ohe(bottle_data.shape, self.class_num['bottle'])
        
            
        return np.vstack([pen_data, book_data, bottle_data]), np.vstack([pen_label, book_label, bottle_label])
    
    def spike_cut(data:np.ndarray):
        pass

    def get_subjects(self):
        return self.subjs_id

        
        
# if __name__ == "__main__":
#     from sklearn.preprocessing import train_test_split
#     # test_class = DataManager(subjs_id = {"Lai":"LYF", "Suguro":"SGR", "Takahashi":"TKH", "test":"TES", "Sugiyama":"SGYM"})
#     # ##TEST CASE 1 ---------------------------------------------------------------------------------------------------------
#     # X, y = test_class.load_dataset() #all subject all trial, all chanels
#     # X, y = test_class.load_dataset(trial = [1,3,4]) #all subject with trial 1 3 and 4
#     # X, y = test_class.load_dataset(subjs = ['Lai', 'Takahashi']) #all 2's subject all data
#     # X, y = test_class.load_dataset(subjs = ['Suguro', 'test', 'Sugiyama'], trial=[1,2,3]) #load 2 subkect and specific trial
#     # raws, labels = test_class.load_dataset(subjs = ['Sugiyama'], trial=[1], excluded_ch = ['Fp1', 'Fp2', 'F7', 'F8', 'Fz', 'F3', 'F4', 'T5']) #exlcude chs
#     # combine flag set False, data is store in list separately
#     # for raw,label in zip(raws, labels):
#     #     d, l = test_class.numpy(raw, label)
#     #     print(d.shape, l.shape)

#     # ##TEST CASE 2 ---------------------------------------------------------------------------------------------------------
#     # raws_combine, labels_conbine = test_class.load_dataset(subjs = ['Sugiyama'], trial=[1], combined = True) #exlcude chs
#     # X, y = test_class.numpy(raws_combine, labels_conbine, correct_length = False, div_size = 100)
#     # print(X.shape, y.shape)
#     excluded_ch = ['Fp1', 'Fp2', 'F7', 'F8', 'Fz', 'F3', 'F4', 'T5']

#     data_manager = DataManager(subjs_id = {"Lai":"LYF", "Suguro":"SGR", "Takahashi":"TKH", "test":"TES", "Sugiyama":"SGYM"})
#     # data_manager = DataManager(subjs_id = {"Lai":"LYF", "test":"TES"})   
#     # subjs = list(data_manager.get_subjects().keys())

#     _raws, _labels = data_manager.load_dataset(trial = [2,3,4,5,6],excluded_ch=excluded_ch, combined = True)
#     _X, _y = data_manager.numpy(_raws, _labels, div_size = 100, correct_length = False)
#     try:
#         X_train, x_val, y_train, y_val = train_test_split(_X, _y, test_size= .2, stratify= _y, random_state = 69)
#         print('subject data (x_train, x_val, y_train and y_val) :',X_train.shape, y_train.shape, x_val.shape, y_val.shape)
#     except Exception as e:
#         print("Error", str(e))
