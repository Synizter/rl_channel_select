import itertools
import numpy as np
from dataset_manager import DataManager
from network_manager import NetworkManager
from model_set import Custom1DCNN

ch_map = {  'F4': 0, 'C4': 1, 'Pa': 2, 'Cz': 3, 'F3': 4, 'C3': 5, 'P3': 6, 
            'F7': 7, 'T3': 8, 'T5': 9, 'Fp1': 10, 'Fp2': 11, 'T4': 12, 
            'F8': 13, 'Fz': 14, 'Pz': 15, 'T6': 16, 'O2': 17, 'O1': 18 }

noise_ch_keys = ['Fp1', 'Fp2', 'F7', 'F8', 'Fz', 'F3', 'F4', 'T5']
ch_keys = [ch for ch in ch_map.keys() if ch not in noise_ch_keys]

possible_state = itertools.combinations(ch_keys, 4)

def encoded_info(state_list):
    '''
    encode state/action information from either tuple or single string with
    location index for q table eg
                                    ---------------------------------------------
           STATE/ACTION             | 0: 'Fp1' | 1: 'Fp2' | 2: 'F7' |  3: 'F8' |... 
     0: ('C4', 'Pa', 'Cz', 'C3')    |          |          |         |          |
     1: ('C4', 'Pa', 'Cz', 'P3')    |          |          |         |          |
     2: ('C4', 'Pa', 'Cz', 'T3')    |          |          |         |          |
        ...
    '''
    return dict(enumerate(state_list))

def decode_info(info):
    return [ch_map[ch] for ch in info]

def extract_target_ch(raws, target_ch:list):
    return raws[:,:,target_ch]

observation = encoded_info(possible_state)
action = encoded_info(noise_ch_keys)

q_table = np.zeros((len(observation), len(action)))
print(q_table.shape)

data_manager = DataManager(subjs_id = {"Lai":"LYF", "Suguro":"SGR", 
                                       "Takahashi":"TKH", "test":"TES", 
                                       "Sugiyama":"SGYM"},
                           data_path = 'Datasets/') 

raws, labels = data_manager.load_dataset(subjs=['Lai'],trial=[3,4,5,6],combined=True)

X, y = data_manager.numpy(raws, labels, correct_length = False, div_size = 100)


#82 1/4 from all states
from sklearn.model_selection import train_test_split

best_candidate = np.zeros((82, 1))
path = 'model_ckpt/LYF/model_%d'
_id = 0
for i, (obk, obv) in enumerate(zip(observation.keys(), observation.values())): #dictionary key
    for j, (ak, av) in enumerate(zip(action.keys(), action.values())):
        
        # All channel to be used
        extracted = list(obv)
        extracted.append(av)
        # print(extracted, decode_info(extracted))
        train_data = extract_target_ch(X, decode_info(extracted))
        print(train_data.shape, _id)

        #spilt train, test for model
        try:
            X_train, x_val, y_train, y_val = train_test_split(train_data, y, test_size= .2, stratify= y, random_state = 69)
            print('subject data (x_train, x_val, y_train and y_val) :',X_train.shape, y_train.shape, x_val.shape, y_val.shape)
        except Exception as e:
            print("Error", str(e))
        #trian model
        train_manager = NetworkManager([X_train, y_train, x_val, y_val], epochs = 90, batchsize = 64, learning_rate = 1e-3)
        # train_manager = NetworkManager([X_train, y_train, x_val, y_val], epochs = 1, batchsize = 128, learning_rate = 1e-4)
        hist, model, reward = train_manager.get_reward((path % _id))
        
        
        with open('train_log.txt', '+a') as f:
            f.write('{},({}),{}\n'.format(_id,extracted, reward))
        
        if best_candidate[0] <= reward:
            best_candidate = np.sort(best_candidate, kind = 'quicksort')
            best_candidate[0] = reward
            print(extracted)
        
        _id += 1
        

print(best_candidate)

        
    

