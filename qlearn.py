# target = 1
# start of episode
# if epilon < random
#   take random action at random state
# else
#   get best q
# step and get reward (train the model using custom1dcnn, using val acc)
# update q using bell man
# store exp replay
# if r start declining, 3 times, stop
# if concatenate ch larger than 4 stop

# import processor
# from model_set import Custom1DCNN

# subject = [i for i in range(1,20 + 1)]
# runs = [3,7,4,6]

# def extract_chs(x, chs):
#     ch_index = {'C3': 8, 'CZ': 10, 'C4': 12, 'FP1': 21, 'FP2': 23, 'F7': 29, 'F3': 31, 'FZ': 33, 'F4': 35, 'F8': 37, 'P3': 48, 'PZ': 50, 'P4': 52, 'O1': 60, 'O2': 62}
#     chs = [ch_index[loc] for loc in ch_index if loc in chs]
#     return x[:, :, chs]
    


# d = processor.load_data(subjects=subject, runs=runs, max_duration=125) #get data from file, max duration is 125s per trial
# X, y = processor.epochs(d, tmin = 0, tmax=2) # tmax for each actionbs
# X = processor.normalize(X)
# y = processor.to_one_hot(y[:,-1])

# print(X.shape, extract_chs(X, ['C4', 'FP1']).shape)

'''
L : FP1 F3 F7 P3 C3 O1
C : CZ FZ PZ
R : FP2 F4 F8 P4 C4 P2


TEST
states = FP1, FP2
actions = C3, C4

possible state = [FP1 C3],[FP1 C4], [FP1, C3, C4], [FP2, C3],[FP2 C4], [FP2 C3, C4]
'''


from itertools import chain, combinations, product

def reset(env):
    return 


a = ['C4', 'C3']
s = ['FP1', 'FP2']

len2 = list(product(s,a))
maxlen = list(product(len2, a))

print(len2, maxlen)

# #pop tupole with 2 state inside
# for i, d in enumerate(t):
#     if 



# target = 1
# start of episode
# if epilon < random
#   take random action at random state
# else
#   get best q
# step and get reward (train the model using custom1dcnn, using val acc)
# update q using bell man
# store exp replay
# if r start declining, 3 times, stop
# if concatenate ch larger than 4 stop
