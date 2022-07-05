from dataset_manager import DataManager
def plot_psd_sub(sub1, sub2, sub2_name, avg = False):
    fig1, axs1 = plt.subplots(2)
    fig2, axs2 = plt.subplots(2)
    fig3, axs3 = plt.subplots(2)
    fig4, axs4 = plt.subplots(2)
    fig5, axs5 = plt.subplots(2)
    figs = [fig1, fig2, fig3, fig4, fig5]
    axes = [axs1, axs2, axs3, axs4, axs5]
    trials = [1,3,4,5,6]
    
    for i, (fig, ax, trial) in enumerate(zip(figs, axes, trials)):
        fig.suptitle('Trial {} :LYF VS {} PSD'.format(trial, sub2_name), fontsize = 20)
        fig.set_size_inches(15,8)
        sub1[i].plot_psd(tmax = 210, fmax = 50, ax = ax[0], n_fft=2048, average = avg)
        ax[0].set_title('LYF')
        sub2[i].plot_psd(tmax = 210, fmax = 50, ax = ax[1], n_fft=2048, average = avg)
        ax[1].set_title(sub2_name)
        fig.savefig("lyf_{}_t{}_psd".format(sub2_name.lower(), trial))
    # plt.show(block= True)
    
def plot_psd_class(sub1, sub2, onsets ,sub2_name, class_name, trial, avg=False):
    if not isinstance(onsets, list):
        raise ValueError("Please provide list of tmin and max")
    fig, axs = plt.subplots(2,3)
    fig.suptitle('Trial {}:PSD of `{}` class (LYF VS {})'.format(trial, class_name, sub2_name), fontsize = 20)
    fig.set_size_inches(15,8)
    
    
    #concatenate same class into long array
    # sub1.time_as_index
    
    for i, onset in enumerate(onsets):
        print(onset)
        sub1.plot_psd(tmin = onset, tmax = onset + 5, fmax = 50, ax = axs[0, i], n_fft=2048, average=avg)
        axs[0, i].set_title('LYF - PSD `{}` Class at {}-{}s'.format(class_name, onset, onset+5))
        sub2.plot_psd(tmin = onset, tmax = onset + 5, fmax = 50, ax = axs[1, i], n_fft=2048, average=avg)
        axs[1, i].set_title('{} - PSD `{}` Class at {}-{}s'.format(class_name, sub2_name, onset, onset+5))
        fig.savefig("lyf_{}_t{}_{}_psd".format(sub2_name.lower(), trial, class_name))
    # plt.show(block= True)

test_class = DataManager(subjs_id = {"Lai":"LYF", "Suguro":"SGR", "Takahashi":"TKH", "test":"TES", "Sugiyama":"SGYM"})
lai_data, _ = test_class.load_dataset(subjs=['Lai'], trial=[1,2, 3, 4, 5, 6])
lai = test_class.mne_raw(lai_data)

sgr_data, _ = test_class.load_dataset(subjs=['Suguro'], trial=[1,2,3,4,5,6])
sgr = test_class.mne_raw(sgr_data)

tkh_data, _ = test_class.load_dataset(subjs=['Takahashi'], trial=[1,2, 3,4,5,6])
tkh = test_class.mne_raw(tkh_data)

test_data, _ = test_class.load_dataset(subjs=['test'], trial=[1,2,3,4,5,6])
test = test_class.mne_raw(test_data)

sgym_data, _ = test_class.load_dataset(subjs=['Sugiyama'], trial=[1,2,3,4,5,6])
sgym = test_class.mne_raw(sgym_data)

# ps = [16, 62, 153]
# bs = [40, 131, 178]
# bts = [85,107,201]

import matplotlib.pyplot as plt

plt.ion()
# # plt.figure(figsize=(100, 100), dpi=120)

    
    
    
# # sgym[0].plot_psd(tmax = 210, fmax=50, average=True)
# # plt.savefig("TEST")
# # plot_psd_sub(lai, sgr, "SGR")
# # plot_psd_sub(lai, tkh, "TKH")
# # plot_psd_sub(lai, test, "TEST")
# # plot_psd_sub(lai, sgym, "SGYM")
# subjs = [sgr, tkh, test, sgym]
# names = ["SGR", "TKH", "TEST", "SGYM"]

# for subj, name in zip(subjs, names):
#     plot_psd_class(lai[0], subj[0], ps, name, "pen", 1, avg = True)
#     plot_psd_class(lai[1], subj[1], ps, name, "pen", 3, avg = True)
#     plot_psd_class(lai[2], subj[2], ps, name, "pen", 4, avg = True)
#     plot_psd_class(lai[3], subj[3], ps, name, "pen", 5, avg = True)
#     plot_psd_class(lai[4], subj[4], ps, name, "pen", 6, avg = True)

# for subj, name in zip(subjs, names):
#     plot_psd_class(lai[0], subj[0], bs, name, "book", 1, avg = True)
#     plot_psd_class(lai[1], subj[1], bs, name, "book", 3, avg = True)
#     plot_psd_class(lai[2], subj[2], bs, name, "book", 4, avg = True)
#     plot_psd_class(lai[3], subj[3], bs, name, "book", 5, avg = True)
#     plot_psd_class(lai[4], subj[4], bs, name, "book", 6, avg = True)

# for subj, name in zip(subjs, names):
#     plot_psd_class(lai[0], subj[0], bts, name, "bottle", 1, avg = True)
#     plot_psd_class(lai[1], subj[1], bts, name, "bottle", 3, avg = True)
#     plot_psd_class(lai[2], subj[2], bts, name, "bottle", 4, avg = True)
#     plot_psd_class(lai[3], subj[3], bts, name, "bottle", 5, avg = True)
#     plot_psd_class(lai[4], subj[4], bts, name, "bottle", 6, avg = True)

def extract_class_signal(data, onsets, stacked = False):
    '''
    extract class's signal and concatenate together
    if stacked is set to False
    '''
    import numpy as np
    sfreq = data.info['sfreq']

    d = np.array([])
    for onset in onsets:
        start, stop = (np.array([onset - 1, onset + 7]) * sfreq).astype(int)
        tmp = data.get_data(start=start, stop=stop, picks = ['eeg'])
        d = np.hstack([d, tmp]) if d.size else tmp
    
    return d

import mne  
def plot_psd_class_2(data, onsets, classes_name, sname = '', trial = 1):
      
    fig, axs = plt.subplots(1,1)
    fig.set_size_inches(15,8)
    fig.suptitle('{} - All {} signal`s PSD in trial {}'.format(sname, classes_name, trial),  fontsize=20)
    data = extract_class_signal(data, onsets)
    print(data.shape)
    t = mne.io.RawArray(data, mne.create_info(19, 500, ch_types='eeg'))
    t.plot_psd(fmax = 50, n_fft=2048, average=True, ax = axs)
    fig.savefig("{}/{}_{}{}_psd".format(classes_name, sname.lower(), classes_name.lower(), trial))

    # plt.show()
    
ps = [16, 62, 153]
bs = [40, 131, 178]
bts = [85,107,201]


trials = [1,2,3,4,5,6]
subs = [lai, sgr, tkh, test, sgym]
n = ['lai', 'sgr', 'tkh', 'test', 'sgym']

for sub, name in zip(subs,n):
    for i, trial in enumerate(trials):
        print(sub, i)
        plot_psd_class_2(sub[i], ps, 'Pen', name, trial)
        plot_psd_class_2(sub[i], bs, 'Book', name, trial)
        plot_psd_class_2(sub[i], bts, 'Bottle', name, trial)
