"""
Loads the recorded and cleaned dataset in the requested way. More explanations are given in the notebook '02_Data_Preparation'.
"""

import os
import numpy as np
import pandas as pd
import pickle
from operator import length_hint
import scipy
from scipy import signal

# The filter experiments can be found in notebook 04_Frequency_Analysis
def butterworth_filter_signal(sig, f_cutoff = 10, f_sample = 120, order = 5):
    """filters signal at specified cutoff frequency using a Butterworth filter of specified order
    :param signal: input signal
    :param f_cutoff: Cutoff frequency
    :param f_sample: original sample rate of signal
    :param order: order of the filter
    :return: filtered signal
    """
    # Normalize the frequency must be between 0 < W < 1. 1 = 100% of Nyquist Freq. 0 = 0% of Nyqist Freq
    w = f_cutoff / (f_sample / 2) 
    b,a = signal.butter(order, w, 'low')
    return signal.filtfilt(b,a, sig)

def downsample_dictionary(dict_s, sample_rate, f_c = None, order = None):
    """
    Filter and interpolate the 120Hz dictionary to a new sample rate. 
    :param dict_s: signal dicitonary with indices and uniform timesteps
    :param sample_rate: new sample rate
    :param f_c:  Cutoff frequency of low pass flter. If none is specified 0.5*sample_rate is taken
    :param order: Order of lowpass filter. default 5

    :return: Dictionary with filtered and downsampled signals
    """
    interp_fs = dict()
    dict_ds = dict() # downsampled dictionary

    # Load original Timesteps
    orig_t = dict_s['Time']
    duration = orig_t[-1] - orig_t[0]

    orig_n = len(orig_t)
    orig_sr = orig_n / (duration)

    # Collect all parameters to convert
    params_s = []
    for k in dict_s.keys():
        if (k != 'sample_rate') and (k != 'Time') and (k!= 'indices'):
            params_s.append(k)
    
    if not f_c:
        f_c = 0.5*sample_rate
    
    if not order:
        order = 5
    print(f'Cutoff frequency: {f_c}')
    for p in params_s:
        temp_signal = butterworth_filter_signal(dict_s[p], f_cutoff = f_c, f_sample = dict_s['sample_rate'], order= order)
        interp_fs[p] = scipy.interpolate.interp1d(orig_t, temp_signal, fill_value='extrapolate')#.copy()

    # Calculate number of samples, start and end times for new sample rate
    start = dict_s['Time'][0]
    end = dict_s['Time'][-1]
    n_duration = end - start
    n_samples = np.floor(sample_rate*n_duration).astype(int)
    end = n_samples / sample_rate + start

    # Create new uniform Time distribution
    uniform_t = np.linspace(start, end, num = n_samples)

    # Signal Interpolation    
    for p in params_s:
        dict_ds[p] = interp_fs[p](uniform_t)

    # update indices:
    old_sr = dict_s['sample_rate']
    new_indices = []
    for i in dict_s['indices']:
        a = int(i[0]/old_sr*sample_rate)
        b = int(i[1]/old_sr*sample_rate)
        new_indices.append([a,b])
    
    #print('old indices:', dict_s['indices'])
    #print('new indices: ', new_indices)
    dict_ds['indices'] = new_indices

    dict_ds['Time'] = uniform_t
    dict_ds['sample_rate'] = sample_rate

    # Print some information about the process
    s_r = len(uniform_t)/(uniform_t[-1] - uniform_t[0])
    n_samples = len(uniform_t)
    print(f'Converted Signals: {params_s}.')
    print(f'\tBefore: Duration: {duration:.3f}s. Avg Sample Rate: {orig_sr:.3f}/s. Number of samples: {orig_n}.')
    print(f'\tAfter: Duration: {end - start :.3f}s. Avg Sample Rate: {s_r:.3f}/s. Number of samples: {n_samples}. Downsampled by factor {sample_rate/orig_sr:.2f}\n')

    return dict_ds



def clean_dict(sig_dict):
    """
    Removes all unwanted windows from the raw data
    """
    cleaned_signals = dict()

    for param in sig_dict.keys():
        if (param != 'sample_rate' and param != 'indices'):
            cleaned_signals[param] = []
        elif( param == 'sample_rate'):
            cleaned_signals[param] = sig_dict[param]

    for ind in sig_dict['indices']:
        for param in sig_dict.keys():
            if (param != 'sample_rate' and param != 'indices'):
                cleaned_signals[param].append(sig_dict[param][ind[0]:ind[1]])

    return cleaned_signals


def load_clean_combine_save(data_path = 'Dataset/Converted_120Hz/', sample_rate = None, f_c = None, order = None, save=False):
    """
    Loads all datasets, cleans them using the specified indices from notebook '02_Data_Preparation' and combines them into a single dictionary.
    """
    files = os.listdir(data_path)

    pickle_name = 'full_data_cleaned.pickle'

    in_dicts = []
    cleaned_dicts = []
    full_dict = dict()

    for f in files:
        print(f)
        with open(data_path + f, 'rb') as infile:
            if sample_rate:
                print(f'Downsampling Dictionary to {sample_rate}Hz')
                in_dicts.append(downsample_dictionary(pickle.load(infile), sample_rate, f_c = f_c, order = order))
            else:
                in_dicts.append(pickle.load(infile))

    for d in in_dicts:
        cleaned_dicts.append(clean_dict(d))

    for k in cleaned_dicts[0].keys():
        if (k != 'sample_rate' and k != 'indices'):
            full_dict[k] = cleaned_dicts[0][k]
            for d in cleaned_dicts[1:]:
                
                full_dict[k] += d[k]
        elif k == 'sample_rate':
            full_dict[k] = in_dicts[0][k]

    if save:
        with open(data_path + pickle_name, 'wb') as handle:
            pickle.dump(full_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return full_dict


def train_val_test_split_dict(sig_dict, test_split = 0.2, val_split = 0.2, param = 'position_0', seed = 0):
    """
    Ensures more or less fair split of the dataset into specified sized sets, without dataleakage.
    Creates splits in unison for the complete dataset
    param sig_dict: full_dictionary with all signals
    param __split: split size of the respective split wrt the whole dataset
    param param: parameter to base the split on. Shouldn't make any difference
    param seed: Sets seed for random shuffle of the used indices
    """
    
    tot_sig_len = sum(a.shape[0] for a in sig_dict[param])
    n_val = tot_sig_len * val_split
    n_test = tot_sig_len * test_split
    n_train = tot_sig_len - n_test - n_val

    np.random.seed(seed)
    rand_idxs = np.arange(length_hint(sig_dict[param]))
    np.random.shuffle(rand_idxs)

    i = 0
    test_split_idx = []
    test_size = 0
    
    while (test_size < n_test):
        test_split_idx.append(rand_idxs[i])
        test_size += sig_dict[param][rand_idxs[i]].shape[0]
        i += 1

    val_split_idx = []
    val_size = 0

    while(val_size < n_val):
        val_split_idx.append(rand_idxs[i])
        val_size += sig_dict[param][rand_idxs[i]].shape[0]
        i+=1

    train_split_idx = []
    train_size = 0
    while (i < rand_idxs.shape[0]):
        train_split_idx.append(rand_idxs[i])
        train_size += sig_dict[param][rand_idxs[i]].shape[0]
        i +=1

    print(f'Train Idxs: {train_split_idx}. \tTotal train length: {train_size}. \n\
Test Idxs: {test_split_idx}. \t\t\t\tTotal test length: {test_size}. \n\
Val Idxs: {val_split_idx}. \t\t\t\t\tTotal val length: {val_size}.')

    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    for k in sig_dict.keys():
        if (k != 'sample_rate' and k != 'indices'):
            train_dict[k] = [sig_dict[k][i] for i in train_split_idx]
            val_dict[k] = [sig_dict[k][i] for i in val_split_idx]
            test_dict[k] = [sig_dict[k][i] for i in test_split_idx]
        elif(k == 'sample_rate'):
            train_dict[k] = sig_dict[k]
            val_dict[k] = sig_dict[k]
            test_dict[k] = sig_dict[k]
    
    return train_dict, val_dict, test_dict



def window_signal(signal, win_len, target_len, stride = 1):
    """
    param signal: input signal 1D as np.array
    param win_len: length of x portion of window
    param target_len: length of target window
    param stride: stride between samples
    return windows: windowed x portions
    return targets: windowed targets
    """

    sig_len = signal.shape[0]
    n = np.ceil((sig_len - (win_len + target_len)) / stride).astype(int)
    if n < 1:
        return np.array([False]), np.array([False])
    
    windows = np.zeros((n, win_len))
    targets = np.zeros((n, target_len))

    i_s = 0                 # start index window
    i_e = win_len           # end index window
    t_s = i_e               # start index target
    t_e = t_s + target_len  # end index target

    for i in range(n):
        windows[i,:] = signal[i_s:i_e]
        targets[i,:] = signal[t_s:t_e]

        i_s += stride
        i_e += stride
        t_s = i_e
        t_e = t_s + target_len

    return [windows, targets]


def create_feature_arrays(sig_dict, feature_params, target_params, win_len=120, target_len = 120, stride = 1, stack = True):
    """
    Creates a Numpy array for every feature, where the data is windowed
    """
    X_dict = dict()
    target_dict = dict()

    for k in set(feature_params + target_params):
        if (k != 'sample_rate' and k != 'indices'):
            X_dict[k] = []
            target_dict[k] = []
            for sig in sig_dict[k]:
                x, target = window_signal(sig, win_len, target_len, stride)
                if x.any():
                    X_dict[k].append(x)
                    target_dict[k].append(target)

            
            X_dict[k] = np.vstack((X_dict[k]))
            target_dict[k] = np.vstack((target_dict[k]))
    
    # Creates Numpy Array without names. The order is the parameter order given in the function
    if stack:
        features = np.stack([X_dict[p] for p in feature_params], axis = 2)
        targets = np.stack([target_dict[p] for p in target_params], axis = 2)
        print(f'Features shape: {features.shape}. \tTargets shape: {targets.shape}.')
        return features, targets

    return X_dict, target_dict


def dataloader(data_path, win_len = 120, target_len = 60, stride = 2, feature_params= ['position_0', 'position_1', 'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w'], 
               target_params=['position_0', 'position_1'], test_split=0.1, val_split=0.1, seed = 3, downsample_rate = 10, stack = True):
    """
    Putting all above functions together to laod the data.plt.plot(np.vstack(a['X_train']))
    :param data_path: path to interpolated dataset
    :param win_len: Signal length of X samples
    :param target_len: Signal length of y samples (targets)
    :param feature_params: features to include in X
    :param target_params: features to include in y (position_0, position_1)
    :param test_split: percentage of data to be used in the test dataset
    :param val_split: percentage of data to be used in the validation dataset
    :param seed: seed for random generator
    :param downsample_rate: Downsample all signals to freq e.g. 10 Hz
    :param stack: creates numpy array and throws away dict keys
    """
    if downsample_rate:
        f_c = downsample_rate/2
    else:
        f_c = None
    full_dict = load_clean_combine_save(data_path, sample_rate = downsample_rate, f_c = f_c, save = False)
    train_dict, val_dict, test_dict = train_val_test_split_dict(full_dict, test_split = test_split, val_split=val_split, seed = seed)
    
    X_train, y_train = create_feature_arrays(train_dict, feature_params, target_params, win_len, target_len, stride, stack)
    X_val, y_val = create_feature_arrays(val_dict, feature_params, target_params, win_len, target_len, stride, stack)
    X_test, y_test = create_feature_arrays(test_dict, feature_params, target_params, win_len, target_len, stride, stack)

    dataset = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'sample_rate': train_dict['sample_rate']
    }

    return dataset
