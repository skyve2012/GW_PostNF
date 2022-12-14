
import torch
import torch.nn as nn
import numpy as np
import inspect
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.cuda.FloatTensor)
import seaborn as sns
import pandas as pd
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import numpy as np
import scipy
import torch


def input_dict_return(input_fn, args_dict):
    input_args = inspect.getfullargspec(input_fn).args
    if input_args[0] == 'self':
        input_args = input_args[1:]
    return {name: args_dict[name] for name in input_args}
 
class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        return v

    def _cosine_simililarity(self, x, y):
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
def add_noise(input, SNR):
    max_entries = np.asarray(input).max(axis=-1)[np.newaxis].transpose()
    max_entries[max_entries==0.] = 1.
    input = np.asarray(input) / max_entries
    noise = np.random.normal(0., 1. , np.asarray(input).shape)/SNR
    output = input + noise
    stds = np.std(output, axis=1)[np.newaxis].transpose() 
    clean_output = input / stds
    return output / stds, stds, clean_output, noise/stds


def add_noise_svd(input, SNR):
    max_entries = np.asarray(input).max(axis=-1)[np.newaxis].transpose()
    max_entries[max_entries==0.] = 1.
    input = np.asarray(input) / max_entries
    noise = np.random.normal(0., 1. , np.asarray(input).shape)/SNR
    output = input + noise
    stds = np.std(output, axis=1)[np.newaxis].transpose() 
    clean_output = input / stds
    outputs_final = scipy.linalg.svd(output / stds, full_matrices=False)[2][:, :100]
    return outputs_final, stds, clean_output, noise/stds


def add_noise_svd_real_noise(input, SNR, noise_input, idx_low, idx_high):
    noisy_data, stds, clean_output, noise_over_stds = add_noise_real(input, SNR, noise_input, idx_low, idx_high)
    final_noisy = pca_obj.transform(noisy_data) 
    return final_noisy, stds, clean_output, noise_over_stds


def add_noise_real(input, SNR, noise_input, idx_low, idx_high):
    '''
    input: input clean signals: [batch size, signal length]
    SNR: batch of SNRs for each waveform: [batch size, 1]
    noise_input: real noise long piece: [noise signal total length]
    idx_low: starting index for noise waveform
    idx high: end index for noise waveform, notice that the starting index and end index will
    be used to control the location of noise signals to be used given the whole "noise_input"
    
    return: normalized noisy waveform (main use), stds, clean waveform  (normalized by stds), normalized noise
    
    '''

    max_entries = np.asarray(input).max(axis=-1)[np.newaxis].transpose()
    max_entries[max_entries==0.] = 1.
    input = np.asarray(input) / max_entries
    noise_idx = np.random.randint(idx_low, idx_high, size=SNR.shape[0])
    def indexing(start_idx, length=4096):
        return noise_input[start_idx:(start_idx+length)]
    noise = np.asarray(list(map(indexing, noise_idx, [4096]*SNR.shape[0])))
    noise = noise / np.std(noise, axis=1)[np.newaxis].transpose()
    noise = noise/SNR
    output = input + noise
    stds = np.std(output, axis=1)[np.newaxis].transpose() 
    clean_output = input / stds
    return group_band_passing(output / stds), stds, clean_output, noise/stds

def shift_single_wave(input):
    shift_index = np.random.randint(0, 2, size=1)[0]
    if shift_index == 0:
        return input
    left_right_index = np.random.randint(2, size=1)[0]
    if left_right_index == 1: # shift to the right
        return np.hstack((np.zeros(shift_index), input[:-shift_index]))
    elif left_right_index == 0:
        return np.hstack((input[shift_index:], np.zeros(shift_index)))
    else:
        assert left_right_index == 1 or left_right_index == 0
        
def add_shift(input):
    input = np.asarray(input)
    return np.asarray(list(map(shift_single_wave, input)))


def flooding(input_loss, flooding_b):
    if flooding_b == 0.:
        return input_loss
    else:
        return torch.abs(input_loss - flooding_b) + flooding_b

    

    
def train_per_step(model_type, data, label, model, optimizer, noise, cur_epoch,
                   cur_step, args_dict, device, scheduler=None, test=False):
    bak_data = data.type(torch.cuda.FloatTensor)
    
    if model_type in ("NF_new"):
        p_x_y = model
        epoch = cur_epoch
        batch_size = args_dict['batch_size']
        pyro = args_dict["pyro"]
        simCLR = args_dict["simCLR"]
        low_SNR_tmp, high_SNR_tmp = 2.2, 2.8
        tmp_SNR = np.random.uniform(low_SNR_tmp, high_SNR_tmp, data.shape[0])[..., np.newaxis]

        if model_type != "NF_new":
            noisy_data0, stds, clean_data, _ = add_noise_svd(add_shift(data.cpu().numpy()), tmp_SNR)
            noisy_data1, _, _, _ = add_noise_svd(add_shift(data.cpu().numpy()), tmp_SNR)
        else:
            noisy_data0, _, _, _ = add_noise_real(data.cpu().numpy(), tmp_SNR, noise_input=noise,
                                                              idx_low=8192*5, idx_high=noise.shape[0]-5*8192)


            
            noisy_data1, _, _, _ = add_noise_real(data.cpu().numpy(), tmp_SNR, noise_input=noise,
                                                              idx_low=8192*5, idx_high=noise.shape[0]-5*8192)


        tmp_SNR = torch.Tensor(tmp_SNR).type(torch.FloatTensor).to(device)
        labels = label.type(torch.FloatTensor).to(device)
        noisy_data0 = torch.Tensor(noisy_data0).type(torch.FloatTensor).to(device)
        if simCLR:
            noisy_data1 = torch.Tensor(noisy_data1).type(torch.FloatTensor).to(device)
            noisy_data = torch.cat([noisy_data0, noisy_data1], axis=0)
            labels = torch.cat([labels, labels], axis=0)
        else:
            noisy_data = noisy_data0
        
        if not test:
            optimizer.zero_grad()
        if simCLR:
            loss_list, out_flow_model, _ = p_x_y(noisy_data, labels)
            logprob_loss = loss_list[0]
            simCLR_loss_out = loss_list[1]
            g_loss = logprob_loss + simCLR_loss_out #+ 100.*pen_loss
        else:
            g_loss, out_flow_model, _ = p_x_y(noisy_data, labels)
        if not test:
            flooding(g_loss, args_dict['flooding_b']).backward()
            if scheduler is not None:
                scheduler.step(flooding(g_loss, args_dict['flooding_b']))
            optimizer.step()
        
        loss_list, out_flow_model, _ = p_x_y(noisy_data, labels)
    
    
        
        if args_dict['distributed_train']:
            features_for_out = p_x_y.module.fc1(p_x_y.module.resenet_block(p_x_y.module.check_dim(noisy_data)).squeeze(1))
            if not pyro:
                tmp_check_samples_in_train = out_flow_model.sample(300, features_for_out).detach().cpu().numpy()
            else:
                tmp_check_samples_in_train = out_flow_model.sample((300,)).permute(1,0,2).detach().cpu().numpy()
        else:
            features_for_out = p_x_y.fc1(p_x_y.resenet_block(p_x_y.check_dim(noisy_data)).squeeze(1))
            if not pyro:
                tmp_check_samples_in_train = out_flow_model.sample(300, features_for_out).detach().cpu().numpy()
            else:
                ttttt = []
                for i in range(2):
                    ttttt.append(out_flow_model.sample((300,)).permute(1,0,2).detach().cpu().numpy())
                    tmp_check_samples_in_train = np.concatenate(ttttt, axis=1)
        
        if simCLR:
            return {'loss': flooding(g_loss, args_dict['flooding_b']), 'features': p_x_y.features_for_check.detach().cpu().numpy(), 
                    'samples_check': tmp_check_samples_in_train, 'labels': labels.detach().cpu().numpy(),
                   'min_values': tmp_check_samples_in_train.min(axis=0).min(axis=0),
                   'max_values':tmp_check_samples_in_train.max(axis=0).max(axis=0),
                    'logprob_loss': flooding(logprob_loss, args_dict['flooding_b']),
                   'simCLR_loss': flooding(simCLR_loss_out, args_dict['flooding_b']),
                   'noisy_inputs': noisy_data.detach().cpu().numpy(),
                   'output_model': p_x_y}
        else:
            return {'loss': flooding(g_loss, args_dict['flooding_b']), 'features': p_x_y.features_for_check.detach().cpu().numpy(), 
                    'samples_check': tmp_check_samples_in_train, 'labels': labels.detach().cpu().numpy(),
                   'min_values': tmp_check_samples_in_train.min(axis=0).min(axis=0),
                   'max_values':tmp_check_samples_in_train.max(axis=0).max(axis=0),
                   'noisy_inputs': noisy_data.detach().cpu().numpy(),
                   'output_model': p_x_y}

    
    
def peanlty_loss_wrapper(encoded_z1, encoded_z2, normal_data_split, K1=30., K2=10.):
    encoded_normal_norm = (torch.norm(encoded_z1[normal_data_split, ...], dim=-1).mean() + torch.norm(encoded_z2[normal_data_split, ...], dim=-1).mean())/2.
    abnormal_data_split = ~normal_data_split
    encoded_abnormal_norm = (torch.norm(encoded_z1[abnormal_data_split, ...], dim=-1).mean() + torch.norm(encoded_z2[abnormal_data_split, ...], dim=-1).mean())/2.
    return K1/encoded_normal_norm + K2*encoded_abnormal_norm
    
    

    
noise_range_map = {1:[1.5, 3.5], 2:[1.4, 3.5], 3:[1.3, 3.5], 4:[1.2, 3.5], 5:[1.1, 3.5], 6:[1.0, 3.5], 7:[.9, 3.5], 8:[.8, 3.5], 9:[.7, 3.5], 10:[.6, 3.5], 11:[.5, 3.5]}

def adjust_n_epoch(original, scale=4.):
    '''
    adjust the actual epoch w.r.t. a scaler
    
    input:
    1. original epoch 
    2. epoch adjustment scaler
    
    return:
    1. adjusted epoch
    '''
    return int(original//scale)


def low_max_snr(epoch, snr_map):
    '''
    input:
    1. epoch: current epoch
    2. snr_map: a pre-defined snr map that maps from given epoch to a snr range (characterized by low and high snrs)
    return:
    1.[returned low snr, returned high snr] the low and high snrs for snr ranges in CL.
    '''

    if epoch <= adjust_n_epoch(30):
        indicator = 1 
    elif adjust_n_epoch(30) < epoch <= adjust_n_epoch(40):
        indicator = 2
    elif adjust_n_epoch(40) < epoch <= adjust_n_epoch(50):
        indicator = 3
    elif adjust_n_epoch(50) < epoch <= adjust_n_epoch(60):
        indicator = 4
    elif adjust_n_epoch(60) < epoch <= adjust_n_epoch(70):
        indicator = 5
    elif adjust_n_epoch(70) < epoch <= adjust_n_epoch(80):
        indicator = 6
    elif adjust_n_epoch(80) < epoch <= adjust_n_epoch(90):
        indicator = 7
    elif adjust_n_epoch(90) < epoch <= adjust_n_epoch(100):
        indicator = 8
    elif adjust_n_epoch(100) < epoch <= adjust_n_epoch(110):
        indicator = 9
    elif adjust_n_epoch(110) < epoch <= adjust_n_epoch(120):
        indicator = 10
    else:
        indicator = 11   
    return snr_map[indicator]    
    

    
    
def plot_waves(noisy_input):
    #samples, labels = bundle
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(noisy_input)

    
    return fig 
    
def plot_posteriors_different_snrs(samples, labels):
    #samples, labels = bundle
    fig = plt.figure(figsize=(20,20))
    for i in range(4):
        ax = fig.add_subplot(2, 4, i+1)
        posterior_output_for_save = samples[i]
        d = {'x': posterior_output_for_save[:, 0], 'y': posterior_output_for_save[:, 1]}
        df = pd.DataFrame(data=d)
        sns.kdeplot(df.x, df.y, ax=ax)
        sns.rugplot(df.x, color="g", ax=ax)
        sns.rugplot(df.y, vertical=True, ax=ax)
        ax.scatter(labels[i][0], labels[i][1], color='r')
        
    
    return fig    
    


def whiten(strain, interp_psd, dt):
    strain = np.hstack((np.zeros(900), strain, np.zeros(1000)))
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht[1000:-900]

def group_band_passing(inputs):
    inputs = np.asarray(inputs)
    return np.asarray(list(map(band_passing, inputs))).astype(np.float32)



def band_passing(inputs):
    fs = 4096.
    bb, ab = butter(4, [20.*2./fs, 600.*2./fs], btype='band')
    inputs = np.hstack((np.zeros([500]), inputs, np.zeros([500])))
    filtered = filtfilt(bb, ab, inputs)[500:-500]
    filtered = filtered / np.std(filtered)
    return filtered
    
    
def shift_single_wave_fix_step(input, shift_index, left):
    if not left: # shift to the right
        shifted = np.hstack((np.zeros(shift_index), input[:-shift_index]))
        if shifted.shape[0] != 8192:
            return input
        else:
            return np.hstack((np.zeros(shift_index), input[:-shift_index]))
    else:
        shifted = np.hstack((input[shift_index:], np.zeros(shift_index)))
        if shifted.shape[0] != 8192:
            return input
        else:
            return np.hstack((input[shift_index:], np.zeros(shift_index)))
    
def merger_align_single(input):
    last_i = len(input) - 1
    target_i = np.argmax(input)
    if last_i - target_i >= 500:
        return shift_single_wave_fix_step(input, last_i - target_i - 500, False)
    else:
        return shift_single_wave_fix_step(input, 500 - (last_i - target_i), True)
    
def merger_align(inputs):
    inputs = np.asarray(inputs)
    return np.asarray(list(map(merger_align_single, inputs))).astype(np.float32)
    
def psd_cal(input_signal, smooth=False):
    fs = 4096
    dt = 1./fs 
    fband=[30.0, 300.0]
    NFFT = 4*fs
    Pxx_H1, freqs = mlab.psd(input_signal[10*8192:-10*8192], Fs = fs, NFFT = NFFT)
    psd_H1 = interp1d(freqs, Pxx_H1)
    Pxx = (1.e-22*(18./(0.1+freqs))**2)**2+0.7e-23**2+((freqs/2000.)*4.e-23)**2
    psd_smooth = interp1d(freqs, Pxx)
    if not smooth:
        return psd_H1
    else:
        return psd_smooth


def wave_downsampling(input_signal):
    input_signal = scipy.signal.resample(input_signal[50*8192*2:-50*8192*2], int(input_signal[50*8192*2:-50*8192*2].shape[0] / 2))
    
    return input_signal    
    