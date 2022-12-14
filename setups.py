import h5py
from utils import *
from torch.utils.data import Dataset
from scipy.signal import resample
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
import copy
import torch

### some meta info for setups ###

signal_data_path_map = {'GW150914': 'new_waveforms_clean5.h5'}

signal_data_path_map_4k = {'GW150914': 'small_data_for_github.h5'}


noise_data_path_map = {'GW150914': 'H-H1_GWOSC_16KHZ_R1-1126257415-4096.hdf5'}


long_real_data_path_map = {'GW150914': 'H-H1_GWOSC_16KHZ_R1-1126257415-4096.hdf5'}


long_real_data_path_map_4k = {'GW150914': 'H-H1_GWOSC_4KHZ_R1-1126257415-4096.hdf5'}


noisy_data_raw_path_map_4k = {'GW150914': 'noisy_raw_GW150914_4k'}

noisy_data_raw_path_map = {'GW150914': 'noisy_raw_GW150914'}


### some dataset object ###

class HDF5Dataset(Dataset):
    """Creates a dataset from an hdf5 file.
    Args:
        data_file (Union[str, Path]): Path to hdf5 file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file,
                 mode: str = 'train',
                 in_memory: bool = False):

        self.data_path = data_file
        self.mode = mode
        env = h5py.File(str(data_file), 'r')
        self._num_examples = len(env[self.mode + '_data'])
        env.close()

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        env = h5py.File(str(self.data_path), 'r')
        data = env[self.mode + '_data'][index]
        label = env[self.mode + '_label'][index]
        env.close()
        return data, label[:2]
    
    

### setup files ###
def setup_dataset(dataset_name, data_path, mode="train", low_res=False):
    if dataset_name[:2] == "GW":
        if low_res:
            final_data_path = data_path + signal_data_path_map_4k[dataset_name]
        else:
            final_data_path = data_path + signal_data_path_map[dataset_name]
        print(final_data_path)
        return HDF5Dataset(final_data_path, mode=mode)

    
def setup_model(model_type, args_dict):
        
    if model_type in ("NF_new"):
        from model_lib import PostNF_P_X_Y_pyro_cond_CLR
        if args_dict['pyro']:
            if args_dict['simCLR']:
                model_args = input_dict_return(PostNF_P_X_Y_pyro_cond_CLR, args_dict)
                model = PostNF_P_X_Y_pyro_cond_CLR(**model_args)

    print("#"*30)
    print('current model type and praramter setup')
    print('Model type: {}'.format(model_type))
    print('Model args:')
    print(model_args)
    print("#"*30)
    return model


def setup_noise_piece(dataset_name, data_path):
    noise_file = h5py.File(data_path + noise_data_path_map[dataset_name], 'r')
    duration = noise_file[u'meta'][u'Duration'][()]
    strain_H1 = noise_file[u'strain'][u'Strain'][()]
    fs = 16384 # original
    # resample part
    strain_H1 = resample(strain_H1[50*8192:-50*8192], int(strain_H1[50*8192:-50*8192].shape[0] / 2))
    fs = int(fs/2)
    dt = 1./fs 
    fband=[30.0, 300.0]
    make_psds = 1
    if make_psds:
        NFFT = 4*fs
        Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)

        psd_H1 = interp1d(freqs, Pxx_H1)


    real_noise_LIGO = whiten(strain_H1, psd_H1, dt)[10*8192:-10*8192]
    noise_file.close()
    return real_noise_LIGO



        
        