from hyperparameters import *
from utils import *
import time
from setups import *
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader
import json
import sys

import torch
import torch.distributed
import h5py
import glob
import itertools
###########################

torch.set_default_tensor_type(torch.cuda.FloatTensor)
### create parser for hyperparameters ##
base_parser = create_base_parser()
distributed_train_parser = create_train_parser(base_parser)
args = distributed_train_parser.parse_args()
args_dict = vars(args)
print('all arguments: {}'.format(args_dict))
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_idx
print(os.environ['CUDA_VISIBLE_DEVICES'])
if args.distributed_train:
    torch.distributed.init_process_group(backend="nccl")
    args.local_rank = int(torch.distributed.get_rank())
    args.world_size = torch.distributed.get_world_size()
    device = torch.device("cuda:0".format(args.local_rank))
    args.world_size = torch.distributed.get_world_size()
    print("### global rank of curr node: {}".format(torch.distributed.get_rank()))
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_idx
    device = torch.device("cuda:{}".format(0))
    

### create dataset ###
args_dict_for_dataset_setup = input_dict_return(setup_dataset, args_dict)
dataset_for_train = setup_dataset(**args_dict_for_dataset_setup)
if args.distributed_train:
    pass

# validation dataset 
args_dict_for_dataset_setup_for_test = args_dict_for_dataset_setup
args_dict_for_dataset_setup_for_test["mode"] = "test"
dataset_for_valid = setup_dataset(**args_dict_for_dataset_setup_for_test)

### create model ###
model_for_train = setup_model(args.model_type, args_dict)
if args.distributed_train:
    if isinstance(model_for_train, list):
        model_for_train = [nn.DataParallel(item) for item in model_for_train]
        model_for_train = [item.to(device) for item in model_for_train]
    else:
        model_for_train = nn.DataParallel(model_for_train)
        model_for_train = model_for_train.to(device)
                                          
else:    
    if isinstance(model_for_train, list):
        model_for_train = [item.cuda() for item in model_for_train]
    else:
        model_for_train = model_for_train.cuda()
        
        
### load pretrained model ###
# TODO:
if args.resume_from_checkpoint:
    print('#'*30)
    print('loading pretrianed checkpoints')
    print('#'*30)
    assert args.pretrained_model_checkpoint is not None
    model_for_train.load_state_dict(torch.load(args.pretrained_model_dir + args.pretrained_model_checkpoint))
    print("Finished loading the model weights")


### setup optimizer ###
if args.model_type in ("NF_new"):
    optimizer = torch.optim.AdamW(model_for_train.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

### setup scheduler ###
if args.scheduler and args.model_type in ("NF_new"):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
else:
    scheduler = None

        

### setup folders etc before training ###

model_name = '{}_{}_{}'.format(args.model_type, args.dataset_name, str(int(time.time())))
checkpoint_dir = args.output_dir + model_name
print('#'*20)
print('current training sesseion: {}'.format(checkpoint_dir))
print('#'*20) 
if args.distributed_train:
    if args.local_rank==0:
        os.mkdir(checkpoint_dir)
elif os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
    
    

if (args.distributed_train and args.local_rank == 0) or not args.distributed_train:
    args_dict['random_seed'] = float(torch.initial_seed())
    args_dict['random_seed_np'] = float(np.random.get_state()[1][0])
    writer = SummaryWriter('{}/summaries'.format(checkpoint_dir))
    with open('{}/config.json'.format(checkpoint_dir), 'w') as fp:
        json.dump(args_dict, fp)      
    

    
### get long data real event for psd calculation ###
if args.online_whiten and args.model_type in ("NF_new"):
    print('data whitening preparation stage')
    long_real_data_parent_path = '/home/hongyu2/data/'
    if args.low_res:
        print('processubg 4k version')
        long_real_file_appendix = long_real_data_path_map_4k[args.dataset_name]
    else:
        long_real_file_appendix = long_real_data_path_map[args.dataset_name]
    long_real_data_final_path = long_real_data_parent_path + long_real_file_appendix
    h5_file = h5py.File(long_real_data_final_path, 'r')
    tmp_data_long_real = h5_file[u'strain'][u'Strain'][()]
    h5_file.close()
    psd_for_whiten = psd_cal(tmp_data_long_real, smooth=False)
    del tmp_data_long_real
    
    if args.model_type == "NF_new":
        ### whiten raw noise surrounding the event ###
        if args.low_res:
            long_raw_noise_final_path = long_real_data_parent_path + noisy_data_raw_path_map_4k[args.dataset_name]
        else:
            long_raw_noise_final_path = long_real_data_parent_path + noisy_data_raw_path_map[args.dataset_name]
        raw_noisy_files = glob.glob(long_raw_noise_final_path + '/*.hdf5')
        tmp_out = []
        for tmp_noisy_file in raw_noisy_files:
            with h5py.File(tmp_noisy_file, 'r') as f:
                raw_noisy_data_tmp = f[u'strain'][u'Strain'][()]
            if args.low_res:
                whiten_tmp_noisy_data = whiten(raw_noisy_data_tmp, psd_for_whiten, 1./4096)[30000:-30000]
            else:
                whiten_tmp_noisy_data = whiten(wave_downsampling(raw_noisy_data_tmp), psd_for_whiten, 1./8192)[30000:-30000]
            tmp_out.append(whiten_tmp_noisy_data)
        noise_for_train = np.hstack(tmp_out)
        del raw_noisy_data_tmp
    else:
        # dummy, useless
        noise_for_train = 0.

    
####### setup data loader #########


# ----------
#  Training
# ----------
print('start training')
tot_step = 0
for epoch in range(1, args.num_train_epochs+1):
    torch.initial_seed()
    if args.distributed_train:
        data_loader = DataLoader(dataset=dataset_for_train,
                              batch_size=args.batch_size, 
                              drop_last=True)#, num_workers=16)
        test_loader = DataLoader(dataset=dataset_for_valid,
                              batch_size=args.batch_size, 
                              drop_last=True)#, num_workers=16)
    else:
        data_loader = DataLoader(dataset_for_train, 
                                 batch_size=args.batch_size, drop_last=True, shuffle=False)#, num_workers=16)
        test_loader = DataLoader(dataset_for_valid, 
                                 batch_size=min(args.batch_size, 16), drop_last=False, shuffle=False)#, num_workers=16)
    i = 0
    if args.distributed_train:
        pass
    for data, label in data_loader:


        if args.online_whiten and args.model_type in ("NF_new"): 

            try:
                if args.low_res:
                    data = np.array(list(map(whiten, data.cpu().numpy(),
                                            itertools.repeat(psd_for_whiten), 
                                            itertools.repeat(1./4096))))
                else:
                    data = np.array(list(map(whiten, data.cpu().numpy(),
                                        itertools.repeat(psd_for_whiten), 
                                        itertools.repeat(1./8192))))
                data = merger_align(data)
                data = torch.Tensor(data)
                if args.short:
                    assert args.input_size == 4096
                    data = data[:, args.input_size:]
                    
            except:
                print('An exception in either merger align or whitening happens')
                continue
        

        if args.model_type in ("NF_new"):
            output_dict = train_per_step(args.model_type, data, label, model_for_train,
                                         optimizer, noise_for_train, epoch, i, args_dict=args_dict,
                                         scheduler=scheduler, device=device)
            if i % args.num_log_iter == 0:         
                print('model session: {}'.format(checkpoint_dir))
                print('epoch: {}, step: {}'.format(epoch, i))
                print('loss: {}'.format(output_dict['loss'].item()))
                if args.simCLR:
                    print('logprob_loss: {}'.format(output_dict['logprob_loss'].item()))
                    print('simCLR_loss: {}'.format(output_dict['simCLR_loss'].item()))
                print('min value: {}'.format(output_dict['min_values']))
                print('max value: {}'.format(output_dict['max_values']))
                if (args.distributed_train and args.local_rank == 0) or not args.distributed_train:
                    writer.add_scalar('training loss',
                                output_dict['loss'],
                                tot_step)
                    if args.simCLR:
                        writer.add_scalar('training logprob_loss',
                                output_dict['logprob_loss'],
                                tot_step)
                        writer.add_scalar('training simCLR_loss',
                                output_dict['simCLR_loss'],
                                tot_step)
                    writer.add_embedding(output_dict['features'][..., :],
                            global_step=tot_step)
                    try:
                        writer.add_scalar('base_variance1', output_dict['features'][0, 2], tot_step)
                        writer.add_scalar('base_variance2', output_dict['features'][0, 3], tot_step)
                        writer.add_scalar('out_variance1', output_dict['samples_check'].var(axis=0)[0][0], tot_step)
                        writer.add_scalar('out_variance2', output_dict['samples_check'].var(axis=0)[0][1], tot_step)
                    except:
                        pass
                    
                    
            if (epoch == 1 or epoch % args.eval_freq == 0) and (i % 3000 == 0 and i != 0): 
                print('#'*30)
                print('start testing')
                print('#'*30)
                with torch.no_grad():
                    model_for_train.eval()
                    tmp_loss = 0.
                    test_i = 0
                    for test_data, test_label in test_loader:
                        if args.online_whiten:
                            if args.low_res:
                                test_data = np.array(list(map(whiten, test_data.cpu().numpy(),
                                            itertools.repeat(psd_for_whiten), 
                                            itertools.repeat(1./4096))))
                            else:
                                test_data = np.array(list(map(whiten, test_data.cpu().numpy(),
                                    itertools.repeat(psd_for_whiten), 
                                    itertools.repeat(1./8192))))

                            
                                
                                
                            test_data = merger_align(test_data)
                            test_data = torch.Tensor(test_data)
                            if args.short:
                                test_data = test_data[:, args.input_size:]
                        output_dict_test = train_per_step(args.model_type, test_data, test_label, model_for_train,
                                         optimizer, noise_for_train, epoch, i, args_dict=args_dict,
                                         scheduler=scheduler, device=device, test=True)
                        tmp_loss += output_dict_test['loss']
                        test_i += 1 
                        if test_i % 100 == 0:
                            print(test_i)
                        if test_i >= 2000:
                            print(test_i)
                            break
                    tmp_loss /= test_i
                    if (args.distributed_train and args.local_rank == 0) or not args.distributed_train:
                        writer.add_scalar('testing loss',
                                tmp_loss,
                                tot_step)
                        if args.simCLR:
                            writer.add_scalar('testing logprob_loss',
                                output_dict_test['logprob_loss'],
                                tot_step)
                            writer.add_scalar('testing simCLR_loss',
                                output_dict_test['simCLR_loss'],
                                tot_step)

                        print('testing loss: {}'.format(tmp_loss))
                        if args.simCLR:
                            print('testing logprob_loss: {}'.format(output_dict_test['logprob_loss'].item()))
                            print('testing simCLR_loss: {}'.format(output_dict_test['simCLR_loss'].item()))
                        print(output_dict_test['samples_check'].shape)
                        print(output_dict_test['labels'].shape)
                        try:
                            writer.add_figure('fake data check', 
                                                  figure=plot_posteriors_different_snrs(output_dict_test['samples_check'],
                                                                                 test_label.cpu().numpy()),
                                                  global_step=tot_step)
                            
                            
                            writer.add_figure('noisy data check', 
                                                  figure=plot_waves(output_dict_test['noisy_inputs'][0].squeeze()),
                                                  global_step=tot_step)
                        except:
                            print('not here')
                            pass
                    
                    model_for_train.train()
            if (epoch == 1 or epoch % args.save_freq == 0) and (i % 3000 == 0 and i != 0):
                torch.save(model_for_train.state_dict(), checkpoint_dir + '/' + 'p_x_y_{}_{}_{:.5f}.pt'.format(epoch, i, tmp_loss))
                del output_dict_test
            

        
        i += 1
        tot_step+=1
        

print('Finished!')                                  
writer.close()                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               
                               