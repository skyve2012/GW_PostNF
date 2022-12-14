from utils import NTXentLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows import flows
SimpleRealNVP=flows.SimpleRealNVP
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from WaveNetBase import WaveNet
from WaveNetBase1 import WaveNet1
#import pyro as pyro_lib
import pyro
from Spline_Coupling_DIY import *
import pyro.distributions as dist
import pyro.distributions.transforms as T
  

class PostNF_P_X_Y_pyro_cond_CLR(nn.Module):
    def __init__(self, model_base='wavenet1', input_size=8192, pretrained=False,
                 output_feature_dim=256, filter_size=4, pool_only=True, batch_size=32, n_parameter=2,
                 model_type="NF_new", n_flow=5, wave_num_blocks=8, wave_num_layers=10, 
                 wave_num_hidden=32, wave_kernel_size=2, use_cosine_similarity=True, temperature=0.3):
        super().__init__()
        self.batch_size_bak = batch_size
        self.batch_size = batch_size
        self.model_type = model_type
        self.model_base = model_base
        self.n_parameter = n_parameter

        if model_base == 'wavenet':
            self.resenet_block = WaveNet(input_size, num_channels=1, num_classes=output_feature_dim, 
                   num_blocks=wave_num_blocks, num_layers=wave_num_layers,
                   num_hidden=wave_num_hidden, kernel_size=wave_kernel_size)#.cuda()
            
        if model_base == 'wavenet1':
            self.resenet_block = WaveNet1()
        tmp_middle_dim = 1019 if model_base == "wavenet" else 1024
        if model_base != 'wavenet1':
            self.fc1 = nn.Linear(tmp_middle_dim, output_feature_dim)
            self.fc2 = nn.Linear(output_feature_dim, output_feature_dim)
            self.fc3 = nn.Linear(output_feature_dim, output_feature_dim)
        else:

            self.fc1 = lambda x: x
            self.fc2 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)
            self.fc3 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1)

        
        num_layers = n_flow
        transforms_container = []
        bound_change = 100. if n_parameter == 2 else 3.#*10.
        self.mean_mult = torch.tensor([30., 30.]) if n_parameter == 2 else torch.tensor([0.5, 0.55, 0.07]) 
        self.std_mult = torch.tensor([5., 5.]) if n_parameter == 2 else torch.tensor([0.05, 0.03, 0.002])
        print('check std: {}'.format(self.std_mult))
        
        
        if model_type in ("NF_new"):
            for i in range(num_layers):
                transforms_container.append(T.conditional_spline_autoregressive(n_parameter, context_dim=output_feature_dim, 
                                                                    bound=bound_change, count_bins=500, hidden_dims=[256, 256]))
#             

            
        
        transforms_container.append(T.LeakyReLUTransform())
        self.transforms_container = transforms_container
        self.transforms_container_params = torch.nn.ModuleList(self.transforms_container[:-1])
        self.CLRLoss=  NTXentLoss('cuda:0', self.batch_size, temperature=temperature, 
                                  use_cosine_similarity=use_cosine_similarity)
        
            
    def forward(self, inputs_signal, inputs_label):
        
        
        
        inputs_signal = self.check_dim(inputs_signal)
        features = self.resenet_block(inputs_signal)
        if self.model_base != "wavenet1": 
            features = features.squeeze()
        
        features = self.fc1(features) 
        if features.ndim == 1:
            features = features.unsqueeze(0)
        
        CLR_encoded = F.relu(self.fc2(features))
        CLR_encoded = self.fc3(CLR_encoded)
        CLR_encoded = CLR_encoded.squeeze()
        if features.shape[0] != 1:
            features = features.squeeze()
        if features.shape[0] == features.shape[1] == 1:
            features = features.squeeze(0)
        self.features_for_check = features

        if self.training:
            CLRLoss_out = self.CLRLoss(CLR_encoded[:self.batch_size, ...], CLR_encoded[self.batch_size:, ...])
        else:
            CLRLoss_out = torch.Tensor([0.])

        
        if features.shape[1] == (self.n_parameter*2+1):
            diagonal_cov = features[..., self.n_parameter:self.n_parameter*2]**2
            p = torch.tanh(features[:, self.n_parameter*2:]).squeeze()
            cov = torch.diag_embed(diagonal_cov) + 5e-3
            cov[:, 0, 1] = p*torch.prod(torch.sqrt(diagonal_cov), dim=1)
            cov[:, 1, 0] = p*torch.prod(torch.sqrt(diagonal_cov), dim=1)
            
            base_dist = dist.MultivariateNormal(features[..., :self.n_parameter], covariance_matrix=cov)
        elif features.shape[1] == (self.n_parameter*2):
            base_dist = dist.Normal(features[..., :self.n_parameter], torch.abs(features[..., self.n_parameter:]) + 1.)
        else:
            if self.model_type in ("NF_new"):
                base_dist = dist.Normal(torch.ones(inputs_signal.shape[0], self.n_parameter)*self.mean_mult,
                                       torch.ones(inputs_signal.shape[0], self.n_parameter)*self.std_mult)
            else:
                base_dist = dist.Normal(features[..., :self.n_parameter], torch.ones_like(features))
        
        if self.model_type in ("NF_new"):
            self.flows_module = dist.ConditionalTransformedDistribution(base_dist, self.transforms_container).condition(features)
            logprob = -self.flows_module.log_prob(inputs_label).mean()
        else:
            self.flows_module = dist.TransformedDistribution(base_dist, self.transforms_container)
            logprob = -self.flows_module.log_prob(inputs_label).mean()
        return [logprob, CLRLoss_out.mean()], self.flows_module, 0.
    
    def check_dim(self, inputs_signal):
        if inputs_signal.ndim == 2:
            if self.model_base not in ("wavenet", "wavenet1"):
                inputs_signal = inputs_signal.unsqueeze(1).unsqueeze(-1)
            else:
                inputs_signal = inputs_signal.unsqueeze(1)
        return inputs_signal
    
    

    
    
