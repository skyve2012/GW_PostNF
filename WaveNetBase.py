import time, copy
from collections import OrderedDict
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#import visdom

class WaveNet(nn.Module):
    def __init__(self, 
                 num_time_samples,
                 num_channels=1,
                 num_classes=256,
                 num_blocks=2,
                 num_layers=14,
                 num_hidden=128,
                 kernel_size=2):
        super(WaveNet, self).__init__()
        self.num_time_samples = num_time_samples
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.kernel_size = kernel_size
        self.receptive_field = 1 + (kernel_size - 1) * \
                               num_blocks * sum([2**k for k in range(num_layers)])
        #self.output_width = num_time_samples - self.receptive_field + 1
        self.output_width = 2048 # the length of the signal in skip connections
        print('receptive_field: {}'.format(self.receptive_field))
        print('Output width: {}'.format(self.output_width))
        
        self.set_device()

        hs = []
        batch_norms = []

        # add gated convs
        first = True
        for b in range(num_blocks):
            for i in range(num_layers):
                rate = 2**i
                if first:
                    h = GatedResidualBlock(num_channels, num_hidden, kernel_size, 
                                           self.output_width, dilation=rate)
                    first = False
                else:
                    h = GatedResidualBlock(num_hidden, num_hidden, kernel_size,
                                           self.output_width, dilation=rate)
                h.name = 'b{}-l{}'.format(b, i)

                hs.append(h)
                batch_norms.append(nn.BatchNorm1d(num_hidden))

        self.hs = nn.ModuleList(hs)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.relu_1 = nn.ReLU()
        self.conv_1_1 = nn.Conv1d(num_hidden, 64, 4, 2)
        self.relu_2 = nn.ReLU()
        self.conv_1_2 = nn.Conv1d(64, 32, 3, 1)
        self.h_class = nn.Conv1d(32, 1, 3, 1)

    def forward(self, x):
        skips = []
        for layer, batch_norm in zip(self.hs, self.batch_norms):
            x, skip = layer(x)
            x = batch_norm(x)
            skips.append(skip)
            

        x = reduce((lambda a, b : torch.add(a, b)), skips)
        x = self.relu_1(self.conv_1_1(x))
        x = self.relu_2(self.conv_1_2(x))
        x = self.h_class(x)
        
        return x.squeeze()

    def set_device(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
    '''
    def train(self, dataloader, num_epochs=25, validation=False, disp_interval=None, use_visdom=False):
        since = time.time()
        
        self.to(self.device)

        if validation:
            phase = 'Validation'
        else:
            phase = 'Training'

        if use_visdom:
            vis = visdom.Visdom()
            gen = Generator(self, dataloader.dataset)
        else:
            vis = None

        losses = []
        for epoch in range(1, num_epochs + 1):
            if not validation:
                self.scheduler.step()
                super().train()
            else:
                self.eval()
                
            # reset loss for current phase and epoch
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # track history only during training phase
                with torch.set_grad_enabled(not validation):
                    outputs = self(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    if not validation:
                        loss.backward()
                        self.optimizer.step()
                        
                running_loss += loss.item() * inputs.size(1)

            losses.append(running_loss)
            if disp_interval is not None and epoch % disp_interval == 0:
                epoch_loss = running_loss / len(dataloader)
                print('Epoch {} / {}'.format(epoch, num_epochs))
                print('Learning Rate: {}'.format(self.scheduler.get_lr()))
                print('{} Loss: {}'.format(phase, epoch_loss))
                print('-' * 10)
                print()

                if vis is not None:
                    # display network weights
                    for m in self.hs:
                        _vis_hist(vis, m.gatedconv.conv_f.weight, 
                                  m.name + ' ' + 'W-conv_f')
                        _vis_hist(vis, m.gatedconv.conv_g.weight, 
                                  m.name + ' ' + 'W-conv_g')
                        _vis_hist(vis, m.gatedconv.conv_f.bias,
                                  m.name + ' ' + 'b-conv_f')
                        _vis_hist(vis, m.gatedconv.conv_g.bias, 
                                  m.name + ' ' + 'b-conv_g')

                    # display raw outputs
                    _vis_hist(vis, outputs, 'Outputs')

                    # display loss over time
                    _vis_plot(vis, np.array(losses) / len(dataloader), 'Losses')

                    # display audio sample
                    _vis_audio(vis, gen, inputs[0], 'Sample Audio', n_samples=44100,
                               sample_rate=dataloader.dataset.sample_rate)
    '''

def _flatten(t):
    t = t.to(torch.device('cpu'))
    return t.data.numpy().reshape([-1])

def _vis_hist(vis, t, title):
    vis.histogram(_flatten(t), win=title, opts={'title': title})

def _vis_plot(vis, t, title):
    vis.line(t, X=np.array(range(len(t))), win=title, opts={'title': title})

def _vis_audio(vis, gen, t, title, n_samples=50, sample_rate=44100):
    y = gen.run(t, n_samples, disp_interval=10).reshape([-1])
    t = gen.dataset.encoder.decode(gen.tensor2numpy(t.cpu())).reshape([-1])
    audio = np.concatenate((t, y))
    vis.audio(audio, win=title, 
              opts={'title': title, 'sample_frequency': sample_rate})

class GatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedConv1d, self).__init__()
        self.dilation = dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, 
                                groups=groups, bias=bias)
        self.sig = nn.Sigmoid()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, x):
        padding_one_side = int(((self.kernel_size-1)*self.dilation))
        x = nn.functional.pad(x, (padding_one_side, 0))
        a = self.conv_f(x)
        return torch.mul(a, self.sig(self.conv_g(x)))

class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_width, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True):
        super(GatedResidualBlock, self).__init__()
        self.output_width = output_width
        self.gatedconv = GatedConv1d(in_channels, out_channels, kernel_size, 
                                     stride=stride, padding=padding, 
                                     dilation=dilation, groups=groups, bias=bias)
        self.conv_1 = nn.Conv1d(out_channels, out_channels, 1, stride=1, padding=0,
                                dilation=1, groups=1, bias=bias)

    def forward(self, x):
        
        skip = self.conv_1(self.gatedconv(x))
        residual = torch.add(skip, x)
        skip_cut = skip.shape[-1] - self.output_width
        skip = skip.narrow(-1, skip_cut, self.output_width)

        return residual, skip

class Generator(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def _shift_insert(self, x, y):
        x = x.narrow(-1, y.shape[-1], x.shape[-1] - y.shape[-1])
        dims = [1] * len(x.shape)
        dims[-1] = y.shape[-1]
        y = y.reshape(dims)
        return torch.cat([x, self.dataset._to_tensor(y)], -1)

    def tensor2numpy(self, x):
        return x.data.numpy()

    def predict(self, x):
        x = x.to(self.model.device)
        self.model.to(self.model.device)
        return self.model(x)

    def run(self, x, num_samples, disp_interval=None):
        x = self.dataset._to_tensor(self.dataset.preprocess(x))
        x = torch.unsqueeze(x, 0)

        y_len = self.dataset.y_len
        out = np.zeros((num_samples // y_len + 1) * y_len)
        n_predicted = 0
        for i in range(num_samples // y_len + 1):
            if disp_interval is not None and i % disp_interval == 0:
                print('Sample {} / {}'.format(i * y_len, num_samples))

            y_i = self.tensor2numpy(self.predict(x).cpu())
            y_i = self.dataset.label2value(y_i.argmax(axis=1))[0]
            y_decoded = self.dataset.encoder.decode(y_i)

            out[n_predicted:n_predicted + len(y_decoded)] = y_decoded
            n_predicted += len(y_decoded)

            # shift sequence and insert generated value
            x = self._shift_insert(x, y_i)

        return out[0:num_samples]


