import torch
import torch.nn as nn
num_blocks = 10

class WaveNet1(nn.Module):
    def __init__(self, scale=1):
        super(WaveNet1, self).__init__()
        self.major = CustomConv()
        
        
    def forward(self, input):
        return self.major(input)
        
    
         
class ConvBlock(nn.Module):
    def __init__(self, num_channels, scale=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = num_channels, kernel_size = 3, padding = scale, dilation = scale)
        self.conv2 = nn.Conv1d(in_channels = num_channels, out_channels = num_channels, kernel_size = 3, padding = scale, dilation = scale)
        self.conv3 = nn.Conv1d(in_channels = num_channels, out_channels = num_channels, kernel_size = 1)
        #self.dp = nn.Dropout(0.3)
        #self.gelu = nn.GELU()
        self.conv4 = nn.Conv1d(in_channels = num_channels, out_channels = num_channels, kernel_size = 1)
        self.act1 =nn.Tanh()
        self.act2 = nn.Tanh()

    def forward(self, t ):
        x,v = t
        z = self.act1(self.conv1(x))
        w = self.act2(self.conv2(x))
        y = z*w
        a = self.conv3(y)
        #a = self.dp(a)
        b = self.conv4(y)
        #b = self.dp(b)
        
        return x+a, v+b

class CustomConv(nn.Module):
    def __init__(self):
        super(CustomConv, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.blocks = self.build_conv_block(num_blocks, 128)
        #self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1) # orginal
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8, stride=4) 
        #self.conv3 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1) # original
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=8, stride=4)

        self.act = nn.ReLU()#nn.ReLU()


    def build_conv_block(self, num_layers, num_channels):
        block = []
        for _ in range(num_layers):
            for i in range(11): # origin 11
                block.append(ConvBlock(num_channels, 2**i))
        return nn.Sequential(*block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        _,x = self.blocks((x,0))
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)

        return x