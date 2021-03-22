import matplotlib.pyplot as plt
import librosa
import numpy as np
import torch


torch.manual_seed(0)
np.random.seed(0)

class STFT(torch.nn.Module):
    # Note: Compiled functions can't take variable number of arguments, have default values for arguments, nor keyword-only arguments
    def __init__(self, sr, n_fft, hop_length, win_length, n_mels, fmin):
        super(STFT, self).__init__()

        # short-time fourier transform params
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.win = torch.nn.Parameter(torch.Tensor(np.hamming(self.win_length)), requires_grad=False)
        
        # mel filterbank params (frequency, n_mels)
        n_fft = self.n_fft
        fb = librosa.filters.mel(sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, htk=True, norm=None).T
        self.fb = torch.nn.Parameter(torch.Tensor(fb), requires_grad=False)
        
        # filterbanks for data augmentation
        fb1 = librosa.filters.mel(sr*0.9, n_fft=n_fft, n_mels=n_mels, fmin=fmin, htk=True, norm=None).T
        self.fb1 = torch.nn.Parameter(torch.Tensor(fb1), requires_grad=False)
        fb2 = librosa.filters.mel(sr*1.1, n_fft=n_fft, n_mels=n_mels, fmin=fmin, htk=True, norm=None).T
        self.fb2 = torch.nn.Parameter(torch.Tensor(fb2), requires_grad=False)

    # background noise estimated as percentile of the energy in each frequency band
    # Ref: Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
    def background_noise_estimate(self, input: torch.Tensor) -> torch.Tensor:
        width = input.shape[-1]
        k = int(0.2*width)
        values, indices = torch.kthvalue(input, k=k, dim=-1, keepdim=True)
        return values

    def apply_mel_filterbank(self, input: torch.Tensor, fb: torch.Tensor):
        return torch.matmul(input.transpose(1, 2), fb).transpose(1, 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # compute a spectrogram via the Short-time Fourier transform (STFT)
        stft = torch.stft(input, 
                          n_fft=self.n_fft, 
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.win,
                          normalized=False)

        # split stft into real and imaginary parts each with shape [N, C, W]
        real_part, imag_part = torch.unbind(stft, -1)
        
        # add the squared real and imaginary parts to get a power spectrum with shape [N, C, W]
        stft = real_part.pow(2) + imag_part.pow(2)
        
        # apply a mel-filterbank (frequency, n_mels)
        if (self.training):
            index = torch.randint(low=0, high=3, size=(1,)).item()
            if (index==0):
                stft = self.apply_mel_filterbank(stft, self.fb)
            elif (index==1):
                stft = self.apply_mel_filterbank(stft, self.fb1)
            else:
                stft = self.apply_mel_filterbank(stft, self.fb2)
        else:
            stft = self.apply_mel_filterbank(stft, self.fb)
        
        # subtract an estimate of the background noise
        stft = torch.log1p(stft)
        stft = stft - self.background_noise_estimate(stft)
        stft = stft.relu()
        return stft

def xavier_init(num_input_fmaps, num_output_fmaps, receptive_field_size):
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    a = np.sqrt(6. / (fan_in + fan_out))
    return a

class Conv1D_Maxout(torch.nn.Module):
    # Note: Compiled functions can't take variable number of arguments, have default values for arguments, nor keyword-only arguments
    def __init__(self, sr: float, fmin: float):
        super(Conv1D_Maxout, self).__init__()
        
        self.dropout = torch.nn.Dropout(p=0.1)
        
        # Group Normalization
        self.input_norm = torch.nn.GroupNorm(1, 1, affine=False)
        
        n_mels = 128
        #fmin = 3000.0
        self.stft = STFT(sr, n_fft=1024, hop_length=32, win_length=512, n_mels=n_mels, fmin=fmin)

        # conv layer params
        hidden_size = 64
        kernel_size = 3
        # 8 dilated conv layers with kernel size 3 gives an effective receptive field size of 511 STFT frames (~0.74 seconds at sr=22050)
        self.conv1 = torch.nn.Conv1d(n_mels, 2*hidden_size, kernel_size=kernel_size, dilation=1)
        self.conv2 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=2)
        self.conv3 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=4)
        self.conv4 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=8)
        self.conv5 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=16)
        self.conv6 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=32)
        self.conv7 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=64)
        self.conv8 = torch.nn.Conv1d(hidden_size, 3*hidden_size, kernel_size=kernel_size, dilation=128)
        
        self.pad1 = torch.nn.ReplicationPad1d(2)
        self.pad2 = torch.nn.ReplicationPad1d(2)
        self.pad3 = torch.nn.ReplicationPad1d(4)
        self.pad4 = torch.nn.ReplicationPad1d(8)
        self.pad5 = torch.nn.ReplicationPad1d(16)
        self.pad6 = torch.nn.ReplicationPad1d(32)
        self.pad7 = torch.nn.ReplicationPad1d(64)
        self.pad8 = torch.nn.ReplicationPad1d(128)
        
        a = xavier_init(n_mels, hidden_size, kernel_size)
        torch.nn.init.uniform_(self.conv1.weight, -a, a)
        
        a = xavier_init(hidden_size, hidden_size, kernel_size)
        torch.nn.init.uniform_(self.conv2.weight, -a, a)
        torch.nn.init.uniform_(self.conv3.weight, -a, a)
        torch.nn.init.uniform_(self.conv4.weight, -a, a)
        torch.nn.init.uniform_(self.conv5.weight, -a, a)
        torch.nn.init.uniform_(self.conv6.weight, -a, a)
        torch.nn.init.uniform_(self.conv7.weight, -a, a)
        torch.nn.init.uniform_(self.conv8.weight, -a, a)
        
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.zeros_(self.conv4.bias)
        torch.nn.init.zeros_(self.conv5.bias)
        torch.nn.init.zeros_(self.conv6.bias)
        torch.nn.init.zeros_(self.conv7.bias)
        torch.nn.init.zeros_(self.conv8.bias)
        
        # classifier params
        n_outputs = 3
        bias = -4.0
        self.classifier = torch.nn.Conv1d(2*hidden_size, n_outputs, 1)
        a = xavier_init(2*hidden_size, n_outputs, 1)
        torch.nn.init.uniform_(self.classifier.weight, -a, a)
        torch.nn.init.constant_(self.classifier.bias, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # normalise the input audio samples to zero mean and std of one
        # input has shape [N, n_samples]
        input = input[:,1:] - 0.97 * input[:,:-1]
        input = self.input_norm(input[:,None,:]).squeeze(1)
        
        stft = self.stft(input)

        # apply a conv layer to extract 'features'
        h = self.conv1(self.dropout(self.pad1(stft)))
        t1, t2 = torch.chunk(h, 2, 1)
        hidden_layer1 = torch.max(t1, t2)
        h = self.conv2(self.dropout(self.pad2(hidden_layer1)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer2 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer1
        h = self.conv3(self.dropout(self.pad3(hidden_layer2)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer3 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer2
        h = self.conv4(self.dropout(self.pad4(hidden_layer3)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer4 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer3
        h = self.conv5(self.dropout(self.pad5(hidden_layer4)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer5 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer4
        h = self.conv6(self.dropout(self.pad6(hidden_layer5)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer6 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer5
        h = self.conv7(self.dropout(self.pad7(hidden_layer6)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer7 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer6
        h = self.conv8(self.dropout(self.pad8(hidden_layer7)))
        t1,t2,gate = torch.chunk(h, 3, 1)
        gate -= 1.0
        hidden_layer8 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer7
        
        # apply a linear conv layer to the features to get the output logits of shape [N, n_outputs, W]
        hidden_layer = torch.cat((hidden_layer7, hidden_layer8), dim=1)
        logits = self.classifier(hidden_layer)

        return logits
    
    @torch.jit.ignore
    def visualise(self, input: torch.Tensor, index: int=0):
        with torch.no_grad():
            ####################################################
            # same as forward method
            ####################################################
            input = input[:,1:] - 0.97 * input[:,:-1]
            input = self.input_norm(input[:,None,:]).squeeze(1)
            stft = self.stft(input)

        
            h = self.conv1(self.pad1(stft))
            t1, t2 = torch.chunk(h, 2, 1)
            hidden_layer1 = torch.max(t1, t2)
            h = self.conv2(self.pad2(hidden_layer1))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer2 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer1
            h = self.conv3(self.pad3(hidden_layer2))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer3 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer2
            h = self.conv4(self.pad4(hidden_layer3))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer4 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer3
            h = self.conv5(self.pad5(hidden_layer4))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer5 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer4
            h = self.conv6(self.pad6(hidden_layer5))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer6 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer5
            h = self.conv7(self.pad7(hidden_layer6))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer7 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer6
            h = self.conv8(self.pad8(hidden_layer7))
            t1,t2,gate = torch.chunk(h, 3, 1)
            gate -= 1.0
            hidden_layer8 = torch.max(t1, t2)*gate.sigmoid()+(1-gate.sigmoid())*hidden_layer7
            
            hidden_layer = torch.cat((hidden_layer7, hidden_layer8), dim=1)
            logits = self.classifier(hidden_layer)
            ####################################################
            
            S = stft.cpu().numpy()[index,:,:]
            plt.imshow(S, origin='lower', aspect='auto')
            plt.colorbar()
            plt.title('mel-spectrogram')
            plt.show()

            H = hidden_layer.cpu().numpy()[index,:,:]
            plt.imshow(H, origin='lower', aspect='auto')
            plt.colorbar()
            plt.title('hidden layer')
            plt.show()
            
            #plt.plot(logits.sigmoid().cpu().numpy()[index,0,:])
            #plt.ylim(0.0, 1.05)
            #plt.title('detection function')
            #plt.show()
            
            plt.plot(logits.sigmoid().cpu().numpy()[index,0,:])
            plt.ylim(0.0, 1.05)
            plt.title('detection function - class1')
            plt.show()
            
            plt.plot(logits.sigmoid().cpu().numpy()[index,1,:])
            plt.ylim(0.0, 1.05)
            plt.title('detection function - class2')
            plt.show()
            
            plt.plot(logits.sigmoid().cpu().numpy()[index,2,:])
            plt.ylim(0.0, 1.05)
            plt.title('detection function - class3')
            plt.show()

print('loaded model_maxout_3class module')
