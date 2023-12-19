import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from scipy import signal

class Audio():
    """
    Reads audio file in a specified format
    """
    def __init__(self, fs=16000, mono=True, normalize=True, preemphasis=False):
        """
        fs (int, optional): samping rate
        mono (boolean, optional): return single channel
        normalize(boolean, optional): peak normalization of signal
        preemphasis (boolean, optional): apply pre-emphasis filter
        """
        self.fs = fs
        self.mono = mono
        self.normalize = normalize
        self.preemphasis = preemphasis

    def read(self, filepath):
        """
        Reads audio file stored at <filepath>
        Parameters:
            filepath (str): audio file path
        Returns:
            waveform (tensor): audio signal, dim(N,)
        """
        assert isinstance(filepath, str), "filepath must be specified as string"
        assert os.path.exists(filepath), f"{filepath} does not exist."

        try:
            waveform, fs = torchaudio.load(filepath)
            # mono channel
            if waveform.shape[0] == 2 and self.mono is True: waveform = waveform[0]
            else: waveform = waveform.reshape(-1)
            # preemphasis
            if self.preemphasis:
                waveform = self.pre_emphasis(waveform)
            # resample
            if fs != self.fs:
                resampler = T.Resample(fs, self.fs, dtype=waveform.dtype)
                waveform = resampler(waveform)
            # normalize
            if self.normalize:
                waveform = self.peak_normalize(waveform)

            return waveform
        except Exception as e: 
            return None

    def peak_normalize(self, waveform):
        """
        Peak normalizes the <waveform>
        Parameter:
            waveform (tensor): waveform, dims: (N,)
        """
        waveform = waveform/torch.max(torch.abs(waveform))
        return waveform
    
    def pre_emphasis(self, waveform, coeff=0.97):
        filtered_sig = torch.empty_like(waveform)
        filtered_sig[1:] = waveform[1:] - coeff*waveform[:-1]
        filtered_sig[0] = waveform[0]
        return filtered_sig


class Augmentations():
    """
    Adds audio distortions such as noise, reverb, and noise+reverb
    """
    def __init__(self):
        pass

    def add_noise(self, clean, noise, snr):
        """
        Adds background <noise> to <clean> signal at desired <SNR> level
        Parameters:
            clean (tensor): clean waveform, dims: (N,)
            noise (tensor): noise waveform, dims: (M,)
            snr (int): SNR level in dB
        Returns:
            noisy signal (tensor), dims: (N,)
        """
        # make equal lengths for clean and noise signals
        if len(clean) > len(noise):
            reps = torch.ceil(torch.tensor(len(clean)/len(noise))).int()
            noise = torch.tile(noise, (reps,))[:len(clean)]
        else:
            start_idx = torch.randint(len(noise) - len(clean), (1,))
            noise = noise[start_idx:start_idx+len(clean)]        
        assert len(noise) == len(clean), f"noise signal {len(noise)} and clean signal {len(clean)} length mismatch"
        
        # add noise at desired snr
        clean_rms = self.rms(clean)
        noise_rms = self.rms(noise)
        factor = torch.pow((clean_rms/noise_rms)/torch.pow(torch.tensor(10), (snr/10)), 0.5)
        noise = factor*noise
        noise_clean = clean + noise
        assert 10*torch.log10(self.rms(clean)/self.rms(noise)) - snr < 1e-4, f"snr mismatch {10*torch.log10(self.rms(clean)/self.rms(noise)), snr}"
        return self.peak_normalize(noise_clean)


    def add_reverb(self, clean, rir):
        """
        Filters <clean> signal with <rir> to get reverberation effect
        Parameters:
            clean (tensor): clean waveform, dims: (N,)
            rir (tensor): room impulse response, dims: (M,)
        Returns:
            reverb added signal (tensor), dims: (N,)
        """
        clean = clean.numpy()
        rir = rir.numpy()
        
        # filering
        p_max = np.argmax(np.abs(rir))
        filtered_clean = signal.convolve(clean, rir, mode="full")

        # time offset
        e = np.empty_like(filtered_clean, dtype=np.float32)
        e[-p_max:] = 0.0
        e[:-p_max] = filtered_clean[p_max:] 
        filtered_clean = e.copy()
        e=None
        filtered_clean = filtered_clean[:len(clean)]
        assert(len(filtered_clean)==len(clean))
        filtered_clean = torch.from_numpy(filtered_clean)
        return self.peak_normalize(filtered_clean)

    def add_noise_reverb(self, clean, noise, snr, rir):
        """
        Adds background <noise> at desired <snr> level and reveberation using <rir> to <clean> signal 
        Parameters:
            clean (tensor): clean waveform, dims: (N,)
            noise (tensor): noise waveform, dims: (M,)
            snr (int): SNR level in dB
            rir (tensor): room impulse response, dims: (M,)
        Returns:
            noise and reverb added signal (tensor), dims: (N,)
        """
        clean_reverb = self.add_reverb(clean, rir)
        noise_reverb = self.add_reverb(noise, rir)
        noise_reverb_clean = self.add_noise(clean_reverb, noise_reverb, snr)
        return self.peak_normalize(noise_reverb_clean)

    def rms(self, waveform):
        """
        Computes RMS of a <waveform>
        Parameters:x
            waveform (tensor): waveform, dims: (N,)
        Returns
            rms (float)
        """
        return torch.mean(torch.pow(waveform, 2))
    
    def peak_normalize(self, waveform):
        """
        Peak normalizes the <waveform>
        Parameter:
            waveform (tensor): waveform, dims: (N,)
        """
        return waveform/torch.max(torch.abs(waveform))

if __name__ == "__main__":
    audioreader = Audio()
    data = audioreader.read("/scratch/sanup/data/LibriSpeech/train-clean-100/1081/128618/1081-128618-0005.flac")
    noise = audioreader.read("/scratch/sanup/data/distortions/noise_16k/Shopping.wav")
    rir = audioreader.read("/scratch/sanup/data/distortions/rir_16k/0.5.wav")

    augmenter = Augmentations()
    noise_added = augmenter.add_noise(data, noise, 10)
    reverb_added = augmenter.add_reverb(data, rir)
    noise_reverb_added = augmenter.add_noise_reverb(data, noise, 10, rir)