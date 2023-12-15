import os
import sys
import glob
import yaml
import pickle
import numpy as np
import torch
import argparse
from tqdm import tqdm
from natsort import natsorted

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
print(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from utils import Audio, Array, AudioFeature
from train import ContrastiveModel



class Indexer():
    """Builds reference database of audio fingerprints."""
    def __init__(self,audiopath, ckpt_path, fs=16000, hop=100, seglen=960, audiofeat_params={"n_fft":512, "hop_length":160,"n_mels":64}, fp_dims=128):
        self.audiopath = audiopath #(dict), {path1: ext1, path2: ext2,...}. It can store multiple parent <path> and <ext>. <ext> refers to extension of audio files.
        self.ckpt_path = ckpt_path #(str), checkpoint path of model weights
        self.fs = fs #(int, optional) sampling frequency of audio
        self.hop = hop #(int, optional), fingerprint generation interval in ms. There is a tradeoff for precise search vs storage space.
                       # Smaller the value, the more precise result and higher storage space.
        self.seglen = seglen #(int, optional), fingerprint length in ms. This is fixed, it cannot be changed.
        # audiofeat_params # (dict), list of parameters to extract time-frequency representation of audio
        self.featextractor = AudioFeature(n_fft=audiofeat_params['n_fft'], hop_length=audiofeat_params['hop_length'], n_mels=audiofeat_params['n_mels'], fs=self.fs)
        self.audioreader = Audio()

        # read reference filenames to index
        print("Loading files list")
        self.files = []
        for k, v in self.audiopath.items():
          self.files.extend(natsorted(glob.glob(os.path.join(k,'**','*.'+v), recursive=True)))

        # load audio fingerprinter
        print("Loading fingerprinter...")
        self.module = ContrastiveModel.load_from_checkpoint(ckpt_path)
        self.module.eval()
        self.module.to(torch.device("cuda"))

        # placeholder for database and metadata
        print("Initializing index...")       
        self.EMB_DB = Array(10000, fp_dims)
        self.METADATA = Array(10000, 2)
        self.FILES = Array(10, dtype=object)


    @torch.no_grad()
    def get_fp(self, filepath):
        """
        Generates sub-fingerprints of an audio track
        Parameters:
            filepath: (str), file path of audio track
        Returns:
            fp: (float32 tensors), sub fingerprints. Dims: N x emb_dim
        """
        hop = int(self.hop*0.1) # hop in no. of frames in spectrogram 
        seglen = int(self.seglen*0.1) # segment length in terms of no. of frames in spectrogram. 0.96s means 96 frames

        # read audio and get spectrograms of overlapping audio chunks of 0.96s at hop rate of 0.1s
        audio = self.audioreader.read(filepath)
        try:
            spectrum = (self.featextractor.get_log_mel_spectrogram(audio)[:, :-1])
            chunks = [spectrum[:,i:i+seglen] for i in range(0,spectrum.shape[1]-seglen-1, hop)]
            chunks = torch.stack(chunks).unsqueeze(1)
            fp = self.module.predict_step(chunks.to(torch.device("cuda")), 1)
            return fp
        except Exception as e: 
            print(e)
            return -1

    
    def create_db(self, savepath="./", N=None):
        """
        Builds reference database and hash table

        Parameters:
            savepath: (str, optional), directory path to store database
            N: (int, optional), number of files to index. If None, all files will be considered
        """
        if N is not None:
            self.files = np.random.choice(self.files, N, replace=False)

        file_idx = 0
        for filepath in tqdm(self.files):
            #fingerprints
            fp = self.get_fp(filepath)
            if isinstance(fp, int) is False:
                # build refdbase of fingerprints
                if len(fp.shape) == 1:
                    fp = fp.unsqueeze(0)
                self.EMB_DB.add(fp.cpu().numpy())
                m = np.concatenate([np.ones((len(fp),1))*file_idx, 0.1*(np.arange(0, len(fp)).reshape(-1,1))], axis=1)
                self.METADATA.add(m)
                self.FILES.add(filepath) 
                file_idx += 1
        if os.path.isdir(savepath) is False:
            os.makedirs(savepath)
        
        print("saving...")
        pickle.dump(self.EMB_DB, open(os.path.join(savepath, "EMB_DB.pkl"), "wb"), protocol=4)
        pickle.dump(self.FILES, open(os.path.join(savepath, "FILES.pkl"), "wb"), protocol=4)
        pickle.dump(self.METADATA, open(os.path.join(savepath, "METADATA.pkl"), "wb"), protocol=4)
        return
    
    def append_db(self, loadpath):
        print("appending fingerprints to existing reference database")
        self.EMB_DB = pickle.load(open(os.path.join(loadpath, "EMB_DB.pkl"), "rb"))
        self.METADATA = pickle.load(open(os.path.join(loadpath, "METADATA.pkl"), "rb"))
        self.FILES = pickle.load(open(os.path.join(loadpath, "FILES.pkl"), "rb"))
        self.build_index(loadpath)
        # print(self.EMB_DB.size, self.METADATA.size, self.FILES.size)
        return

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action="store", required=False, type=str, default="/scratch/sanup/PB_optimized/config/create_refdbase.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    indexer = Indexer(cfg['audiopath'], ckpt_path=cfg['ckpt_path'], fs=cfg['fs'], hop=cfg['hop'], seglen=cfg['seglen'], audiofeat_params=cfg['audiofeat_params'], fp_dims=cfg['fp_dims'])
    indexer.create_db(os.path.dirname(cfg['ckpt_path']))
