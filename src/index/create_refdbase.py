import os
import sys
import glob
import yaml
import pickle
import numpy as np
import torch
import argparse
import faiss
import time
from tqdm import tqdm
from natsort import natsorted

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(PROJECT_ROOT)

from utils import Audio, Array, AudioFeature
from train import ContrastiveModel


class DatabaseBuilder():
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
        self.module.to("cuda")

        # placeholder for database and metadata
        print("Initializing index...")       
        self.EMB_DB = Array(10000, fp_dims)
        self.METADATA = Array(10000, 2)
        self.FILES = Array(100, dtype=object)


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
            
            if torch.sum(torch.isnan(fp).sum(1)) > 0:
                fp = fp[torch.isnan(fp).sum(1) ==0] # added on 25.11.23 to avoid Nans in fp
            return fp
        except Exception as e: 
            print(e)
            return -1

    
    def create_db(self, save, N=None):
        """
        Builds reference database and hash table

        Parameters:
            savepath: (str, optional), directory path to store database
            N: (int, optional), number of files to index. If None, all files will be considered
        """
        if N is not None:
            self.files = np.random.choice(self.files, N, replace=False)

        file_idx = self.FILES.size
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
        if os.path.isdir(save[0]) is False:
            os.makedirs(save[0])
        
        print("saving...")
        pickle.dump(self.EMB_DB, open(os.path.join(save[0], save[1]), "wb"), protocol=4)
        pickle.dump(self.FILES, open(os.path.join(save[0], save[2]), "wb"), protocol=4)
        pickle.dump(self.METADATA, open(os.path.join(save[0], save[3]), "wb"), protocol=4)
        return self.EMB_DB.getdata()
    
    def append_db(self, save):
        self.EMB_DB = pickle.load(open(os.path.join(save[0], save[1]), "rb"))
        self.FILES = pickle.load(open(os.path.join(save[0], save[2]), "rb"))
        self.METADATA = pickle.load(open(os.path.join(save[0], save[3]), "rb"))
        idx = self.EMB_DB.size
        self.create_db(save)
        return self.EMB_DB.getdata()[idx:]

def build_index(emb_db, index_params):

    # index parameters
    inp_dims = emb_db.shape[1]
    index_type = index_params['index_type']
    centroids = index_params['centroids']
    num_codebooks = index_params['num_codebooks']
    codewords_bit = index_params['codewords_bit']
    nprobe = index_params['nprobe']
    save_path = index_params['save_index_path']     
    gpu_device = index_params['gpu_device']                        
    use_gpu = index_params['use_gpu']


    # to create Voronoi cells and cell selection.   
    quantizer = faiss.IndexFlatIP(inp_dims)

    if index_type.lower() == "brute_force":
        index = quantizer

    elif index_type.lower() == "ivfflat":
        # inverted file (pruning) + Flat (no compression)
        index = faiss.IndexIVFFlat(quantizer, inp_dims, centroids, faiss.METRIC_INNER_PRODUCT)
        
    elif index_type.lower() == "ivfpq":
        # inverted file (pruning) + PQ encoding (compression)
        index = faiss.IndexIVFPQ(quantizer, inp_dims, centroids, num_codebooks, codewords_bit, faiss.METRIC_INNER_PRODUCT)

    elif index_type.lower() == "ivfpqr":
        warnings.warn(f'{index_type} only supports L2 metric')
        # inverted file (pruning) + PQR encoding (refined + compression) 
        num_codebooks_refine, codewords_bit_refine = index_params['num_codebooks_refine'], index_params['codewords_bit_refine']
        index = faiss.IndexIVFPQR(quantizer, inp_dims, centroids, num_codebooks, codewords_bit, num_codebooks_refine, codewords_bit_refine) # only L2 metric allowed

    elif index_type.lower() == "hnsw":
        # graph-based indexing (HNSW), doesnt support GPU and deletion of the item from the index
        if use_gpu == True:
            raise NotImplementedError(f'Faiss does not provide GPU support for: {index_type}')
        kNN, efConstruct, efSearch = index_params['knn'], index_params['efconstruct'], index_params['efsearch']
        index = faiss.IndexHNSWFlat(inp_dims, kNN, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = efConstruct
        index.hnsw.efSearch = efSearch
    else:
        raise NotImplementedError(f'index type: {index_type} not available for indexing')

    if "ivf" in index_type:
        index.nprobe = index_params['nprobe']   
        index.set_direct_map_type(faiss.DirectMap.Array) 

    if use_gpu:
        print("Copying index to GPU")
        index = faiss.index_cpu_to_gpu(provider=faiss.StandardGpuResources(), device=gpu_device, index=index)
    
    print("Building index...")
    s = time.time()
    index.train(emb_db)
    index.add(emb_db, )
    print(f"Index build time: {time.time()-s}")

    print(f"Saving index at: {save_path}")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    faiss.write_index(index, index_type+".index")
    return index

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', action="store", required=False, type=str, default=os.getcwd().replace("src/index", "config")+"/create_refdbase.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    db_builder = DatabaseBuilder(cfg['audiopath'], ckpt_path=cfg['ckpt_path'], fs=cfg['fs'], hop=cfg['hop'], seglen=cfg['seglen'], audiofeat_params=cfg['audiofeat_params'], fp_dims=cfg['fp_dims'])
    
    if cfg['append_dbase']:
        print(f"appending to existing fingerprints database stored at: {cfg['save_fp_db'][0]}")
        emb_db = db_builder.append_db(cfg['save_fp_db'])
        emb_db = emb_db/np.linalg.norm(emb_db, axis=1).reshape(-1,1)
        index = faiss.read_index(cfg['load_index_path'])
        print(f"updating index stored at: {cfg['load_index_path']}")
        index.add(emb_db)
        faiss.write_index(index, cfg['load_index_path'])
    else:
        print(f"Building fingerprints database at: {cfg['save_fp_db'][0]}")
        emb_db = db_builder.create_db(cfg['save_fp_db'])
        emb_db = emb_db/np.linalg.norm(emb_db, axis=1).reshape(-1,1)
        index = build_index(emb_db, cfg)


        