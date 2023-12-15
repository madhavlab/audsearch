import os
import sys
import glob
import yaml
import pickle
import numpy as np
import torch
import time
import argparse
import falconn
from natsort import natsorted
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(PROJECT_ROOT)

from utils import Audio, AudioFeature, Augmentations
from train import ContrastiveModel


class LSH_Index():
    """LSH Indexer"""
    def __init__(self, dbase, tables=50, bits=18, probes=1000):
        """
        Parameters:
        ----------
            dbase: (np.ndarray), reference database of fingerprints
            tables: (int, optional), LSH parameter, no. of hash tables to create
            bits: (int, optional), no. of bits to encode fingeprints
            probes: (int, optional), LSH parameter, total no. of hash buckets to probe across hash tables
        """
        self.dbase = dbase
        self.tables = tables
        self.bits = bits
        self.probes = probes
    
    def build_index(self):
        """Build index structure using LSH"""
        self.dbase = self.dbase/np.linalg.norm(self.dbase, axis=1).reshape(-1,1)
        # center = np.mean(self.dbase, axis=0)
        # self.dbase -= center
        # self.dbase = self.dbase[:1000]

        print("Indexing database...")
        number_of_tables = self.tables
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = len(self.dbase[1])
        params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
        params_cp.distance_function = falconn.DistanceFunction.NegativeInnerProduct
        params_cp.l = number_of_tables
        params_cp.num_rotations = 1
        params_cp.seed = 5721840
        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
        falconn.compute_number_of_hash_functions(self.bits, params_cp)

        table = falconn.LSHIndex(params_cp)
        table.setup(self.dbase)
        
        query_object = table.construct_query_object()
        query_object.set_num_probes(self.probes)
        
        return query_object

def lsh_retrieval(query):
    i = query_object.find_k_nearest_neighbors(query, 5) # search top-5 matches for each subfingerprint
    # cands = len(query_object.get_unique_candidates(query))
    return i#, cands

class Search():
    """
    LSH-based indexer to perform audio retrieval
    """
    def __init__(self,checkpoint, metadata, query_object, seglen=960, fs=16000, featparams={"n_fft":512, "hop_length":160, "n_mels":64}, mode="cpu"):
        """
        Parameters:
        ---------
            checkpoint: (str), model weights path
            metadata: (list), [METADATA, FILENAMES], METADATA looks like: [[file ID_1, timestamp_1], [file ID_1, timestamp_2], ...[file ID_A, timestamp_N]]. FILENAMES looks like [FILENAME_1, ..., FILENAME_A]
            query_object: (class object), LSH class instance
            seglen: (int, optional), fingerprint length in ms. This is fixed, it cannot be changed.
            fs: (int, optional), sampling rate of an audio
            featparams: (dict, optional), required parameters to transform signal to log Mel spectrogram
            mode: (str, optional), perform search on either "cpu" or "cuda" device.
        """    
        self.dbase_meta = pickle.load(open(metadata[0], "rb")).getdata()
        self.files = pickle.load(open(metadata[1], "rb")).getdata()
        self.query_object = query_object
        self.seglen = seglen
        self.fs =fs  
        self.featextract = AudioFeature(n_fft=featparams['n_fft'], hop_length=featparams['hop_length'], n_mels=featparams['n_mels'], fs=self.fs) #STFT parameters
        self.extractor = self.featextract.get_log_mel_spectrogram 
        self.audioreader = Audio()
        self.mode= mode
        self.workers = 10
        self.parallel = False
        print("Loading fingerprinter...")
        self.model = ContrastiveModel.load_from_checkpoint(checkpoint)
        self.model.eval()
        self.model.to(torch.device(self.mode))


    def preprocess_signal(self,filepath):
        """
        Reads audio signal stored at <filepath>
        Parameters:
        ----------
            filepath: (str), file path of an audio file

        Returns: 
        -------
            audio: (float32 tensor), preprocessed audio data
        """
        audio = self.audioreader.read(filepath)
        return audio
    
    def get_segments(self, audio, hop=100):
        """Generates consecutive segments of length 1s in an <audio> track with a hop rate of <hop> samples. 
        Parameters:
        -----------
            audio: (float32 tensor), clean audio signal
            hop: (int, optional), window hop length in samples. default=100 (0.1 s)

        Returns:
        -------
            chunks: (float32 2D tensor), Batch of spectrograms corresponding to segments of fixed length
        """
        hop = int(hop*0.1) # hop in no. of frames in spectrogram 
        seglen = int(self.seglen*0.1) # segment length in terms of no. of frames in spectrogram. 0.96s means 96 frames

        spectrum = self.extractor(audio)[:, :-1]
        chunks = [spectrum[:,i:i+seglen] for i in range(0,spectrum.shape[1]-seglen-1, hop)]
        chunks = torch.stack(chunks).unsqueeze(1)
        return  chunks


    @torch.no_grad()
    def generate_embeddings(self, chunks):
        """Generates embeddings for a batch(size N) of segments generated from an audio file
        Parameters:
        ----------
            chunks: (float32 tensor), Batch(size N) of spectrograms corresponding to segments of fixed length. 

        Returns:
        --------
            fp: (np.ndarray, float32), sub fingerprints. Dims: N x emb_dim
        """
        with torch.no_grad():
            if self.mode == "cuda":
                fp = self.model.predict_step(chunks.to(torch.device("cuda")), 1)    
            else:
                fp = self.model.predict_step(chunks, 1)   
        return fp

    # find nearest neighbors for each subfingerprint using LSH
    def lookup(self, queries):
        """
        Performs audio retrieval process
        Parameters:
        ----------
            queries: (float32 tensor), batch of embeddings of size NxD (no of subfp x fp dims)

        Returns:
        -------
            id_match: (int), matched fileID index 
            timeoffset: (float), located query timestamp in identified audio file
            l_evidence: (int), number of subfingerprints supporting computed timeoffset 
            cands: (int), average number of search candidates for each sub-fingerprint 
        """
        
        emb_idx = []
        # cands = 0
        if len(queries.shape) == 1:
            queries = queries.reshape(1,-1)

        # LSH-based top-k match retrieval either using multiple processes or single process
        if self.parallel:    
            with ProcessPoolExecutor(max_workers=self.workers) as executor:
                results = executor.map(lsh_retrieval, list(queries))
                emb_idx = list(results)
        else:
            for idx in range(len(queries)):
                i = lsh_retrieval(queries[idx])
                # cands += candidates 
                emb_idx.append(i)

        # cands = cands/len(queries)
        emb_idx = np.asarray(emb_idx, dtype=int)

        # identify the audio file and keep only mathches that comes from it
        topk=5 
        ele,c = np.unique(self.dbase_meta[emb_idx[:,:topk]][:,:,0], return_counts=True)
        top_file = ele[np.argmax(c)]
        # id_match = np.sum(self.dbase_meta[emb_idx[:,:topk]][:,0,0]== top_file)

        mask = self.dbase_meta[emb_idx[:,:topk]][:,:,0] == top_file
        emb_idx[~mask] = -1
        A = []
        rank = 0
        for i in range(len(emb_idx[:,rank])):
            if emb_idx[:,rank][i] >= 0:
                A.append(np.arange(emb_idx[:,rank][i]-i,emb_idx[:,rank][i]-i+len(emb_idx[:,rank])) - emb_idx[:,0])
        A = np.array(A)
        if len(A) == 0:
            return "-1",-1,-1
        else:
            U, C = np.unique(np.asarray(A, dtype=int), axis=0, return_counts=True) # get all unique sequence candidates and their counts
            evidence = np.where((A == U[np.argmax(np.sum(U==0,axis=1))]).all(axis=1))[0]  # sequence cand with max counts
            offset =self.dbase_meta[(A[evidence]+emb_idx[:,0])[0,0]][1] # time stamp of the first index of sequence candidate
            return self.files[int(top_file)-1], offset, len(evidence)#, cands
    
    def subfingerprints_search(self,query):
        """Identifies reference audio and locate matching timestamp
        Parameters:
        ----------
            query: (str or tensor), query
        
        Returns:
        -------
            id_match: (int), matched fileID index 
            timeoffset: (float), located query timestamp in identified audio file
            l_evidence: (int), number of subfingerprints supporting computed timeoffset 
            cands: (int), average number of search candidates for each sub-fingerprint 
        """
        if isinstance(query, str):
            audio_positive = self.preprocess_signal(query)
        else: 
            audio_positive = query

        chunks= self.get_segments(audio_positive)

        # generate embeddings/subfingerprints
        fps = self.generate_embeddings(chunks)
        queries = F.normalize(fps, dim=-1)
        queries = queries.cpu().numpy()
        id_match, timeoffset, l_evidence = self.lookup(queries)
        
        return id_match, timeoffset, l_evidence

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", action="store", type=str, required=False, help="Type of distortion to add. Used for evaluating system peformance.")
    parser.add_argument("-n", action="store", type=int, required=False, help="No. of test queries. Used for evaluating system peformance.")
    parser.add_argument("--config", action="store", required=False, type=str, default="/scratch/sanup/PB_optimized/config/search.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Paths to reference database and trained model
    dbase_meta = [cfg["metadata"], cfg["files"]]
    dbase = pickle.load(open(cfg["emb_db"], "rb"))
    dbase = dbase.getdata()

    indexer = LSH_Index(dbase, tables=50, bits=18, probes=1000)
    global query_object
    query_object = indexer.build_index()

    # Path containing list of database audio filenames
    ckpt_path= cfg["checkpoint"]
    API = Search(ckpt_path, dbase_meta, query_object, mode="cuda")


    fs = 16000
    files = natsorted(glob.glob("/scratch/sanup/data/PB/train/B/**/*.wav", recursive=True))
    # files = np.random.choice(files, 1000, replace=False)
    noises = natsorted(glob.glob("/scratch/sanup/data/distortions/RIRS_NOISES/pointsource_noises/*.wav"))
    rirs = natsorted(glob.glob("/scratch/sanup/data/distortions/RIRS_NOISES/real_rirs_isotropic_noises/*.wav"))
    
    reader = Audio()
    distorter = Augmentations()
    feat_extractor = AudioFeature(n_fft=512,hop_length=160, n_mels=64, fs=fs)


    rir04 = reader.read(rirs[2])
    rir05 = reader.read(rirs[3])
    snr=0
    length=5
    for i in range(10):
        query_fname = np.random.choice(files, 1)[0]
        query_audio = reader.read(query_fname)
        offset_with_buffer = np.random.randint(len(query_audio) - (fs*(length+1)) - 1)
        noise = reader.read(np.random.choice(noises))
        
        #create clean, noise and noise+reverb added query
        clean_query = query_audio[offset_with_buffer+fs: offset_with_buffer+fs+(fs*length)]
        noise_query = distorter.add_noise(query_audio[offset_with_buffer+fs: offset_with_buffer+fs+(fs*length)], noise, snr)
        noise_reverb_04_query = distorter.add_noise_reverb(query_audio[offset_with_buffer:offset_with_buffer+(1+length)*fs], noise, snr, rir04)[fs: (1+length)*fs]
        noise_reverb_05_query = distorter.add_noise_reverb(query_audio[offset_with_buffer:offset_with_buffer+(1+length)*fs], noise, snr, rir05)[fs: (1+length)*fs]

        query_timeoffset = str((offset_with_buffer + fs)/fs)
        # filename = query_timeoffset+"_"+query_fname.split("/")[-2] + "_" +query_fname.split("/")[-1].split('.')[0]
        print(query_timeoffset, query_fname)
        s = time.time()
        songid, timeoffset, levi  = API.subfingerprints_search(clean_query)
        print(songid, timeoffset, levi, time.time()-s)



    ###############################################################################################################################################
    ## IF QUERIES ARE ALREADY CREATED AND STORED. MAINLY USED FOR SYSTEM EVALUATION PURPOSE.

    # # to search a single query (for demo)
    # queries = natsorted(glob.glob("/nlsasfs/home/nltm-st/vipular/AFP2/evaluations/data/queries/NOISE/2/25/*.wav", recursive=True))
    # for i in range(10):
    #     querypath = np.random.choice(queries)
    #     timeoff, folder, fname = os.path.basename(querypath).split("_")
    #     fname = fname.split(".")[0]
    #     print(f"GT meta: {fname}, {timeoff}")
    #     songid, timeoffset, levi, cands  = API.subfingerprints_search(querypath)
    #     print(f"Retrieved meta: {os.path.basename(songid).split('.mp3')[0]}, {timeoffset}, {levi}, {cands}")


    # to search for a batch of queries (for exps)
    # if "NOISE" in args.d:
    #     RESULTS = {}
    #     for length in tqdm([1,2,3,5]):
    #         for snr in tqdm(range(0, 25, 5), leave=False):
    #             path = os.path.join("/nlsasfs/home/nltm-st/vipular/AFP2/evaluations/data/queries/", args.d, str(length), str(snr), "*.wav")
    #             queries = natsorted(glob.glob(path, recursive=True))
                
    #             R = []
    #             for querypath in tqdm(queries[:args.n], leave=False):
    #                 timeoff, folder, fname = os.path.basename(querypath).split("_")
    #                 fname = fname.split(".")[0]
    #                 # print(f"GT meta: {fname}, {timeoff}")
    #                 try:
    #                     songid, timeoffset, levi, cands  = API.subfingerprints_search(querypath)
    #                     # print(f"Retrieved meta: {os.path.basename(songid).split('.mp3')[0]}, {timeoffset}, {levi}")
    #                     R.append([float(fname), float(timeoff), float(os.path.basename(songid).split('.mp3')[0]), timeoffset, cands, levi])
    #                 except Exception as e:
    #                     print(e)

    #             RESULTS[str(length)+"_"+str(snr)] = R
    #             pickle.dump(RESULTS, open("./results_Ours_LSH/"+args.d+".pkl", "wb"))

    # else:
    #     RESULTS = {}
    #     for length in tqdm([1,2,3,5]):
    #         for snr in tqdm([0.2, 0.4, 0.5, 0.7, 0.8], leave=False):
    #             path = os.path.join("/nlsasfs/home/nltm-st/vipular/AFP2/evaluations/data/queries/", args.d, str(length), str(snr), "*.wav")
    #             queries = natsorted(glob.glob(path, recursive=True))
                
    #             R = []
    #             for querypath in tqdm(queries[:args.n], leave=False):
    #                 timeoff, folder, fname = os.path.basename(querypath).split("_")
    #                 fname = fname.split(".")[0]
    #                 # print(f"GT meta: {fname}, {timeoff}")
    #                 try:
    #                     songid, timeoffset, levi, cands  = API.subfingerprints_search(querypath)
    #                     # print(f"Retrieved meta: {os.path.basename(songid).split('.mp3')[0]}, {timeoffset}, {levi}")
    #                     R.append([float(fname), float(timeoff), float(os.path.basename(songid).split('.mp3')[0]), timeoffset, cands, levi])
    #                 except Exception as e:
    #                     print(e)

    #             RESULTS[str(length)+"_"+str(snr)] = R

    #             pickle.dump(RESULTS, open("./results_Ours_LSH/"+args.d+".pkl", "wb"))
        