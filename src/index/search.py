import os
import sys
import glob
import yaml
import pickle
import numpy as np
import torch
import time
import argparse
# import falconn
import faiss
from natsort import natsorted
import torch.nn.functional as F

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(PROJECT_ROOT)

from utils import Audio, AudioFeature, Augmentations
from train import ContrastiveModel



# def build_index(emb_db, index_params):

#     # index parameters
#     inp_dims = emb_db.shape[1]
#     index_type = index_params['index_type']
#     centroids = index_params['centroids']
#     num_codebooks = index_params['num_codebooks']
#     codewords_bit = index_params['codewords_bit']
#     nprobe = index_params['nprobe']
#     save_path = index_params['save_index_path']     
#     gpu_device = index_params['gpu_device']                        
#     use_gpu = index_params['use_gpu']


#     # to create Voronoi cells and cell selection.   
#     quantizer = faiss.IndexFlatIP(inp_dims)

#     if index_type.lower() == "brute_force":
#         index = quantizer

#     elif index_type.lower() == "ivfflat":
#         # inverted file (pruning) + Flat (no compression)
#         index = faiss.IndexIVFFlat(quantizer, inp_dims, centroids, faiss.METRIC_INNER_PRODUCT)
        
#     elif index_type.lower() == "ivfpq":
#         # inverted file (pruning) + PQ encoding (compression)
#         index = faiss.IndexIVFPQ(quantizer, inp_dims, centroids, num_codebooks, codewords_bit, faiss.METRIC_INNER_PRODUCT)

#     elif index_type.lower() == "ivfpqr":
#         warnings.warn(f'{index_type} only supports L2 metric')
#         # inverted file (pruning) + PQR encoding (refined + compression) 
#         num_codebooks_refine, codewords_bit_refine = index_params['num_codebooks_refine'], index_params['codewords_bit_refine']
#         index = faiss.IndexIVFPQR(quantizer, inp_dims, centroids, num_codebooks, codewords_bit, num_codebooks_refine, codewords_bit_refine) # only L2 metric allowed

#     elif index_type.lower() == "hnsw":
#         # graph-based indexing (HNSW), doesnt support GPU and deletion of the item from the index
#         if use_gpu == True:
#             raise NotImplementedError(f'Faiss does not provide GPU support for: {index_type}')
#         kNN, efConstruct, efSearch = index_params['knn'], index_params['efconstruct'], index_params['efsearch']
#         index = faiss.IndexHNSWFlat(inp_dims, kNN, faiss.METRIC_INNER_PRODUCT)
#         index.hnsw.efConstruction = efConstruct
#         index.hnsw.efSearch = efSearch
#     else:
#         raise NotImplementedError(f'index type: {index_type} not available for indexing')

#     if "ivf" in index_type:
#         index.nprobe = index_params['nprobe']   
#         index.set_direct_map_type(faiss.DirectMap.Array) 

#     if use_gpu:
#         print("Copying index to GPU")
#         index = faiss.index_cpu_to_gpu(provider=faiss.StandardGpuResources(), device=gpu_device, index=index)
    
#     print("Building index...")
#     s = time.time()
#     index.train(emb_db)
#     index.add(emb_db, )
#     print(f"Index build time: {time.time()-s}")

#     print(f"Saving index at: {save_path}")
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     faiss.write_index(index, index_type+".index")
#     return index

class Search():
    """
    LSH-based indexer to perform audio retrieval
    """
    def __init__(self,checkpoint, metadata, index, seglen=960, fs=16000, featparams={"n_fft":512, "hop_length":160, "n_mels":64}, mode="cpu"):
        """
        Parameters:
        ---------
            checkpoint: (str), model weights path
            metadata: (list), [METADATA, FILENAMES], METADATA looks like: [[file ID_1, timestamp_1], [file ID_1, timestamp_2], ...[file ID_A, timestamp_N]]. FILENAMES looks like [FILENAME_1, ..., FILENAME_A]
            index: (class object), Faiss index
            seglen: (int, optional), fingerprint length in ms. This is fixed, it cannot be changed.
            fs: (int, optional), sampling rate of an audio
            featparams: (dict, optional), required parameters to transform signal to log Mel spectrogram
            mode: (str, optional), perform search on either "cpu" or "cuda" device.
        """    
        self.dbase_meta = pickle.load(open(metadata[0], "rb")).getdata()
        self.files = pickle.load(open(metadata[1], "rb")).getdata()
        self.index = index
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
            fp: (float32 tensor), sub fingerprints. Dims: N x emb_dim
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
            queries: (np.ndarray float32), batch of embeddings of size NxD (no of subfp x fp dims)

        Returns:
        -------
            id_match: (int), matched fileID index 
            timeoffset: (float), located query timestamp in identified audio file
            l_evidence: (int), number of subfingerprints supporting computed timeoffset 
        """
        
        emb_idx = []
        if len(queries.shape) == 1:
            queries = queries.reshape(1,-1)

        _, emb_idx = index.search(queries, 5)

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
            return "-1",-1,-1,-1
        else:
            U, C = np.unique(np.asarray(A, dtype=int), axis=0, return_counts=True) # get all unique sequence candidates and their counts
            evidence = np.where((A == U[np.argmax(np.sum(U==0,axis=1))]).all(axis=1))[0]  # sequence cand with max counts
            offset =self.dbase_meta[(A[evidence]+emb_idx[:,0])[0,0]][1] # time stamp of the first index of sequence candidate
            return self.files[int(top_file)], offset, len(evidence)
    
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
    parser.add_argument("--config", action="store", required=False, type=str, default=os.getcwd().replace("src/index", "config")+"/search.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Paths to reference database and trained model
    dbase_meta = [cfg["metadata"], cfg["files"]]
    dbase = pickle.load(open(cfg["emb_db"], "rb"))
    dbase = dbase.getdata()
    # dbase = np.load(cfg["emb_db"])
    
    print(f"Reading index from :{cfg['load_index_path']}")
    index = faiss.read_index(cfg['load_index_path'])
    # index.nprobe= 500
    if cfg['use_gpu']:
        print("Copying index to GPU")
        index = faiss.index_cpu_to_gpu(provider=faiss.StandardGpuResources(), device=cfg['gpu_device'], index=index)

    
    # Path containing list of database audio filenames
    ckpt_path= cfg["checkpoint"]
    API = Search(ckpt_path, dbase_meta, index, mode="cpu")

    fs = 16000
    files = natsorted(glob.glob("/home/anup/PARAMSANAGANK_backup/data/PB/train/C1/**/*.wav", recursive=True))
    # files = np.random.choice(files, 1000, replace=False)
    noises = natsorted(glob.glob("/home/anup/FMA/pointsource_noises/*.wav"))
    rirs = natsorted(glob.glob("/home/anup/FMA/real_rirs_isotropic_noises/*.wav"))
    
    reader = Audio()
    distorter = Augmentations()
    feat_extractor = AudioFeature(n_fft=512,hop_length=160, n_mels=64, fs=fs)


    # DEMO PURPOSE --> find best match for a given distorted(user's choice) query of specific length(user's choice)
    rir04 = reader.read(rirs[2])
    rir05 = reader.read(rirs[3])
    snr=0
    length=5
    for i in range(10):
        ff = np.random.choice(len(files),1)
        
        query_fname = files[int(ff)] # np.random.choice(files, 1)[0]
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
        print(songid, timeoffset, time.time()-s)



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
        