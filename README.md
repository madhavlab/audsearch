<div style="background-color: #f2f2f2; padding: 10px;">

  # Audio-Search

### Table of contents 
----------------------------------------------------------------------
* [**System Requirments**](#requirements)
* [**Installation**](#installation)
* [**Repository Structure**](#repository-structure)
* [**Command Execution**](#command-execution)
    * [For Training the model](#training)
    * [For creating a reference database](#creating-reference-database)
    * [For Audio retrieval](#audio-retrieval)
* [**Dataset and model weights**](#dataset-and-model-weights)
* [**Tutorial Slides and Recording**](#tutorial-slides-and-recording) 
* [**References**](#references)
</div>

--------------------------------------------------------------------------------------------------------------------------
<div style="background-color: #e6f7ff; padding: 10px;">

## Installation

#### Create a Conda environment with Python 3.7:
   ```python
   conda create -n <env_name> python=3.7
```
#### Install packages for the QbE system via the following commands:
 ```python
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   pip install pytorch-lightning==1.9.5
   pip install natsort
   pip install scipy
   pip install matplotlib 
   pip install faiss-cpu --no-cache
   pip install tensorboard
  ```
</div>

------------------------------------------------------------------------------------------------------------------------------------
<div style="background-color: #e6f7ff; padding: 10px;">

## Repository Structure

```python
├── config
│   ├── create_refdbase.yaml # Used for parameters defined in /src/index/create_refdbase.
│   ├── main.yaml # Used for parameters defined in main.py. It contains parameters for model training. 
│   └── search.yaml  # Used for parameters defined in /src/index/search.py.
├── src
│   ├── index
│   │   ├──create_refdbase.py # creates fingerprints database and builds an index. 
│   │   └──search.py # performs search and also used for demonstration purposes.
│   ├── model
│   │   ├──custom_CNN.py # model
│   │   └──feedforward.py # projection layer
│   ├── train
│   │   └──contrastive_learning.py # Pytorch Lightning module for training the model.
│   ├── utils
│   │   ├──audio.py # Reads and preprocess the audio files.
│   │   ├──callbacks.py # Used during training to track progress
│   │   ├──dataclass.py # Custom datatype to store reference database. Helps in fast appending to numpy array.
│   │   ├──dataset.py # Custom dataset class compatible with our model training.
│   │   ├──features.py # To transform raw audio into time-frequency representation.
│   │   ├──losses.py # Loss metric defined used for training.
│   │   └──similarity.py # Similarity metric used to find similarity between embeddings during training.
│   ├── main.py # Main module to start training.
│   ├── tutorial.ipynb
```
</div>


--------------------------------------------------------------------------------------------
## Command Execution 
### Training
1. To initiate training, update the `main.yaml` file, focusing on specifying paths for training/validation data and noise/RIR files. Ensure that the paths are correctly set.
2. After updating the configuration file, execute the following command from the src/ directory: `python main.py --subdir <repository name>`. This command will create a directory inside your parent working directory <pwd> at '<pwd>/checkpoints/<repository name>`.
3. If you need to resume training from a checkpoint, use the following command from the src/ directory: `python main.py --subdir <repository name> -c <checkpoint path (*.ckpt)>`
4. Make sure to replace <repository name>, <main.yaml path>, and <PB directory path> with the actual values. 
---------------------------------------------------------------------------------------------------------------------
### Creating reference Database
1. To create a reference database, first, update the `create_refdbase.yaml` file, focusing on specifying the path corresponding to the reference audio files.
2. After updating the configuration file, execute the following command from the `index/` directory: `python create_refdbase.py`
3. To update the existing index with new audio files, update the following keys in `create_refdbase.yaml`:
- set `append_db` to 'True' 
- set `load_index_path` to a path of existing index file
--------------------------------------------------------------------------------------------------------------------
### Audio Retrieval
1. To perform audio retrieval, start by updating the `search.yaml file`. Specifically, please make sure that you specify the paths for the fingerprints database, metadata, and model weights.
2. After updating the configuration file, execute the following command from the `index/` directory: `python search.py` to run the search demonstration.
3. The demonstration shows audio retrieval for 10 noisy query files, each with a length of 5 seconds.
4. Make sure to replace `<search.yaml path>` with the actual path to your configuration file.
5. This command will initiate the audio retrieval process based on the configured settings for a demonstration.

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Dataset and model weights
You can access the Free Music Archive (FMA) dataset on Kaggle [here](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?select=fma_medium).
and model weights from [here](https://drive.google.com/file/d/17pUMR2n8tQlXH6jFBZkownzR-3t6JSNE/view?usp=drive_link). 

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Tutorial slides and recording
[Tutorial slides](https://docs.google.com/presentation/d/1tR92Cq3baK_xeE2Uj1s1IPlYlRcN5xMv_spFGcgH8Ys/edit#slide=id.p)

[Tutorial video](https://iitk-my.sharepoint.com/:f:/g/personal/params21_iitk_ac_in/EqL7zmsX4rpCprdJV9IWtAABejBCiKhwryT9hqQn-zAp6g?e=r7hqvR)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## References

1. Anup Singh, Kris Demuynck, and Vipul Arora. Simultaneously Learning Robust Audio Embeddings and Balanced Hash Codes for Query-by-Example. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), ICASSP 2023*, pp. 1-5, 2023 [![Paper](https://img.shields.io/badge/Paper-IEEE-informational?style=flat&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/10096103)
2.  Anup Singh, Kris Demuynck, and Vipul Arora. Attention-based Audio Embeddings for Query-by-Example. In *Proceedings of the 23rd International Society for Music Information Retrieval Conference, ISMIR 2022*,  pp.52-58, 2022.
  [![Paper](https://img.shields.io/badge/Paper-ISMIR%202022-informational?style=flat&logoColor=white)](https://archives.ismir.net/ismir2022/paper/000005.pdf)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

