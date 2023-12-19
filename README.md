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
* [**Dataset**](#dataset)
</div>

---------------------------------------------------------------------------------------------------------------------------
<div style="background-color: #e6f7ff; padding: 10px;">

## Requirements
### Minimum
* NVIDIA GPU with CUDA 10+
* 25 GB of free SSD space for mini-dataset experiments
</div>


--------------------------------------------------------------------------------------------------------------------------
<div style="background-color: #e6f7ff; padding: 10px;">

## Installation
### Install packages for the QbE system via the following commands:
#### Create a Conda environment named "PB" with Python 3.7:
   ```python
   conda create -n PB python=3.7
```
 ```python
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   pip install pytorch-lightning==1.9.5
   pip install natsort
   pip install scipy
   pip install matplotlib 
   pip install faiss
   pip install tensorboard
  ```
</div>

------------------------------------------------------------------------------------------------------------------------------------
<div style="background-color: #e6f7ff; padding: 10px;">

## Repository Structure

```python
├── config
    ├── create_refdbase.yaml # Used for parameters defined in /src/index/create_refdbase.
    ├── main.yaml # Contains all the important parameters of the system and is used for parameters defined in main.py. 
    ├── search.yaml  # Used for parameters defined in /src/index/search.py.
├── src
    ├── index
        ├──__init__.py
        ├──create_refdbase.py
        ├──search.py
    ├── model
        ├──__init__.py
        ├──custom_CNN.py # DL model used as fingerprinter
        ├──feedforward.py # projection layer (NN architecture)
    ├── train
        ├──__init__.py
        ├──contrastive_learning.py # Pytorch Lightning module for training the model.
    ├── utils
        ├──__init__.py
        ├──audio.py #Reads and preprocess the audio files.
        ├──callbacks.py # Used during training to track progress
        ├──dataclass.py # Custom datatype to store reference database. Helps in fast appending to numpy array.
        ├──dataset.py # Custom dataset class compatible with our model training.
        ├──features.py # To transform raw audio into time-frequency representation.
        ├──losses.py # Loss metric defined used for training.
        ├──similarity.py # Similarity metric used to find similarity between embeddings during training.
        ├── main.py #Integrates all modules
        ├── tutorial #For audio retrieval demo purposes.
├── docs
    ├── index.md # documentation file
├── README.md
```
</div>


--------------------------------------------------------------------------------------------
## Command Execution 
### Training
1. To initiate training, update the `main.yaml` file, focusing on specifying paths for training/validation data and noise/RIR files. Ensure that the paths are correctly set.
2. After updating the configuration file, execute the following command from the src/ directory: `python main.py --subdir <repository name> --config <main.yaml
   path> -d <PB directory path>`
3. This command will create a directory inside PB/checkpoints/ with the specified repository name.
4. If you need to resume training from a checkpoint, use the following command from the src/ directory: `python main.py --subdir <repository name> -c <checkpoint(*.ckpt)
   path> -d <PB directory path>`
5. Make sure to replace <repository name>, <main.yaml path>, and <PB directory path> with the actual values. 
---------------------------------------------------------------------------------------------------------------------
### Creating reference Database
1. To create a reference database, first, update the `create_refdbase.yaml` file, focusing on specifying the path corresponding to the reference audio files.
2. After updating the configuration file, execute the following command from the `index/` directory: `python create_refdbase.py --config <create_refdbase.yaml path>`
3. Ensure that you replace `<create_refdbase.yaml path>` with the actual path to your configuration file.
4. This command will initiate creating a reference database based on the specified configuration.
--------------------------------------------------------------------------------------------------------------------
### Audio Retrieval
1. To perform audio retrieval, start by updating the `search.yaml file`. Specifically, please make sure that you specify the paths for the fingerprints database, metadata, and model weights.
2. After updating the configuration file, execute the following command from the `index/` directory: `python search.py --config <search.yaml path>`
3. In this demonstration, the command will perform audio retrieval for 10 noisy query files, each with a length of 5 seconds.
4. Make sure to replace `<search.yaml path>` with the actual path to your configuration file.
5. This command will initiate the audio retrieval process based on the configured settings for a demonstration.
   
------------------------------------------------------------------------------------------------------------------------------
## Dataset
You can access the dataset on Kaggle [here](https://www.kaggle.com/datasets/imsparsh/fma-free-music-archive-small-medium?select=fma_medium).
It is a free Music Archive with a Large number of Genres for Music Analysis

-----------------------------------------------------------------------------------------------------------------------------------------------------------
