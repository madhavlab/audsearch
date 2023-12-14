<div style="background-color: #f2f2f2; padding: 10px;">

  # Audio-Fingerprinting 

## Table of contents
----------------------------------------------------------------------
* [**Config**](#config): Contains configuration files (.yaml)  
* **Checkpoints**: To store model weights during training
* **SRC**: Contains all modules
  * [**Models**](#models): To define DL models
  * **Index**: To create a reference database of fingerprints and perform audio retrieval
  * [**Train**](#train): To train the model
  * [**Utils**](#utils): Helping modules used by modules in Index, Train. Also used by the main.py file
* **main.py**: Integrates all the above modules. This is called for training the model
* Command Execution
* [For Training the model](#training)
* [For creating a reference database](#creating-reference-database)
* [For Audio retrieval](#audio-retrieval)
</div>

---------------------------------------------------------------------------------------------------------------------------
<div style="background-color: #e6f7ff; padding: 10px;">

## Config
```yaml
main.yaml  # Used for parameters defined in main.py . This contains all the important parameters of the system.
create_refdbase.yaml # Used for parameters defined in /src/index/create_refdbase. 
search.yaml  # Used for parameters defined in /src/index/search.py. 
```
</div>

--------------------------------------------------------------------------------------------
## SRC                                                                       
----------------------------------------------------------------------------------------
<div style="background-color: #ccffcc; padding: 10px;">

### Models
```Python 
custom_CNN.py # DL model used as fingerprinter
feedforward.py # projection layer (NN architecture)
```
### Train
```Python 
contrastive_learning.py # Pytorch Lightning module for training the model.
```
### Utils
```Python 
audio.py #Reads and preprocess the audio files.
callbacks.py # Used during training to track progress
dataclass.py # Custom datatype to store reference database. Helps in fast appending to numpy array.
dataset.py # Custom dataset class compatible with our model training.
features.py # To transform raw audio into time-frequency representation.
losses.py # Loss metric defined used for training.
similarity.py # Similarity metric used to find similarity between embeddings during training.
main.py  #Integrates all modules.
demo.ipynb #For audio retrieval demo purposes.
```
</div>

------------------------------------------------------------------------------------------------------------------------------------

# Command Execution 
## Training
1. To initiate training, update the `main.yaml` file, focusing on specifying paths for training/validation data and noise/RIR files. Ensure that the paths are correctly set.
2. After updating the configuration file, execute the following command from the src/ directory: `python main.py --subdir <repository name> --config <main.yaml
   path> -d <PB directory path>`
3. This command will create a directory inside PB/checkpoints/ with the specified repository name.
4. If you need to resume training from a checkpoint, use the following command from the src/ directory: `python main.py --subdir <repository name> -c <checkpoint(*.ckpt)
   path> -d <PB directory path>`
5. Make sure to replace <repository name>, <main.yaml path>, and <PB directory path> with the actual values. 
---------------------------------------------------------------------------------------------------------------------
## Creating reference Database
1. To create a reference database, first, update the `create_refdbase.yaml` file, focusing on specifying the path corresponding to the reference audio files.
2. After updating the configuration file, execute the following command from the `index/` directory: `python create_refdbase.py --config <create_refdbase.yaml path>`
3. Ensure that you replace `<create_refdbase.yaml path>` with the actual path to your configuration file.
4. This command will initiate the process of creating a reference database based on the specified configuration.
--------------------------------------------------------------------------------------------------------------------
## Audio Retrieval
1. To perform audio retrieval, start by updating the `search.yaml file`. Specifically, ensure that you specify the paths for the fingerprints database, metadata, and model weights.
2. After updating the configuration file, execute the following command from the `index/` directory: `python search.py --config <search.yaml path>`
3. In this demonstration, the command will perform audio retrieval for 10 noisy query files, each with a length of 5 seconds.
4. Make sure to replace `<search.yaml path>` with the actual path to your configuration file.
5. This command will initiate the audio retrieval process based on the configured settings for a demonstration.
   
------------------------------------------------------------------------------------------------------------------------------
