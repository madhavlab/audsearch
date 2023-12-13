<style>
  /* Define custom colors for sections */
  .introduction-section {
    background-color: #e6f7ff; /* light blue */
    padding: 10px;
  }

  .installation-section {
    background-color: #ccffcc; /* light green */
    padding: 10px;
  }

  .usage-section {
    background-color: #ffd9b3; /* light orange */
    padding: 10px;
  }

  .contributing-section {
    background-color: #ffcce6; /* light pink */
    padding: 10px;
  }

  .license-section {
    background-color: #e6ccff; /* light purple */
    padding: 10px;
  }
</style>

# My Project

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

<div class="introduction-section" id="introduction">

## Introduction
This is an awesome project that does amazing things.

</div>

<div class="installation-section" id="installation">

## Installation
To install, follow these steps...

</div>

<div class="usage-section" id="usage">

## Usage
Here's how you can use this project...

</div>

<div class="contributing-section" id="contributing">

## Contributing
If you want to contribute, please follow these guidelines...

</div>

<div class="license-section" id="license">

## License
This project is licensed under the XYZ License - see the [LICENSE.md](LICENSE.md) file for details.

</div>







<div style="background-color: #f2f2f2; padding: 10px;">
# Title

## Layout

* [**Config**](#config): Contains configuration files (.yaml)  
* **Checkpoints**: To store model weights during training
* **SRC**: Contains all modules
  * [**Models**](#models): To define DL models
  * **Index**: To create a reference database of fingerprints and perform audio retrieval
  * [**Train**](#train): To train the model
  * [**Utils**](#utils): Helping modules used by modules in Index, Train. Also used by the main.py file
* **main.py**: Integrates all the above modules. This is called for training the model

## Command Execution

* [For Training the model](#training)
* [For creating a reference database](#reference-database)
* [For Audio retrieval](#audio-retrieval)
</div>
<div style="background-color: #e6f7ff; padding: 10px;">

## Config
```yaml
main.yaml  # Used for parameters defined in main.py . This contains all the important parameters of the system.
create_refdbase.yaml # Used for parameters defined in /src/index/create_refdbase. 
search.yaml  # Used for parameters defined in /src/index/search.py. 
```
</div>
## SRC
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


# Command Execution 
## Training
1. Update `main.yaml` file: mainly the paths corresponding to the train/validation data and noise/rir files need to be
   specified.
2. Execute the command from the src/ directory: `python main.py --subdir <repository name> --config <main.yaml
   path> -d <PB directory path>`. <repository name> will be created inside PB/checkpoints/ repository
3. To resume training from a checkpoint execute command from the src/ directory: `python main.py --subdir <repository name> -c <checkpoint(*.ckpt)
   path> -d <PB directory path>`
## Creating reference Database
1. Update `create_refdbase.yaml` file: mainly the patch corresponding to reference audio files need to be specified
2. Execute the command from index/ directory: `python create_refdbase.py --config <create_refdbase.yaml path>`
## Audio Retrieval
1. Update `search.yaml` file: Specify the fingerprints database and metadata paths and model weights path.
2. Execute the command from index/ directory: `python search.py --config <search.yaml path>`. For now, it will perform
   audio retrieval for 10 noisy query files of length 5s for demo purposes.
