# Title
## Layout
* **Config**: Contains configuration files (.yaml)  
* **Checkpoints**: To store model weights during training
* **SRC**: Contains all modules
  * **Modes**: To define DL models
  * **Index**: To create a reference database of fingerprints and perform audio retrieval
  * **Train**: To train the model
  * **Utils**: Helping modules used by modules in Index, Train. Also used by the main.py file
* **main.py**: Integrates all the above modules. This is called for training the model
