import numpy as np
from numpy import size
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


class MyCallBack(pl.Callback):
    def __init__(self):
        super().__init__()

    def plot_img(self, figure,figname,logger,trainer):
        figure.tight_layout()
        figure.canvas.draw()  #dump to memory
        img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,)) #read from memory buffer
        img = img / 255.0 #RGB
        logger.experiment.add_image(figname, img, global_step=trainer.global_step, dataformats='HWC') # add to logger
        plt.close(figure)
        return 

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        if batch_idx % 20 == 0:
            logger = trainer.logger

            figure, ax = plt.subplots(1,1)
            im = ax.matshow(pl_module.d, aspect="auto")
            figure.colorbar(im, shrink=0.8)
            self.plot_img(figure, "batch_similarity", logger, trainer)

            

            
        