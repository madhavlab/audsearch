from gc import callbacks
import os
import yaml
import pickle
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import SSLDataset, ContrastiveLoss, CosineSimilarity, BiLinearSimilarity, MyCallBack
from models import CustomArch7, FpNetwork, ProjectionHead1
from train import ContrastiveModel

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

#########################################################################################################################################
#PARSE ARGS
#########################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--subdir', type=str, required=True, action="store", help="checkpoint directory")
parser.add_argument('--config', type=str, required=False, action="store", default="main", help="config file")
parser.add_argument('--device', type=int, required=False, action="store", nargs='+', default=[0], help="cuda device")
parser.add_argument('--strateg', type=str, required=False, action="store")
parser.add_argument('-c','--train_checkpoint', type=str, required=False, action="store", help="checkpoint file(.ckpt) path")
parser.add_argument('-d','--parent_dir', type=str, required=False, action="store", default="/scratch/sanup/PB_optimized/", help="parent working directory")
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()


if args.train_checkpoint is not None:
    cfg = pickle.load(open(os.path.join(args.parent_dir, "checkpoints", args.train_checkpoint.split("checkpoints")[1].split("/")[1], "params.pkl"), "rb"))
    cfg['checkpoint'] = args.train_checkpoint
    cfg['seed'] = 9
    cfg['train_clean'] = {"/scratch/sanup/data/PB/train/": "wav"}
    print("Loading saved hyperparams...")
else:
    with open("../config/"+args.config+".yaml") as f:
        print(f"Reading config file: {args.config}...")
        cfg = yaml.load(f, Loader=yaml.FullLoader)

pl.seed_everything(cfg['seed'], workers=True) 

#########################################################################################################################################
# PREPARE DATASET
#########################################################################################################################################
train_dataset = SSLDataset(audiopath=cfg['train_clean'], noisepath=cfg['train_noise'], rirpath=cfg['train_rir'], fs=cfg['fs'], seglen=cfg['seglen'],
                 power_thresh=cfg['powerthresh'], audiofeat=cfg['audiofeat'], audiofeat_params=cfg['audiofeat_params'], max_offset=cfg['max_offset'], 
                 snr_range=cfg['snr_range'],specaug=cfg['specaug'], distort_probs=cfg['train_distort_probs'])

valid_dataset = SSLDataset(audiopath=cfg['valid_clean'], noisepath=cfg['valid_noise'], rirpath=cfg['valid_rir'], fs=cfg['fs'], seglen=cfg['seglen'],
                 power_thresh=cfg['powerthresh'], audiofeat=cfg['audiofeat'], audiofeat_params=cfg['audiofeat_params'], max_offset=cfg['max_offset'], 
                 snr_range=cfg['snr_range'], distort_probs=cfg['valid_distort_probs'])

print(f"##########\nTotal files -->\nTraining files: {len(train_dataset)}\nValidation files: {len(valid_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=int(cfg['batchsize']), shuffle=True, drop_last=True, num_workers=cfg['load_workers'], pin_memory=False)
valid_dataloader = DataLoader(valid_dataset, batch_size=int(cfg['batchsize']), shuffle=False, drop_last=True, num_workers=cfg['load_workers'],pin_memory=False)

#########################################################################################################################################
# TRAINING 
#########################################################################################################################################

#Encoder
if cfg['encoder'] == "Ours":
    encoder = CustomArch7()
elif cfg['encoder'] == 'Baseline':
    network = FpNetwork(cfg['emb_dims'], h=cfg['h'], u=cfg['u'], F=256, T=cfg['T'])
else:
    raise NotImplementedError

# Encoder + Projection Head
if cfg['encoder'] != "Baseline":
    projection = ProjectionHead1(h=cfg['h'], d=cfg['d'], u=cfg['u'])
    network = nn.Sequential(encoder, projection)
else:
    network = encoder

# Loss Function
if cfg['similarity'] == "cosine similarity":
    similaritylayer = CosineSimilarity()
else:
    similaritylayer = BiLinearSimilarity(dims=cfg['emb_dims'])
loss = ContrastiveLoss(batchsize=cfg['batchsize'], temperature=cfg['temperature'], similaritylayer=similaritylayer, world_size=cfg['world_size'])

# Training Module
train_module = ContrastiveModel(network=network, loss=loss, lr=cfg['lr'], optimizer=cfg['optimizer'],
                                     weight_decay=cfg['weight_decay'], lr_scheduler=cfg['lr_scheduler'], world_size=cfg['world_size'])


# save experiments parameters
dic_path = os.path.join("../checkpoints", args.subdir)
if os.path.isdir(dic_path) is False:
    os.makedirs(dic_path)
pickle.dump(cfg, open(os.path.join(dic_path, "params.pkl"), "wb"))


# callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(filename='{epoch}-{valid_loss:.2f}-{train_loss:.2f}-{valid_top1_acc:.2f}--{valid_top5_acc:.2f}',
                                        monitor="train_loss", save_top_k=3)

#logger
version = "_".join(["temp:"+str(cfg['temperature']), "bsz:"+str(cfg['batchsize']), "lr:"+str(cfg['lr']), "seg:"+str(cfg['seglen']), "emb:"+str(cfg['emb_dims'])])
logger = TensorBoardLogger(save_dir=os.path.join(args.parent_dir, "checkpoints"), name=args.subdir, version=version, default_hp_metric=False)


custom_callback = MyCallBack() 
trainer = Trainer(accelerator="gpu", callbacks=[lr_monitor, checkpoint_callback, custom_callback],strategy=args.strateg, gpus=args.device,
                 logger=logger) #, deterministic=True


# checkpoint loading if required
if args.train_checkpoint is not None:
    print(f"Training resumes from checkpoint: {cfg['checkpoint']}")
    trainer.fit(train_module, train_dataloader, valid_dataloader,ckpt_path=cfg['checkpoint'])
else: 
    print("training from scratch begins")
    trainer.fit(train_module, train_dataloader, valid_dataloader)

#########################################################################################################################################
