import torch
import pytorch_lightning as pl
import torch.distributed as dist


class ContrastiveModel(pl.LightningModule):
    """Provides Self Supervised Learning model"""
    def __init__(self, network, loss, lr, optimizer, weight_decay=None, lr_scheduler=None, world_size=1):
        super().__init__()
        
        self.network = network
        self.loss = loss
        self.lr= lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.world_size = world_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_hyperparameters()

    def forward(self, anchor, positive):
        emb_anchor = self.network(anchor)
        emb_pos = self.network(positive)
        return emb_anchor, emb_pos

    def step(self, batch, mode='train'):
        anchor, positive = batch
        # forward pass
        self.emb_anchor, self.emb_positive = self(anchor, positive)
        if self.world_size > 1:
            embeddings_list = [torch.ones_like(self.emb_anchor) for _ in range(dist.get_world_size())]
            dist.all_gather(embeddings_list, self.emb_anchor)
            embeddings_list[dist.get_rank()] = self.emb_anchor
            self.emb_anchor = torch.cat(embeddings_list)
            
            embeddings_list = [torch.ones_like(self.emb_positive) for _ in range(dist.get_world_size())]
            dist.all_gather(embeddings_list, self.emb_positive)
            embeddings_list[dist.get_rank()] = self.emb_positive
            self.emb_positive = torch.cat(embeddings_list)

        #loss
        loss, distance = self.loss(self.emb_anchor, self.emb_positive)
        distance[range(distance.shape[0]), range(distance.shape[0])] = -1
        self.d = distance.detach().cpu()
        # mini-batch search
        with torch.no_grad():
            # the anchor should be the closest(or atleast in top 5) to its positive and vice versa
            positive_idx = torch.cat((torch.arange(distance.shape[0]/2,distance.shape[0]), torch.arange(0,distance.shape[0]/2)))
            positive_idx = positive_idx.to(self._device)
            top5 = torch.argsort(distance, descending=True)[:,:5]
            top1_acc = torch.sum(top5[:,0]==positive_idx)/(distance.shape[0])
            top5_acc = torch.sum(torch.sum(top5== positive_idx.reshape(-1,1), dim=1) > 0)/(distance.shape[0])
        
        # logging
        self.log(mode+"_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(mode+"_top1_acc", top1_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(mode+"_top5_acc", top5_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train") 

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="valid")  
    
    def predict_step(self, batch, batch_idx):
        emb_anchor = self.network(batch)
        return emb_anchor

    def configure_optimizers(self):
        if self.optimizer == "AdamW":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError
            
        if self.lr_scheduler['apply']:
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.scheduler['max_lr'], epochs=self.scheduler['epochs'],
            #                                                 steps_per_epoch=self.steps_per_epoch, div_factor=self.scheduler['div_factor'],
            #                                                 final_div_factor=self.scheduler['final_div_factor'], pct_start=self.scheduler['pct_start'])

            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
            return {'optimizer': optimizer, 'lr_scheduler':{"scheduler": scheduler, "interval":"epoch"}}
        else:
            return {'optimizer': optimizer}
