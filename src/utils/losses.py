import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class ContrastiveLoss(nn.Module):
    """Provides loss for the contrastive learning task
    """

    def __init__(self, batchsize, temperature, similaritylayer, world_size=1):
        super().__init__()
        """
        Parameters:
        -----------
        batchsize: int, no. of samples in a batch
        temperature: float, parameter value for contrastive loss 
        similarity layer: nn.Module, layer that computes similarity
        world_size: int, no. of gpus used for ddp parallel training
        """
        self.batchsize = batchsize
        self.similaritylayer = similaritylayer
        self.temperature = temperature
        self.world_size = world_size
        
    def forward(self, emb_i, emb_j):
        """
        Parameters:
        -----------
        emb_i: float, tensor, a batch containing embeddings of anchor samples
        emb_j: float, tensor, a batch containing embeddings of positive samples

        Returns:
        --------
        average contrastive loss across batch samples
        """
        
        
        similarities1 = self.similaritylayer(emb_i, emb_j)
        similarities = torch.exp(similarities1/self.temperature)
        
        # get similarities between anchor-positive pairs
        pos_zij = torch.diag(similarities, self.batchsize*self.world_size)
        pos_zji = torch.diag(similarities, -self.batchsize*self.world_size)
        numerator = torch.cat([pos_zij, pos_zji],dim=0)

        # get similarities between anchor-negative pairs
        mask = ~torch.diag(torch.ones(2*self.batchsize*self.world_size)).bool()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = mask.to(device)

        # print(self.world_size, similarities.shape, 2*self.batchsize*self.world_size)
        denominator =  torch.sum(torch.masked_select(similarities, mask).view(2*self.batchsize*self.world_size, 2*self.batchsize*self.world_size-1), dim=1)
        loss = torch.mean(-torch.log(numerator/denominator))

        return loss, similarities1
        
