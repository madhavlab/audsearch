import torch 
import torch.nn as nn
import torch.nn.functional as F

class CosineSimilarity(nn.Module):
    """Computes the cosine similarity between embeddings
    """
    def __init__(self):
        super().__init__()

    def forward(self, emb_i, emb_j):
        """
        Parameters:
        -----------
        emb_i, emb_j: float tensors

        Returns:
        --------
        Cosine similarity between samples(embeddings) in a batch 
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        Z = torch.cat([z_i, z_j], dim=0)

        return torch.matmul(Z, torch.t(Z))


class BiLinearSimilarity(nn.Module):
    """Provides Bilinear Similarity between embeddings.
    Its a general version of cosine similarity
    """
    def __init__(self, dims):
        super().__init__()
        """
        dims: int, embedding dimesion
        """
        self.dims = dims
        W = torch.Tensor(dims,dims)
        self.W = nn.Parameter(W)
        nn.init.xavier_uniform_(self.W) 

    def forward(self, emb_i, emb_j):
        """
        Parameters:
        -----------
        emb_i, emb_j: float tensors

        Returns:
        --------
        Bilinear similarity between samples(embeddings) in a batch 
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        Z = torch.cat([z_i, z_j], dim=0)
        return torch.matmul(Z, torch.matmul(self.W, torch.t(Z)))
