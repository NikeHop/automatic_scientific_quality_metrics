"""
Score Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreModel(nn.Module):
    def __init__(self,config:dict, use_context:bool)->None:
        super().__init__()

        self.use_context = use_context
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=config["nheads"],
                dim_feedforward=1024,
                dropout=config["dropout"],
                activation='relu',
                batch_first=True
            ),
            num_layers=config['num_layers']
        )
        self.dropout = nn.Dropout(config['dropout'])
        self.mlp = nn.Sequential(
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

    def forward(self,data:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
        """
        data: (batch_size, seq_len, 768)
        mask: (batch_size, seq_len)
        """
        if self.use_context:
            paper_embeddings = self.encoder(data,src_key_padding_mask=mask)

            # Select the paper embedding
            B = mask.shape[0]
            seq_len = (~mask).sum(dim=1)
            paper_embeddings = paper_embeddings[torch.arange(B),seq_len-1]
        else:
            paper_embeddings = data
            paper_embeddings = self.dropout(paper_embeddings)

        # Map to output
        output = self.mlp(paper_embeddings)

        return output