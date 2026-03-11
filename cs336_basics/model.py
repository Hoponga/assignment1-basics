import torch 
import torch.nn as nn
import math


class Linear(nn.Module): 

    def __init__(self, in_features, out_features, device=None, dtype=None): 
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features 
        sd = math.sqrt(2.0/(in_features + out_features))
        weight_tensor = nn.init.trunc_normal_(torch.empty(out_features, in_features, dtype = dtype), std=sd*sd, a = -3*sd, b = 3*sd)

        self.weight = nn.Parameter(weight_tensor)
        if device: 
            self.device = device 
            self.weight = self.weight.to(device)
        

    
    # x is of shape [..., in_features]
    def forward(self, x): 
        return x @ self.weight.T
    


# embed and unembed table for llm logits
class Embedding(nn.Module): 

    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None): 

        super().__init__() 

        self.vocab_size = num_embeddings  # vocab size 
        self.embedding_dim = embedding_dim  # hidden dim 
        self.device = device 

        embedding_weights = nn.init.trunc_normal_(torch.empty(self.vocab_size, self.embedding_dim, dtype = dtype), a = -3, b = 3)



        self.embedding_table = nn.Parameter(embedding_weights)


    def forward(self, token_ids : torch.Tensor):

        # for each token id index into embedding table and return corresponding vector 
        return self.embedding_table[token_ids]





