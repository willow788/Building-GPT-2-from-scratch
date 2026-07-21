import numpy as np
import torch.nn as nn
import math
import torch

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        #asserting that the embedding dimn is divisible by number of heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        #query key and value matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)


        #final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    #defining the forward pass
    def forward(self, x):
        #batch size , seq length embedding dim is equal to the 
        #size of the input tensor
        batch_size, seq_length, embed_dim = x.size()

        #project input to q v k 
        Q = self.q_proj(x)  
        K = self.k_proj(x)  
        V = self.v_proj(x)  

        #reshaping for multi head attention
        Q  = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2) 

        #compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        #casual masking to prevent attending to future tokens
        mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_length, seq_length)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        #softmax
        attention_Weights = torch.softmax(scores, dim=-1)

        #weighted sum of values
        attn_outputs = torch.matmul(attention_Weights, V)

        #merging heads and projecting output
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(batch_size, seq_length, self.embed_dim)

        #final projection
        output = self.out_proj(attn_outputs)

        return output, attention_Weights
    

   
        
 #driver code to test the attention module
if __name__ == "__main__":
    x = torch.rand(2, 3, 64)
    attention = MultiHeadSelfAttention(embed_dim=64, num_heads=8)

    out = attention(x)
    print("Output shape:", out[0].shape)  # should be (2, 3, 64)
    print("Attention weights shape:", out[1].shape)  # should be (2, 8, 3, 3)
         


