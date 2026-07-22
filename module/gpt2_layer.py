import torch
import torch.nn as nn

try:
    from module.attention import MultiHeadSelfAttention
except ModuleNotFoundError:
    from attention import MultiHeadSelfAttention

class GPT2Layer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super().__init__()

        #attention layer
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        #adding normalization layers
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        #mlp feedforward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    #forward pass for the GPT2 layer
    def forward(self, x):

        #applying attention layer
        attention_output, _ = self.attention(x)
        x = x + attention_output  #residual connection
        x = self.layer_norm1(x)  #layer normalization

        #mlp block
        mlp_output = self.mlp(x)

        #residual connection
        x = x + mlp_output
        x = self.layer_norm2(x)  #layer normalization

        return x


#driver code to test the GPT2Layer
if __name__ == "__main__":
    x = torch.rand(2, 3, 64)  #batch size, sequence length, embedding dimension
    gpt2_layer = GPT2Layer(embed_dim=64, num_heads=8, mlp_hidden_dim=128)
    output = gpt2_layer(x)
    print(output.shape)  #should print torch.Size([2, 3, 64])
        
