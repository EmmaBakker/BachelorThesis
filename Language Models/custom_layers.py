import torch
import torch.nn as nn
import numpy as np
import math

# Transformer Attention Mechanism
class transformer_att(nn.Module):
    def __init__(self, in_size, h):
        super(transformer_att, self).__init__()
        self.att_size = int(in_size / h)                    # Size of each attention head
        self.Q = nn.Linear(in_size, in_size, bias=False)    # Query linear transformation
        self.K = nn.Linear(in_size, in_size, bias=False)    # Key linear transformation
        self.V = nn.Linear(in_size, in_size, bias=False)    # Value linear transformation
        self.fc = nn.Linear(in_size, in_size, bias=False)   # Final linear transformation
        self.softmax = nn.Softmax(dim=-1)  
        self.h = h  
        self.dropout = nn.Dropout(0.1)                    

    def forward(self, q, k, v, mask=None):
        # Scale factor for attention scores
        scale = torch.sqrt(torch.FloatTensor([self.h])).item()
        batch_size = q.size(0)  # Batch size

        # Linear transformation and splitting into multiple heads
        Q = self.Q(q).view(batch_size, -1, self.h, self.att_size).transpose(1, 2)
        K = self.K(k).view(batch_size, -1, self.h, self.att_size).transpose(1, 2)
        V = self.V(v).view(batch_size, -1, self.h, self.att_size).transpose(1, 2)

        # Compute attention scores
        self.alpha = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply mask (if provided)
        if mask is not None:
            mask = mask.unsqueeze(1)
            self.alpha = self.alpha.masked_fill(mask == 0, -1e9)

        # Softmax over the last dimension (attention scores)
        self.alpha = self.softmax(self.alpha)

        # Compute the attention output
        att_applied = torch.matmul(self.dropout(self.alpha), V)

        # Concatenate multiple heads and apply final linear transformation
        att = att_applied.transpose(1, 2).reshape(batch_size, -1, self.att_size * self.h)
        output = self.fc(att)
        return output

# Feed Forward Network used in Transformer
class transformer_ff(nn.Module):
    def __init__(self, in_size, fc_size):
        super(transformer_ff, self).__init__()
        self.ff_1 = nn.Linear(in_size, fc_size)  # First linear layer
        self.ff_2 = nn.Linear(fc_size, in_size)  # Second linear layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, input):
        # Apply the two linear layers with ReLU in between
        output = self.ff_2(self.relu(self.ff_1(input)))
        return output

# Single Decoder Cell in Transformer Decoder
class transformer_decoder_cell(nn.Module):
    def __init__(self, in_size, fc_size, h):
        super(transformer_decoder_cell, self).__init__()
        assert in_size % h == 0 
        self.att = transformer_att(in_size, h)          # Attention mechanism
        self.ff = transformer_ff(in_size, fc_size)      # Feed forward network
        self.norm1 = nn.LayerNorm(in_size)              # Layer normalization after attention
        self.norm2 = nn.LayerNorm(in_size)              # Layer normalization after feed forward
        self.dropout = nn.Dropout(0.1)  

    def forward(self, input, dec_mask=None):
        # Apply attention mechanism
        att_output = self.att(input, input, input, dec_mask)
        att_output = self.dropout(att_output) + input
        att_output = self.norm1(att_output)

        # Apply feed forward network
        ff_output = self.ff(att_output)
        ff_output = self.dropout(ff_output) + att_output
        output = self.norm2(ff_output)
        return output

# Transformer decoder consisting of multiple decoder Cells
class transformer_decoder(nn.Module):
    def __init__(self, in_size, fc_size, n_layers, h):
        super(transformer_decoder, self).__init__()
        # Stack of transformer decoder cells
        self.tf_stack = nn.ModuleList([transformer_decoder_cell(in_size, fc_size, h) for _ in range(n_layers)])
        self.dropout = nn.Dropout(0.1) 

    def forward(self, input, dec_mask=None):
        # Pass input through each decoder cell in the stack
        for tf in self.tf_stack:
            input = tf(self.dropout(input), dec_mask)
        return input

# Complete Transformer model with positional encoding and mask creation
class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()

    # Positional embedding for input sequences
    def pos_embedding(self, sent_len, d_model):
        pos_emb = torch.zeros(sent_len, d_model)
        position = torch.arange(0, sent_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        if self.is_cuda:
            pos_emb = pos_emb.cuda()

        return pos_emb

    # Create decoder mask to prevent attention to future tokens
    def create_dec_mask(self, input):
        seq_len = input.size(1)
        # Mask for padding tokens
        mask = (input != 0).unsqueeze(1).to(dtype=torch.uint8)
        # Upper triangular matrix for masking future tokens
        triu = torch.triu(torch.ones(1, seq_len, seq_len, dtype=torch.uint8), diagonal=1) == 0
        if self.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        return torch.tensor(triu, dtype=torch.uint8, device=device) & mask

    # Initialize weights of the model
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
