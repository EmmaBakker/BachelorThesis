from os.path import exists, join
from os import mkdir
import os
import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from custom_layers import transformer_decoder, transformer

# Function to initialize shared embedding layer for both models
def initialize_shared_embedding(num_embeddings, embedding_dim):
    """ Initializes a shared embedding tensor for both models """
    embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    torch.nn.init.xavier_uniform_(embedding.weight)
    return embedding

# Base class for the models
class BaseModel(nn.Module):
    def __init__(self, device):
        super(BaseModel, self).__init__()
        self.device = device
        self.to(self.device)
        print(f"Initialized model on device: {self.device}")

    def save_model(self, path_to_saved_models, mapper, epochs_trained, args, config_index, additional_description=""):
        try:
            # Create a description for the model to be saved
            model_desc = f"{args.language}_seed_{args.seed}_epoch_{epochs_trained}_{additional_description}_{config_index}"
            modelfolder = os.path.join(path_to_saved_models, f"model_{model_desc}/")
            if not os.path.exists(modelfolder):
                os.makedirs(modelfolder)

            # Save hyperparameters
            json.dump(self.hyperparams, open(os.path.join(modelfolder, "hyperparams.json"), 'w'))

            # Save model weights
            for label, weights in self.state_dict().items():
                fname = os.path.join(modelfolder, f"{label}_weights.json")
                if self.device.type == 'cuda':
                    weights_in_mem = weights.cpu()
                    npw = weights_in_mem.numpy()
                else:
                    npw = weights.numpy()
                np.savetxt(fname, npw)

            # Save the state dictionary
            torch.save(self.state_dict(), os.path.join(modelfolder, "model"))

            # Save the word-to-index mapping if provided
            if mapper is not None:
                mapper.save(os.path.join(modelfolder, "w2i"))

            # Save the training arguments
            json.dump(vars(args), open(os.path.join(modelfolder, "args.json"), 'w'))
            print(f"Model saved at {modelfolder}")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")

# LSTM Model class
class LSTMModel(BaseModel):
    def __init__(self, hyperparams, device, shared_embedding):
        super(LSTMModel, self).__init__(device)

        # Extracting hyperparameters
        self.hyperparams = hyperparams
        self.hidden_dim = hyperparams["hidden_dim"]             # Number of features in the hidden state
        self.embedding_dim = hyperparams["embedding_dim"]       # Size of each embedding vector
        self.n_layers = hyperparams["n_lstm_layers"]            # Number of recurrent layers (stacked LSTMs)
        self.output_size = hyperparams["output_size"]           # Size of the output feature vector

        # Defining the layers
        self.embedding = shared_embedding

        # LSTM layer
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.n_layers,
                            batch_first=True, bidirectional=False)
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, self.output_size)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights for LSTM and fully connected layers
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Send input to device (GPU/CPU)
        x = x.to(self.device)

        # Define the forward pass
        batch_size = x.size(0)

        # Initializing hidden and cell states for first input
        hidden = self.init_hidden(batch_size)

        # Embedding the words
        embeded = self.embedding(x)

        # Passing in the input, hidden, and cell state into the model and obtaining outputs
        out, hidden = self.lstm(embeded, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        out = F.log_softmax(out, dim=2)

        return out, hidden

    def init_hidden(self, batch_size):
        """
        Initializes hidden and cell states for the LSTM
        """
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device))
        return hidden

# Transformer Model class
class NWPTransformer(BaseModel, transformer):
    def __init__(self, config, device, shared_embedding):
        super(NWPTransformer, self).__init__(device)

        self.hyperparams = config

        embed = config['embed']
        tf = config['tf']

        self.is_cuda = config['cuda']

        self.embed = shared_embedding

        # Prepare positional embeddings
        self.pos_emb = self.pos_embedding(tf['max_len'], embed['embedding_dim'])
        self.TF_dec = transformer_decoder(in_size=tf['in_size'], 
                                          fc_size=tf['fc_size'], 
                                          n_layers=tf['n_layers'], 
                                          h=tf['h'])

        # Linear layer for classification
        self.linear = nn.Linear(tf['in_size'], embed['n_embeddings'])

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        super().init_weights() 

    def forward(self, input, l=False):
        input = input.to(self.device)
        
        # Create the decoder mask for padding and preventing peeking into the future
        dec_mask = self.create_dec_mask(input)
        
        # Dynamic position embedding based on current sequence length
        current_seq_len = input.size(1)
        self.pos_emb = self.pos_embedding(current_seq_len, self.embed.embedding_dim).to(self.device)
        
        # Retrieve and scale embeddings 
        d_emb = self.embed(input.long()) * np.sqrt(self.embed.embedding_dim)
        
        # Apply the (stacked) decoder transformer
        decoded = self.TF_dec(d_emb + self.pos_emb, dec_mask=dec_mask)
        
        # Apply the classification layer to the transformer output
        out = F.log_softmax(self.linear(decoded), dim=-1)
        
        # Create the targets
        targs = torch.nn.functional.pad(input[:, 1:], [0, 1]).long()

        return out, targs
