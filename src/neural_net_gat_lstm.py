# src/neural_net_gat_lstm.py
# GAT-LSTM Hybrid Model: GAT Encoder + LSTM Decoder (from baseline)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# Import baseline components
from src.neural_net import Decoder, EncoderLayer

# Import GAT from PyTorch Geometric
from torch_geometric.nn import GATConv

def dense_to_sparse_edge_index(adj):
    """Convert dense adjacency matrix [N, N] to sparse edge_index [2, num_edges] format for PyG."""
    edge_index = torch.nonzero(adj.detach(), as_tuple=False).t()
    return edge_index

class GATEncoder(nn.Module):
    """
    GAT-based Encoder combining GAT layers (local) with Transformer attention layers (global).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(GATEncoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers
        n_gat_layers = args.n_gat_layers

        # Feature projection layers (same as baseline)
        self.L1 = nn.Linear(features_dim, hidden_size // 2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size // 2)

        # GAT layers for local graph processing
        self.n_gat_layers = n_gat_layers
        if self.n_gat_layers > 0:
            self.gat_layers = nn.ModuleList([
                GATConv(in_channels=hidden_size, out_channels=hidden_size, 
                       heads=n_heads, concat=False, dropout=0.1)
                for _ in range(n_gat_layers)
            ])
            self.gat_layer_norm = nn.LayerNorm(hidden_size)

        # Transformer attention layers for global context
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, d_ff, n_heads) 
            for _ in range(n_layers)
        ])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        batch_size = emb_inp.size(0)

        # Stage 1: Local graph processing with GAT
        if self.n_gat_layers > 0:
            gat_outputs = []
            for i in range(batch_size):
                x_i = emb_inp[i]
                adj_i = mask[i]
                edge_index_i = dense_to_sparse_edge_index(adj_i)
                
                # Pass through GAT layers
                for gat_layer in self.gat_layers:
                    identity = x_i
                    out = gat_layer(x_i, edge_index_i)
                    out = F.relu(out)
                    x_i = identity + out

                gat_outputs.append(x_i)
            
            gat_processed_emb = torch.stack(gat_outputs, dim=0)
            # Residual connection + layer norm
            emb_inp = self.gat_layer_norm(emb_inp + F.relu(gat_processed_emb))

        # Stage 2: Global context with Transformer attention
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


class GATLSTMPointerNetwork(nn.Module):
    """
    Hybrid Pointer Network with GAT Encoder and LSTM Decoder (baseline).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(GATLSTMPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        
        # LSTM Decoder from baseline
        self.decoder = Decoder(hidden_dim)
        
        # GAT-based Encoder
        self.encoder = GATEncoder(features_dim, dfeatures_dim, hidden_dim, args)
        
        # Dummy tensor for checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device))

    def forward(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step=False):
        policy, dec_hidden, enc_outputs = self._one_step(enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step)
        return policy, dec_hidden, enc_outputs

    def _one_step(self, enc_inputs, enc_hidden, adj_mask, dec_input, dec_hidden, mask, first_step):
        if self.use_checkpoint:
            enc_outputs = checkpoint.checkpoint(self.encoder, enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)
        else:
            enc_outputs = self.encoder(enc_inputs, enc_hidden, adj_mask, self.dummy_tensor)

        if first_step:
            return None, None, enc_outputs
        else:
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))
