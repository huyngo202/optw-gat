import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

import math
import numpy as np

# ------------------------------------------------------------------------------
# Transformer model components
# Reusing and adapting from src/neural_net.py

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v]
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor  # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask==0, -1e32)

        attn = self.softmax(attn)
        outputs = torch.bmm(attn, v) # outputs: [b_size x len_q x d_v]
        return outputs, attn


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(_MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, self.d_v))

        self.attention = ScaledDotProductAttention(self.d_k)
        
        init.xavier_uniform_(self.w_q)
        init.xavier_uniform_(self.w_k)
        init.xavier_uniform_(self.w_v)

    def forward(self, q, k, v, attn_mask=None, is_adj=True):
        (d_k, d_v, d_model, n_heads) = (self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)

        if attn_mask is not None:
            if is_adj:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_heads, 1, 1))
            else:
                outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.unsqueeze(1).repeat(n_heads, 1, 1))
        else:
            outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=None)

        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // n_heads
        self.attention = _MultiHeadAttention(d_model, n_heads)
        self.proj = nn.Linear(n_heads * self.d_k, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask = None, is_adj = True):
        residual = q
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask, is_adj=is_adj)
        outputs = torch.cat(outputs, dim=-1)
        outputs = self.proj(outputs)
        return self.layer_norm(residual + outputs), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        outputs = self.w_2(F.relu(self.w_1(x)))
        return self.layer_norm(residual + outputs)


# ------------------------------------------------------------------------------
# New Transformer Decoder Components

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, dec_input, enc_outputs, mask):
        # dec_input: [b_size x 1 x d_model] (Query)
        # enc_outputs: [b_size x seq_len x d_model] (Key, Value)
        # mask: [b_size x seq_len]
        
        # Cross Attention
        # Note: is_adj=False because mask is [b_size x seq_len], not [b_size x seq_len x seq_len]
        # We want to mask positions in enc_outputs that are invalid/visited
        
        # The existing MultiHeadAttention logic for `is_adj=False` expects mask to be [b_size x seq_len] 
        # and expands it to [n_heads x b_size x 1 x seq_len] inside _MultiHeadAttention if we pass it correctly.
        # Let's check _MultiHeadAttention.forward:
        # if is_adj: ... attn_mask.repeat(n_heads, 1, 1) -> [n_heads*b_size x seq_len x seq_len]
        # else: ... attn_mask.unsqueeze(1).repeat(n_heads, 1, 1) -> [n_heads*b_size x 1 x seq_len]
        
        # Our mask is [b_size x seq_len].
        # We need to pass it such that it broadcasts correctly.
        
        out, attn = self.cross_attn(dec_input, enc_outputs, enc_outputs, attn_mask=mask, is_adj=False)
        out = self.pos_ffn(out)
        return out, attn

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, args):
        super(TransformerDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = args.n_layers # Reuse n_layers or add a new arg if needed
        self.n_heads = args.n_heads
        self.ff_dim = args.ff_dim
        
        # Input projection to combine graph context + current node
        # We assume input is [current_node_emb] + [graph_emb]
        self.input_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, self.ff_dim, self.n_heads) 
            for _ in range(self.n_layers)
        ])
        
        # Final Pointer Attention (Single Head to get logits)
        # We calculate logits = (Q @ K^T) / sqrt(d_k)
        # We can use a linear projection for Q and K before dot product
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.scale_factor = np.sqrt(hidden_size)
        
        # Value head for PPO
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Dummy hidden state for compatibility with RunEpisode
        self.dummy_h = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.dummy_c = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False)
        self.hidden_0 = (self.dummy_h, self.dummy_c)

    def forward(self, dec_input, hidden, enc_outputs, mask, return_value=False):
        # dec_input: [b_size x 1 x hidden_size] (Current Node Embedding)
        # enc_outputs: [b_size x seq_len x hidden_size]
        # mask: [b_size x seq_len] (1 for valid, 0 for invalid/visited)
        # return_value: If True, also return value prediction
        
        b_size = dec_input.size(0)
        
        # 1. Construct Context
        # Graph Embedding: Mean of encoder outputs (simple version)
        # [b_size x 1 x hidden_size]
        graph_emb = enc_outputs.mean(dim=1, keepdim=True)
        
        # Concatenate [graph_emb; dec_input]
        # [b_size x 1 x 2*hidden_size]
        combined_input = torch.cat([graph_emb, dec_input], dim=2)
        
        # Project to hidden_size
        # [b_size x 1 x hidden_size]
        query = self.input_proj(combined_input)
        
        # 2. Pass through Decoder Layers
        for layer in self.layers:
            query, _ = layer(query, enc_outputs, mask)
            
        # 3. Compute Pointer Logits
        # Q: [b_size x 1 x hidden_size]
        # K: [b_size x seq_len x hidden_size]
        
        Q = self.W_q(query)
        K = self.W_k(enc_outputs)
        
        # Dot product
        # [b_size x 1 x seq_len]
        logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale_factor
        
        # Squeeze to [b_size x seq_len]
        logits = logits.squeeze(1)
        
        policy = F.softmax(logits + mask.float().log(), dim=1)
        
        # Return dummy hidden state
        dummy_hidden = (self.dummy_h.expand(b_size, -1), self.dummy_c.expand(b_size, -1))
        
        if return_value:
            # Compute value prediction from query representation
            # query: [b_size x 1 x hidden_size]
            value = self.value_head(query)  # [b_size x 1 x 1]
            value = value.squeeze(-1)  # [b_size x 1]
            return policy, dummy_hidden, value
        else:
            return policy, dummy_hidden


# ------------------------------------------------------------------------------
# Encoder (Same as original)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, enc_inp, rec_enc_inp, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inp, rec_enc_inp, enc_inp, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(Encoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers

        self.L1 = nn.Linear(features_dim, hidden_size//2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size//2)

        self.layers = nn.ModuleList([EncoderLayer(hidden_size, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp


# ------------------------------------------------------------------------------
# TransformerPointerNetwork

class TransformerPointerNetwork(nn.Module):

    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        super(TransformerPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        
        # CHANGED: Use TransformerDecoder
        self.decoder = TransformerDecoder(hidden_dim, args)
        
        self.encoder = Encoder(features_dim, dfeatures_dim, hidden_dim, args)
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()

    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def _load_model_weights(self, path_string, device):
        self.load_state_dict(torch.load(path_string, map_location=device), strict=False)


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
            # dec_input is [b_size x hidden_size] (embedding)
            # We need to unsqueeze to [b_size x 1 x hidden_size] for the decoder
            if dec_input.dim() == 2:
                dec_input = dec_input.unsqueeze(1)
                
            policy, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
            return policy, dec_hidden, enc_outputs

    def sta_emb(self, sta_inp):
        return torch.tanh(self.encoder.L1(sta_inp))

    def dyn_emb(self, dyn_inp):
        return torch.tanh(self.encoder.L2(dyn_inp))
