# src/hybrid_neural_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import các thành phần có thể tái sử dụng từ mô hình gốc
from src.neural_net import Decoder, EncoderLayer, RecPointerNetwork

# Import thư viện GNN
from torch_geometric.nn import GATConv

def dense_to_sparse_edge_index(adj):
    """Chuyển đổi ma trận kề dày [N, N] sang định dạng edge_index [2, num_edges] của PyG."""
    # Đảm bảo ma trận không có gradient để tránh lỗi
    edge_index = torch.nonzero(adj.detach(), as_tuple=False).t()
    return edge_index

class HybridEncoder(nn.Module):
    """
    Bộ mã hóa lai kết hợp GAT (xử lý cục bộ) và Transformer (xử lý toàn cục).
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_size, args):
        super(HybridEncoder, self).__init__()

        n_heads = args.n_heads
        d_ff = args.ff_dim
        n_layers = args.n_layers
        n_gat_layers = args.n_gat_layers

        # Các lớp chiếu feature ban đầu, giữ nguyên
        self.L1 = nn.Linear(features_dim, hidden_size // 2)
        self.L2 = nn.Linear(dfeatures_dim, hidden_size // 2)

        # --- TÍCH HỢP GAT ---
        self.n_gat_layers = n_gat_layers
        if self.n_gat_layers > 0:
            self.gat_layers = nn.ModuleList([
                GATConv(in_channels=hidden_size, out_channels=hidden_size, heads=n_heads, concat=False, dropout=0.1)
                for _ in range(n_gat_layers)
            ])
            self.gat_layer_norm = nn.LayerNorm(hidden_size)

        # --- CÁC LỚP TRANSFORMER ---
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, d_ff, n_heads) for _ in range(n_layers)])

    def forward(self, emb_inp, rec_inp, mask, dummy_arg):
        batch_size = emb_inp.size(0)

        # --- Giai đoạn 1: Xử lý cục bộ bằng GAT ---
        if self.n_gat_layers > 0:
            gat_outputs = []
            for i in range(batch_size):
                x_i = emb_inp[i]
                adj_i = mask[i]
                edge_index_i = dense_to_sparse_edge_index(adj_i)
                
                # Truyền qua các lớp GAT
                for gat_layer in self.gat_layers:
                    # GATConv trả về embedding mới, không cần F.relu ngay lập tức
                    # F.relu thường được dùng giữa các layer
                    x_i = gat_layer(x_i, edge_index_i)

                gat_outputs.append(x_i)
            
            gat_processed_emb = torch.stack(gat_outputs, dim=0)
            emb_inp = self.gat_layer_norm(emb_inp + F.relu(gat_processed_emb))

        # --- Giai đoạn 2: Xử lý toàn cục bằng Transformer ---
        for layer in self.layers:
            emb_inp, _ = layer(emb_inp, rec_inp, mask)

        return emb_inp

class HybridPointerNetwork(RecPointerNetwork):
    """
    Lớp Pointer Network mới sử dụng HybridEncoder.
    Chúng ta kế thừa từ RecPointerNetwork để tái sử dụng các phương thức khác.
    """
    def __init__(self, features_dim, dfeatures_dim, hidden_dim, args):
        # Gọi __init__ của lớp cha (nn.Module)
        super(RecPointerNetwork, self).__init__()

        self.features_dim = features_dim
        self.dfeatures_dim = dfeatures_dim
        self.use_checkpoint = args.use_checkpoint
        self.hidden_dim = hidden_dim
        self.decoder = Decoder(hidden_dim) # Tái sử dụng Decoder từ file gốc
        
        # SỬ DỤNG BỘ MÃ HÓA LAI MỚI
        self.encoder = HybridEncoder(features_dim, dfeatures_dim, hidden_dim, args)
        
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        self._initialize_parameters()
        # Các phương thức còn lại (forward, _one_step, etc.) được kế thừa từ RecPointerNetwork