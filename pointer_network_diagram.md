
# Pointer Network Architecture

Đây là sơ đồ kiến trúc Pointer Network được mô tả bằng Mermaid. Bạn có thể chèn trực tiếp vào file markdown báo cáo.

```mermaid
graph TD
    %% Encoder Section
    subgraph Encoder["Encoder (Input Sequence)"]
        direction LR
        x1(("x1")) --> e1["e1"]
        x2(("x2")) --> e2["e2"]
        x3(("x3")) --> e3["e3"]
        xn(("xn")) --> en["en"]
        
        %% RNN Connections
        e1 --> e2
        e2 --> e3
        e3 -.-> en
    end

    %% Decoder Section
    subgraph Decoder["Decoder (Step t)"]
        direction TB
        input_dec["Input (y_{t-1})"] --> d_lstm["Decoder LSTM"]
        h_dec_prev["d_{t-1}"] --> d_lstm
        d_lstm --> d_curr["d_t"]
    end

    %% Attention/Pointer Mechanism
    subgraph Attention["Attention & Pointer Mechanism"]
        direction TB
        %% Attention connections
        e1 & e2 & e3 & en -.-> |"Search"| att_score{"Attention Scores<br/>u^t_i"}
        d_curr --> att_score
        
        att_score --> softmax["Softmax"]
        softmax --> prob["Distribution<br/>a^t (Pointer)"]
        
        prob --> selection(("Output Index<br/>(Argmax / Sampling)"))
    end

    %% Input feeding back to decoder (optional visualization)
    selection -.-> |"Next Input"| input_next["y_t"]
    
    %% Styling
    style Encoder fill:#e1f5fe,stroke:#01579b
    style Decoder fill:#f3e5f5,stroke:#4a148c
    style Attention fill:#fff3e0,stroke:#e65100
    style selection fill:#b9f6ca,stroke:#1b5e20,stroke-width:2px
```

## Giải thích thành phần:
- **Encoder**: Xử lý chuỗi đầu vào $x_1, \dots, x_n$ thành các vector trạng thái ẩn $e_1, \dots, e_n$.
- **Decoder**: Tại bước $t$, maintain trạng thái ẩn $d_t$ dựa trên đầu vào trước đó.
- **Attention/Pointer**:
  - Tính điểm tương đồng (Attention Scores) giữa vector trạng thái decoder $d_t$ và toàn bộ các trạng thái encoder $e_i$.
  - **Softmax**: Chuẩn hóa điểm số thành phân phối xác suất.
  - **Output**: Thay vì sinh từ vựng mới (như Seq2Seq truyền thống), Pointer Network dùng chính phân phối xác suất này làm "con trỏ" (pointer) để chọn một trong các chỉ số đầu vào làm đầu ra tiếp theo.

## Mã LaTeX (nếu cần mô tả toán học):
$$
\begin{aligned}
u^t_i &= v^T \tanh(W_1 e_i + W_2 d_t) \quad \forall i \in (1, \dots, n) \\
a^t &= \text{softmax}(u^t) \\
y_t &= \text{argmax}(a^t)
\end{aligned}
$$
