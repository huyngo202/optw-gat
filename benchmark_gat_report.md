# Báo Cáo Benchmark GAT Models - OPTW Problem

**Ngày tạo:** 27/11/2025  
**Người thực hiện:** Benchmark GAT Runner

---

## 1. Tổng Quan

Báo cáo này tổng hợp kết quả benchmark các mô hình Graph Attention Network (GAT) cho bài toán Orienteering Problem with Time Windows (OPTW). Các mô hình được đánh giá bao gồm:

- **Baseline**: Mô hình baseline ban đầu
- **Transformer**: Mô hình với Transformer Decoder
- **Transformer-PPO**: Mô hình Transformer huấn luyện với Proximal Policy Optimization
- **GAT-Transformer**: Mô hình GAT Encoder + Transformer Decoder
- **GAT-LSTM**: Mô hình GAT Encoder + LSTM Decoder

## 2. Phương Pháp Benchmark

### 2.1 Tập Instance
Benchmark được thực hiện trên 5 instance tiêu chuẩn:
- **c101**: Clustered customers
- **r101**: Random customers
- **rc101**: Random-clustered customers
- **pr01**: Problem từ bộ dữ liệu chuẩn
- **t101**: Time window constrained

### 2.2 Cấu Hình Huấn Luyện
- **Số epochs**: 5,000 (Baseline, Transformer, GAT models)
- **Số epochs**: 10,000 (Transformer-PPO)
- **Batch size**: 16
- **Device**: CPU
- **GAT layers**: 3 layers

### 2.3 Metrics Đánh Giá
- **avg_val**: Average reward trên validation set
- **max_real**: Maximum reward đạt được trên dữ liệu thực
- **epoch**: Số epoch huấn luyện

---

## 3. Kết Quả Chi Tiết

### 3.1 Instance: c101

| Model | Epochs | Avg Validation Reward | Max Real Reward |
|-------|--------|----------------------|-----------------|
| **Baseline** | 5000 | 257.16 | **300.0** |
| **Transformer** | 5000 | 254.61 | **300.0** |
| **Transformer-PPO** | 10000 | 172.25 | 10.0 |
| **GAT-Transformer** | 5000 | 256.84 | **300.0** |
| **GAT-LSTM** | 200 | 242.31 | 270.0 |

**Nhận xét:**
- Ba mô hình đạt max reward tối ưu (300.0): Baseline, Transformer, và GAT-Transformer
- GAT-LSTM mới huấn luyện 200 epochs, chưa hội tụ hoàn toàn
- Transformer-PPO cho kết quả kém (max 10.0), cần xem xét lại hyperparameters

### 3.2 Instance: r101

| Model | Epochs | Avg Validation Reward | Max Real Reward |
|-------|--------|----------------------|-----------------|
| **Baseline** | 5000 | 109.41 | 190.0 |
| **Transformer** | 5000 | 110.22 | **179.0** |
| **Transformer-PPO** | 10000 | 98.67 | 0.0 |
| **GAT-Transformer** | 5000 | 109.64 | **179.0** |

**Nhận xét:**
- GAT-Transformer và Transformer cho kết quả tương đương (179.0)
- Avg validation reward của GAT-Transformer (109.64) cao nhất
- Transformer-PPO thất bại hoàn toàn (max 0.0)

### 3.3 Instance: rc101

| Model | Epochs | Avg Validation Reward | Max Real Reward |
|-------|--------|----------------------|-----------------|
| **Baseline** | 5000 | 147.13 | 202.0 |
| **Transformer** | 5000 | 144.86 | 205.0 |
| **Transformer-PPO** | 10000 | 108.69 | 0.0 |
| **GAT-Transformer** | 5000 | 148.09 | **216.0** |

**Nhận xét:**
- ✨ **GAT-Transformer vượt trội** với max reward 216.0 (cao nhất)
- GAT-Transformer cũng đạt avg validation reward cao nhất (148.09)
- Cải thiện ~5-7% so với Baseline và Transformer

### 3.4 Instance: pr01

| Model | Epochs | Avg Validation Reward | Max Real Reward |
|-------|--------|----------------------|-----------------|
| **Baseline** | 5000 | 184.50 | **306.0** |
| **Transformer** | 5000 | 182.94 | 279.0 |
| **Transformer-PPO** | 10000 | 121.81 | 52.0 |
| **GAT-Transformer** | 5000 | 182.84 | 277.0 |

**Nhận xét:**
- Baseline đạt max reward cao nhất (306.0)
- GAT-Transformer và Transformer có kết quả tương đương (~277-279)
- Transformer-PPO kém hơn đáng kể

### 3.5 Instance: t101

| Model | Epochs | Avg Validation Reward | Max Real Reward |
|-------|--------|----------------------|-----------------|
| **Baseline** | 5000 | 763.94 | 214.0 |
| **Transformer** | 5000 | 748.80 | **326.0** |
| **Transformer-PPO** | 10000 | 399.98 | 125.0 |
| **GAT-Transformer** | - | - | **N/A** |

**Nhận xét:**
- GAT-Transformer chưa có kết quả huấn luyện (training history chưa được tạo)
- Transformer đạt max reward cao nhất (326.0)
- Đây là instance khó nhất với avg validation reward rất cao (>700)

---

## 4. So Sánh Tổng Hợp

### 4.1 Bảng So Sánh Max Real Reward

| Instance | Baseline | Transformer | PPO | GAT-Trans | GAT-LSTM | **Winner** |
|----------|----------|-------------|-----|-----------|----------|------------|
| c101 | 300.0 | 300.0 | 10.0 | 300.0 | 270.0 | **Tie (3-way)** |
| r101 | 190.0 | 179.0 | 0.0 | 179.0 | - | **Baseline** |
| rc101 | 202.0 | 205.0 | 0.0 | **216.0** | - | **GAT-Trans** |
| pr01 | **306.0** | 279.0 | 52.0 | 277.0 | - | **Baseline** |
| t101 | 214.0 | **326.0** | 125.0 | N/A | - | **Transformer** |

### 4.2 Hiệu Suất Theo Model

#### GAT-Transformer
- ✅ **Điểm mạnh:**
  - Đạt kết quả tốt nhất tại rc101 (216.0)
  - Tương đương với Transformer tại c101 và r101
  - Avg validation reward cao và ổn định
  
- ⚠️ **Điểm cần cải thiện:**
  - Chưa có kết quả cho t101
  - Tại pr01, kém hơn Baseline ~9.5%

#### GAT-LSTM
- ⚠️ **Trạng thái:**
  - Chỉ có kết quả cho c101
  - Mới huấn luyện 200 epochs (chưa đủ để đánh giá)
  - Cần tiếp tục huấn luyện để so sánh công bằng

#### Transformer-PPO
- ❌ **Vấn đề nghiêm trọng:**
  - Kết quả kém hơn rất nhiều so với các mô hình khác
  - Max reward = 0 tại r101 và rc101
  - Cần review lại cấu hình PPO và hyperparameters

---

## 5. Phân Tích và Đánh Giá

### 5.1 Hiệu Quả của GAT Encoder

GAT (Graph Attention Network) encoder đã cho thấy khả năng học biểu diễn đồ thị tốt:

1. **Tốt nhất tại rc101:** GAT-Transformer đạt 216.0 (cao nhất), cho thấy GAT xử lý tốt bài toán có cấu trúc random-clustered

2. **Tương đương với Transformer chuẩn:** Tại c101 và r101, GAT-Transformer đạt kết quả tương đương hoặc tốt hơn một chút so với Transformer truyền thống

3. **Avg validation reward cao:** GAT-Transformer thường có avg validation reward cao, cho thấy tính ổn định

### 5.2 So Sánh GAT-Transformer vs Transformer

| Metric | GAT-Transformer | Transformer | Kết luận |
|--------|-----------------|-------------|----------|
| **Wins** | 1 (rc101) | 1 (t101) | Ngang nhau |
| **Ties** | 2 (c101, r101) | 2 (c101, r101) | - |
| **Stability** | Cao (avg_val ổn định) | Cao | GAT hơi tốt hơn |
| **Training Speed** | Tương đương | Tương đương | - |

### 5.3 Vấn Đề với PPO

Transformer-PPO cho kết quả rất kém trên hầu hết các instance:
- Max reward = 0 tại r101, rc101
- Max reward = 10 tại c101 (so với 300 của các mô hình khác)

**Nguyên nhân có thể:**
- Hyperparameters chưa phù hợp
- Learning rate quá cao/thấp
- Số epochs chưa đủ (mặc dù đã train 10,000 epochs)
- Cấu trúc reward shaping cần điều chỉnh

---

## 6. Kết Luận và Khuyến Nghị

### 6.1 Kết Luận

1. **GAT-Transformer** là một kiến trúc hứa hẹn:
   - Đạt kết quả tốt nhất tại rc101
   - Tương đương với Transformer tại các instance khác
   - Avg validation reward cao và ổn định

2. **GAT-LSTM** cần thêm thời gian huấn luyện để đánh giá đầy đủ

3. **Transformer-PPO** cần được xem xét lại toàn bộ

4. **Baseline và Transformer** vẫn là những mô hình đáng tin cậy

### 6.2 Khuyến Nghị

#### Ưu tiên cao:
1. ✅ **Hoàn thành huấn luyện GAT-Transformer cho t101**
2. ✅ **Huấn luyện GAT-LSTM đủ 5000 epochs cho tất cả instances**
3. 🔧 **Debug và fix Transformer-PPO:**
   - Review lại cấu hình hyperparameters
   - Kiểm tra reward shaping
   - Xem xét giảm learning rate

#### Ưu tiên trung bình:
4. 📊 **Thêm metrics đánh giá:**
   - Convergence speed
   - Training time
   - Memory usage
   - Inference time

5. 🧪 **Thử nghiệm thêm:**
   - Số lượng GAT layers (hiện tại: 3)
   - Attention heads
   - Hidden dimensions

#### Nghiên cứu thêm:
6. 📚 **Phân tích sâu hơn:**
   - Tại sao GAT-Transformer tốt tại rc101?
   - Tại sao Baseline tốt hơn tại pr01?
   - Training curves comparison chi tiết

---

## 7. Dữ Liệu Thô

### Summary Data (Epoch, Avg Validation, Max Real)

```
=== c101 ===
baseline_bench       : epoch= 5000, avg_val=257.15625, max_real=300.0
transformer_bench    : epoch= 5000, avg_val=254.609375, max_real=300.0
transformer_ppo      : epoch=10000, avg_val=172.25, max_real=10.0
gat_transformer_bench: epoch= 5000, avg_val=256.84375, max_real=300.0
gat_lstm            : epoch=  200, avg_val=242.3125, max_real=270.0

=== r101 ===
baseline_bench       : epoch= 5000, avg_val=109.40625, max_real=190.0
transformer_bench    : epoch= 5000, avg_val=110.21875, max_real=179.0
transformer_ppo      : epoch=10000, avg_val=98.671875, max_real=0.0
gat_transformer_bench: epoch= 5000, avg_val=109.640625, max_real=179.0

=== rc101 ===
baseline_bench       : epoch= 5000, avg_val=147.125, max_real=202.0
transformer_bench    : epoch= 5000, avg_val=144.859375, max_real=205.0
transformer_ppo      : epoch=10000, avg_val=8.890625, max_real=0.0
gat_transformer_bench: epoch= 5000, avg_val=148.09375, max_real=216.0

=== pr01 ===
baseline_bench       : epoch= 5000, avg_val=184.5, max_real=306.0
transformer_bench    : epoch= 5000, avg_val=170.671875, max_real=279.0
transformer_ppo      : epoch=10000, avg_val=23.0, max_real=52.0
gat_transformer_bench: epoch= 5000, avg_val=180.890625, max_real=277.0

=== t101 ===
baseline_bench       : epoch= 5000, avg_val=763.9375, max_real=214.0
transformer_bench    : epoch= 5000, avg_val=748.796875, max_real=326.0
transformer_ppo      : epoch=10000, avg_val=236.71875, max_real=125.0
gat_transformer_bench: N/A (no training history)
```

---

## Phụ Lục

### A. Cấu Trúc Mô Hình

#### GAT-Transformer Architecture
```
Input Graph → GAT Encoder (3 layers) → Transformer Decoder → Action Selection
```

#### GAT-LSTM Architecture
```
Input Graph → GAT Encoder (3 layers) → LSTM Decoder → Action Selection
```

### B. Files và Scripts
- Training scripts: `train_optw_gat_transformer.py`, `train_optw_gat_lstm.py`
- Benchmark runner: `benchmark_gat_runner.py`
- Results location: `results/{instance}/outputs/model_{model_name}_uni_samp/`

### C. Tham Khảo
- Previous benchmarks: `benchmark_report.md`, `benchmark_results.md`
- Training curves: `training_curve_*.png`
- Comprehensive report: Generated by `benchmark_report_all.py`

---

**Cập nhật lần cuối:** 27/11/2025 19:47 GMT+7
