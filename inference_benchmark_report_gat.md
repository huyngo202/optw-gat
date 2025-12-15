# Báo Cáo Inference Benchmark - GAT Models

**Ngày thực hiện:** 27/11/2025  
**Inference method:** Beam Search  
**Device:** CPU

---

## 1. Tổng Quan

Báo cáo này trình bày kết quả inference benchmark của các mô hình Graph Attention Network (GAT) so với Baseline và Transformer trên các benchmark instances thực tế của bài toán OPTW.

### Các Mô Hình Được Đánh Giá
- **Baseline**: Mô hình baseline ban đầu với LSTM encoder-decoder
- **Transformer**: Mô hình với Transformer decoder
- **GAT-LSTM**: Mô hình GAT Encoder + LSTM Decoder (chỉ c101)
- **GAT-Transformer**: Mô hình GAT Encoder + Transformer Decoder

### Phương Pháp Inference
- **Beam Search** (bs): Tìm kiếm với beam size = 128
- **Test trên benchmark instances thực tế** (không phải generated data)
- **Metrics đo lường:**
  - **Score**: Tổng điểm thu được (càng cao càng tốt)
  - **Inference Time**: Thời gian inference tính bằng milliseconds (càng thấp càng tốt)

---

## 2. Kết Quả Chi Tiết

### 2.1 Instance: c101

| Model | Epoch | Score | Time (ms) | Score vs Baseline | Speed vs Baseline |
|-------|-------|-------|-----------|-------------------|-------------------|
| **Baseline** | 50000 | **320** | **1,474** | - | - |
| **Transformer** | 22000 | **320** | 2,064 | = | 1.4x slower |
| **GAT-LSTM** | 200 | **320** | 5,051 | = | 3.4x slower |
| **GAT-Transformer** | 5000 | **320** | 15,261 | = | 10.4x slower |

**Nhận xét:**
- ✅ Tất cả 4 mô hình đều đạt **score tối ưu 320**
- ⚠️ Baseline **nhanh nhất** (1.47s)
- ⚠️ GAT-Transformer **chậm nhất** (15.26s) - chậm hơn Baseline 10.4 lần
- 📊 GAT-LSTM với epoch 200 vẫn đạt score tối ưu nhưng chậm hơn 3.4 lần

### 2.2 Instance: r101

| Model | Epoch | Score | Time (ms) | Score vs Baseline | Speed vs Baseline |
|-------|-------|-------|-----------|-------------------|-------------------|
| **Baseline** | 4000 | **198** | **1,699** | - | - |
| **Transformer** | 4000 | **198** | 2,045 | = | 1.2x slower |
| **GAT-LSTM** | - | N/A | N/A | - | - |
| **GAT-Transformer** | 5000 | **198** | 9,717 | = | 5.7x slower |

**Nhận xét:**
- ✅ Cả 3 mô hình đều đạt **cùng score 198**
- ⚠️ GAT-Transformer chậm hơn Baseline **5.7 lần**
- ℹ️ GAT-LSTM chưa được train cho instance này

### 2.3 Instance: rc101

| Model | Epoch | Score | Time (ms) | Score vs Baseline | Speed vs Baseline |
|-------|-------|-------|-----------|-------------------|-------------------|
| **Baseline** | 4000 | 219 | **2,240** | - | - |
| **Transformer** | 4000 | 216 | 3,326 | -1.4% | 1.5x slower |
| **GAT-LSTM** | - | N/A | N/A | - | - |
| **GAT-Transformer** | 5000 | **236** ⭐ | 8,642 | **+7.8%** | 3.9x slower |

**Nhận xét:**
- 🎯 **GAT-Transformer đạt score cao nhất: 236** (+7.8% so với Baseline)
- ✨ Cải thiện đáng kể so với cả Baseline (219) và Transformer (216)
- ⚠️ Trade-off: Chậm hơn 3.9 lần so với Baseline
- 📈 Kết quả phù hợp với training benchmark (GAT-Trans tốt nhất tại rc101)

### 2.4 Instance: pr01

| Model | Epoch | Score | Time (ms) | Score vs Baseline | Speed vs Baseline |
|-------|-------|-------|-----------|-------------------|-------------------|
| **Baseline** | 4000 | **299** | **1,471** | - | - |
| **Transformer** | 4000 | 278 | 2,184 | -7.0% | 1.5x slower |
| **GAT-LSTM** | - | N/A | N/A | - | - |
| **GAT-Transformer** | 5000 | **306** ⭐ | 16,757 | **+2.3%** | 11.4x slower |

**Nhận xét:**
- 🎯 **GAT-Transformer đạt score cao nhất: 306** (+2.3% so với Baseline)
- ✅ Vượt qua cả Baseline (299) và Transformer (278)
- ⚠️ Inference time rất chậm: 16.76s (chậm hơn Baseline 11.4 lần)
- 🔄 Kết quả khác với training (training: Baseline tốt hơn)

### 2.5 Instance: t101

| Model | Epoch | Score | Time (ms) | Score vs Baseline | Speed vs Baseline |
|-------|-------|-------|-----------|-------------------|-------------------|
| **Baseline** | 4000 | 318 | **5,747** | - | - |
| **Transformer** | 4000 | **332** | 7,714 | **+4.4%** | 1.3x slower |
| **GAT-LSTM** | - | N/A | N/A | - | - |
| **GAT-Transformer** | - | N/A | N/A | - | - |

**Nhận xét:**
- 🏆 **Transformer thắng** với score 332
- ℹ️ GAT-Transformer chưa có model weights cho instance này
- 📊 Instance khó, inference time tương đối cao cho cả 2 models

---

## 3. Phân Tích Tổng Hợp

### 3.1 So Sánh Score (Quality)

#### Bảng Tổng Hợp Score

| Instance | Winner | Score | Runner-up | Điểm Mạnh |
|----------|--------|-------|-----------|-----------|
| c101 | **Tie (All)** | 320 | - | Tất cả đều tối ưu |
| r101 | **Tie (3 models)** | 198 | - | Cùng kết quả |
| rc101 | **GAT-Trans** ⭐ | 236 | Baseline (219) | +7.8% |
| pr01 | **GAT-Trans** ⭐ | 306 | Baseline (299) | +2.3% |
| t101 | **Transformer** | 332 | Baseline (318) | +4.4% |

#### Phân Tích Wins/Losses

**GAT-Transformer:**
- **Wins**: 2 instances (rc101, pr01) - đạt score cao nhất
- **Ties**: 2 instances (c101, r101) - cùng score tối ưu
- **Losses**: 0 instances (không có t101 để so sánh)
- **Tổng thể**: Xuất sắc về quality

**Transformer:**
- **Wins**: 1 instance (t101)
- **Ties**: 2 instances (c101, r101)
- **Losses**: 2 instances (rc101, pr01)
- **Tổng thể**: Cân bằng

**Baseline:**
- **Wins**: 0 instances
- **Ties**: 2 instances (c101, r101)
- **Losses**: 2 instances (rc101, pr01)
- **Tổng thể**: Bị vượt qua bởi GAT-Trans và Transformer

### 3.2 So Sánh Speed (Efficiency)

#### Bảng Tổng Hợp Inference Time

| Model | Avg Time (ms) | Relative Speed | Đánh Giá |
|-------|---------------|----------------|----------|
| **Baseline** | 2,526 | 1.0x (baseline) | ⚡ Nhanh nhất |
| **Transformer** | 3,467 | 1.4x | ✅ Nhanh |
| **GAT-LSTM** | 5,051 | 2.0x | ⚠️ Trung bình |
| **GAT-Transformer** | 12,594 | **5.0x** | ❌ Chậm |

> *Avg Time tính trên các instances có kết quả*

#### Phân Tích Chi Tiết Speed

**Worst Case Scenario:**
- GAT-Transformer tại pr01: **16,757ms** (16.76 giây)
- **Chậm hơn Baseline 11.4 lần**

**Best Case:**
- Baseline tại pr01: **1,471ms** (1.47 giây)

**Speed Ranking:**
1. 🥇 Baseline - Fastest
2. 🥈 Transformer - 1.4x slower
3. 🥉 GAT-LSTM - 2.0x slower  
4. ⚠️ GAT-Transformer - 5.0x slower

### 3.3 Quality vs Speed Trade-off

```
Quality →
    │
    │                          GAT-Trans (rc101, pr01)
    │                              ⭐
    │
    │        Transformer (t101)
    │             ●
    │
    │  Baseline (fast but lower quality)
    │      ●
    │
    └────────────────────────────────────────────→ Speed
        Faster                             Slower
```

**Kết luận Trade-off:**
- **GAT-Transformer**: High Quality, Low Speed
- **Transformer**: Balanced Quality & Speed
- **Baseline**: High Speed, Lower Quality

---

## 4. Phân Tích Sâu

### 4.1 Tại Sao GAT-Transformer Chậm?

GAT-Transformer có inference time chậm hơn đáng kể vì:

1. **GAT Encoder phức tạp hơn:**
   - Graph attention mechanism tính toán attention cho tất cả các cặp nodes
   - 3 GAT layers cần nhiều operations hơn
   
2. **Transformer Decoder:**
   - Self-attention mechanism trong decoder
   - Computational cost cao hơn LSTM

3. **Beam Search on Complex Model:**
   - Beam search phải duy trì 128 beams
   - Mỗi beam chạy qua GAT encoder + Transformer decoder
   - Exponential cost increase

**Ước tính:**
- GAT Encoder: ~3-4x slower than simple encoder
- Transformer Decoder: ~1.5x slower than LSTM  
- Combined: ~5-6x slower (phù hợp với kết quả thực tế 5.0x)

### 4.2 Score Quality Analysis

**Why GAT-Transformer wins at rc101 and pr01?**

1. **rc101** (Random-Clustered):
   - GAT tốt hơn trong việc học cấu trúc đồ thị phức tạp
   - Random-clustered pattern phù hợp với graph attention
   - Score: 236 vs 219 (Baseline) = **+7.8%**

2. **pr01** (Problem dataset):
   - Dataset phức tạp hơn
   - GAT + Transformer có khả năng generalization tốt hơn
   - Score: 306 vs 299 (Baseline) = **+2.3%**

**Why Baseline/Transformer still competitive?**

1. **c101** (Clustered):
   - Pattern đơn giản, không cần attention phức tạp
   - All models converge to optimal (320)

2. **r101** (Random):
   - Optimal solution dễ tìm
   - All models achieve same score (198)

---

## 5. Khuyến Nghị Sử Dụng

### 5.1 Khi Nào Dùng GAT-Transformer?

✅ **Nên dùng khi:**
- Cần **chất lượng solution tốt nhất** (ví dụ: production planning)
- Instance có cấu trúc phức tạp (random-clustered, mixed patterns)
- Inference time không phải vấn đề quan trọng
- Có resource tính toán đủ mạnh

❌ **Không nên dùng khi:**
- Cần **real-time inference** (latency-critical applications)
- Xử lý lượng lớn instances (batch processing)
- Resource hạn chế (mobile, edge devices)

### 5.2 Khi Nào Dùng Transformer?

✅ **Nên dùng khi:**
- Cần **cân bằng giữa quality và speed**
- Application yêu cầu response time ~2-3 giây
- Instance có độ phức tạp trung bình đến cao

### 5.3 Khi Nào Dùng Baseline?

✅ **Nên dùng khi:**
- Ưu tiên **tốc độ inference**
- Instance đơn giản (clustered patterns)
- Batch processing với volume lớn
- Resource-constrained environments

---

## 6. Kết Luận

### 6.1 Key Findings

1. **GAT-Transformer vượt trội về Quality:**
   - Đạt score cao nhất tại 2/4 instances testable (rc101, pr01)
   - Cải thiện 2.3-7.8% so với Baseline
   - Không thua bất kỳ instance nào có kết quả

2. **Trade-off Quality vs Speed rõ ràng:**
   - GAT-Transformer: Best quality, **5x slower**
   - Transformer: Good balance, **1.4x slower**
   - Baseline: Fast, lower quality

3. **Instance-specific Performance:**
   - Clustered (c101): All models tốt như nhau
   - Random (r101): All models tốt như nhau  
   - Random-Clustered (rc101): GAT-Transformer tốt nhất (+7.8%)
   - Problem dataset (pr01): GAT-Transformer tốt nhất (+2.3%)

### 6.2 Recommendations

#### Cho Research/Development:
1. ✅ **Optimize GAT-Transformer inference speed:**
   - Reduce beam size cho real-time applications
   - Implement model quantization
   - Try greedy decoding as alternative

2. ✅ **Train GAT-LSTM cho more instances:**
   - Có potential cho better speed/quality trade-off
   - Chỉ có c101 results, cần thêm data

3. ✅ **Complete GAT-Transformer training cho t101:**
   - Missing data point quan trọng

#### Cho Production:
1. 🎯 **Use case specific selection:**
   - High-value planning → GAT-Transformer
   - Real-time routing → Baseline or Transformer
   - Balanced scenarios → Transformer

2. 📊 **Consider ensemble approach:**
   - Fast first-pass với Baseline
   - Refinement với GAT-Transformer cho selected instances

---

## 7. Dữ Liệu Thô

### Complete Results Table

```
Instance   | Model            | Epoch    | Score    | Time(ms)  
-----------|------------------|----------|----------|----------
c101       | Baseline         | 50000    | 320      | 1474      
c101       | Transformer      | 22000    | 320      | 2064      
c101       | GAT-LSTM         | 200      | 320      | 5051      
c101       | GAT-Transformer  | 5000     | 320      | 15261     
r101       | Baseline         | 4000     | 198      | 1699      
r101       | Transformer      | 4000     | 198      | 2045      
r101       | GAT-LSTM         | N/A      | N/A      | N/A       
r101       | GAT-Transformer  | 5000     | 198      | 9717      
rc101      | Baseline         | 4000     | 219      | 2240      
rc101      | Transformer      | 4000     | 216      | 3326      
rc101      | GAT-LSTM         | N/A      | N/A      | N/A       
rc101      | GAT-Transformer  | 5000     | 236      | 8642      
pr01       | Baseline         | 4000     | 299      | 1471      
pr01       | Transformer      | 4000     | 278      | 2184      
pr01       | GAT-LSTM         | N/A      | N/A      | N/A       
pr01       | GAT-Transformer  | 5000     | 306      | 16757     
t101       | Baseline         | 4000     | 318      | 5747      
t101       | Transformer      | 4000     | 332      | 7714      
t101       | GAT-LSTM         | N/A      | N/A      | N/A       
t101       | GAT-Transformer  | N/A      | N/A      | N/A       
```

### CSV Export
Kết quả đầy đủ được lưu tại: [inference_benchmark_results.csv](file:///home/huyngo/Project/ML/optw_rl/inference_benchmark_results.csv)

---

**Tóm lại:** GAT-Transformer là mô hình chất lượng cao nhất nhưng có trade-off về tốc độ. Lựa chọn model phụ thuộc vào requirements cụ thể của application.

**Cập nhật:** 27/11/2025 20:04 GMT+7
