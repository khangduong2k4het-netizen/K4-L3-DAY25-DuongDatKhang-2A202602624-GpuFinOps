# Day 25 — GPU FinOps Lab: Bài viết ngắn

Dữ liệu: `python data/generate.py` (seed cố định). Mọi số liệu lấy từ `python missions/run_all.py`; chi phí GPU tính theo tháng 30 ngày.
Kiểm tra: `python verify.py` → 11/11; `pytest -q` → 15 passed; test riêng cho phần mở rộng: `pytest -q extensions_tests/check_extensions.py` → 4 passed.

## 1. Baseline và Optimized

| | Baseline | Optimized |
|---|---|---|
| Tổng chi phí / tháng | $27,133 | $14,881 (**-45%**) |
| Inference $/1M-token | $6.488 | $1.149 (-82.3%) |
| GPU (workloads) / tháng | $25,667 on-demand | $15,877 (-38.1%) |

## 2. Từng đòn bẩy

| Đòn bẩy | USD/tháng | Tỷ trọng |
|---|---|---|
| Purchasing (spot/reserved) | $9,790 | 80% |
| Inference (cascade/cache/batch) | $1,207 | 10% |
| Right-size util-lies | $655 | 5% |
| Kill idle GPUs | $600 | 5% |

Purchasing đóng góp nhiều nhất về số tiền vì chi phí GPU ($25.7k/tháng) lớn gấp khoảng 17 lần chi phí API ($1.47k/tháng). Inference có tỷ lệ giảm cao nhất (82%) nhưng áp dụng trên nền chi phí nhỏ. Nếu tính lần lượt từng bước trên inference: cascade giảm $37.40/ngày, thêm cache giảm $1.02, thêm batch giảm $1.80. Như vậy cascade (chuyển request sang model small) là phần quyết định.

## 3. GPU-Util Lie

- `gpu-h100-4`: util 98.2% nhưng MFU 0.194, MBU 0.207. GPU luôn "bận" nhưng chỉ dùng khoảng 1/5 compute.
- `gpu-a10g-1`: util 96.9%, MFU 0.268.

Hạ một bậc (H100→A100, A10G→L4) tiết kiệm $511 + $144 = **$655/tháng**. Ngoài ra `gpu-h100-5` để không 8 giờ/ngày, tốn $20/ngày = **$600/tháng**. Lưu ý: chưa có benchmark nào chứng minh throughput giữ nguyên sau khi hạ bậc.

## 4. Phần mở rộng đã làm

### Extension 1 — `recommend_tier()` dựa trên chi phí

**Thay đổi** (`finops/pricing.py`, `missions/m3_purchasing.py`): khi có giá, `recommend_tier()` tính chi phí tháng của mọi lựa chọn hợp lệ rồi chọn mức rẻ nhất:
- **Spot:** tỷ lệ gián đoạn theo từng loại GPU (`SPOT_INTERRUPT_RATE`). Bỏ qua spot nếu tỷ lệ > 15%/giờ.
- **Reserved:** commit bị tính **24h/ngày** dù có dùng hay không. Commit còn bị tính trọn kỳ hạn dù nhu cầu kết thúc sớm (3yr cho nhu cầu 24 tháng → trả gấp 1.5 lần).
- **1yr hay 3yr:** chọn theo thời gian cần GPU (horizon).

Nếu không truyền giá, hàm vẫn dùng policy cũ nên test gốc không đổi.

**Giả định:**
- Tỷ lệ gián đoạn: H100/H200 3%, B200 4%, A100/MI300X 5%, L4 8%, A10G 10% mỗi giờ. Mốc H100 <5% lấy từ deck. Các GPU nhỏ, phổ biến được đặt cao hơn, tham khảo cách phân nhóm của AWS Spot Advisor. Đây là giả định, không phải số đo.
- Job chạy đủ 30 ngày/tháng được coi là dịch vụ dài hạn, horizon 36 tháng. Job ngắn hơn là dự án một lần nên không reserve.

| Job | GPU | Trước | Sau | Trước $/tháng | Sau $/tháng |
|---|---|---|---|---|---|
| job-train-llm | H100 | spot | spot | 7,596 | 7,524 |
| job-train-embed | A100 | spot | spot | 1,393 | 1,393 |
| job-finetune | H100 | spot | spot | 570 | 564 |
| job-infer-chat | A10G | reserved | reserved 3yr | 2,592 | 2,592 |
| job-infer-rag | A100 | reserved | reserved 3yr | 2,160 | 2,160 |
| job-infer-search | L4 | reserved | reserved 3yr | 972 | **1,296** |
| job-dev-sandbox | A10G | spot | spot | 203 | 207 |
| job-batch-eval | H100 | spot | spot | 142 | 141 |

**savings_pct M3: 39.1% → 38.1%.** Mức giảm này là do sửa một chỗ tính quá lạc quan, không phải do policy kém hơn:
- **job-infer-search:** chạy 18h/ngày nhưng reserved phải trả 24h. Code cũ chỉ tính 18h nên báo tiết kiệm $756; thực tế là $432.
- **Spot H100:** rẻ hơn một chút vì tỷ lệ gián đoạn thấp hơn (3% so với mặc định 5%).
- **Spot A10G:** đắt hơn một chút vì tỷ lệ gián đoạn cao hơn (10%).

Với horizon mặc định không job nào đổi tier. Tier chỉ đổi khi horizon ngắn hơn:

| Horizon | infer-chat | infer-rag | infer-search | Savings |
|---|---|---|---|---|
| 12 hoặc 24 tháng | reserved 1yr | reserved 1yr | **on_demand** | 29.7% |
| 36 tháng | reserved 3yr | reserved 3yr | reserved 3yr | 38.1% |

Ở 12 tháng, 1yr L4 tính 24h × $0.60 = $14.4/ngày, bằng đúng on-demand 18h × $0.80, nên không còn lợi gì khi commit. **Insight:** quyết định 3yr chỉ đáng khi chắc chắn dịch vụ còn chạy ít nhất khoảng 2 năm. Sai horizon làm mất khoảng 8 điểm phần trăm savings.

### Extension 3 — `cache_is_worth_it()`

**Thay đổi:** thêm `cache_break_even_reads()`, `cache_cost_multiplier()` và `cache_is_worth_it()` vào `finops/pricing.py`. M2 chỉ bật cache cho team vượt ngưỡng hòa vốn, và tính cả tiền ghi cache.

**Công thức:** một prefix được ghi 1 lần rồi đọc R lần. Có cache tốn `1.25 + 0.1R`, không cache tốn `1 + R` (tính theo giá input gốc). Hòa vốn tại **R\* = 0.25/0.9 ≈ 0.28 lần đọc** với TTL 5 phút. Với TTL 1 giờ (ghi 2×), R\* ≈ 1.11. Mức giá ghi 1.25×/2× và đọc 0.1× theo bảng giá prompt caching của Anthropic.

**Giả định:** CSV không có số lần đọc mỗi prefix, nên đặt số đọc theo team:

| Team | Số đọc/prefix | Hệ số giá | Bật cache? |
|---|---|---|---|
| assistant | 20 | 0.155 | có |
| rag | 10 | 0.205 | có |
| eval | 3 | 0.388 | có |
| search | 0.2 | 1.058 | **không** |

| | $/ngày | $/1M-token |
|---|---|---|
| Trước (tiền ghi = 0) | 8.48 | 1.126 |
| Sau (có tiền ghi, có gating) | 8.65 | 1.149 |

Hai kịch bản biên (đã gồm cascade và batch), $/ngày:

| Kịch bản | Không cache | Luôn cache | Gated |
|---|---|---|---|
| Ít đọc (0.2/prefix) | 9.66 | 9.73 (lỗ) | 9.66 |
| Nhiều đọc (20/prefix) | 9.66 | 8.56 | 8.56 |

**Insight:**
- Bỏ qua tiền ghi làm chi phí bị báo thấp hơn khoảng 2%.
- Khi số đọc thấp, bật cache cho mọi request còn **đắt hơn** không cache. Gating không bao giờ tệ hơn lựa chọn tốt nhất trong hai cách.
- Ngưỡng R\* rất thấp, nên rủi ro thật nằm ở TTL hết hạn trước khi prefix được đọc lại. Cần đo số đọc mỗi prefix thay vì đoán.

### Extension 4 — Ngân sách Reasoning

**Thay đổi:** `reasoning_budget()` trong M2 tách nhóm `is_reasoning` để tính tiền, token và Wh (dùng `wh_per_query`), sau đó áp cap. M5 có thêm section tương ứng.

| Nhóm | Requests | Tokens | Chi phí | Năng lượng |
|---|---|---|---|---|
| Không reasoning | 91.6% | 83.5% | 83.8% | 6.0% |
| Reasoning | 8.4% | 16.5% | 16.2% | **94.0%** |

- **Cap 10%:** traffic reasoning đã ở mức 8.4%, dưới cap, nên **savings = 0**.
- **Thử cap 5%:** hạ 81 request reasoning ở route small (giữ lại các request route large vì router đã đánh giá là khó). Tiết kiệm chỉ $1.60/tháng nhưng giảm **357 kWh/tháng (38% năng lượng inference)**. Phần này giả định request bị hạ chỉ sinh 1/6 output trên cùng model, nên chất lượng câu trả lời có thể giảm.

**Insight:** reasoning gần như "miễn phí" về tiền (vì phần lớn chạy trên model small) nhưng là gánh nặng năng lượng (hệ số 80×). Cần quản trị nó bằng ngân sách Wh/carbon, không chỉ bằng USD. Đề xuất:
- Chỉ dùng reasoning khi router chọn route large hoặc khi confidence của lần trả lời đầu thấp hơn ngưỡng.
- Đặt cap theo Wh cho team eval.

## 5. Khuyến nghị cho NimbusAI

1. **Chốt chiến lược mua GPU (khoảng $9.8k/tháng):**
   - Đưa training lên spot có checkpoint.
   - Reserve chat/rag **sau khi** xác nhận horizon. Nếu dịch vụ không chắc chạy quá 2 năm thì chọn 1yr.
   - Giữ `job-infer-search` ở on-demand cho đến khi chắc chắn chạy 3 năm.
2. **Xử lý GPU lãng phí ($1.26k/tháng):**
   - Benchmark `gpu-h100-4` trên A100 rồi mới chuyển.
   - Tắt `gpu-h100-5` khi idle bằng autoscaling hoặc lịch tắt.
3. **Đo lường trước khi tối ưu tiếp inference:**
   - Ghi log số đọc mỗi prefix cache để thay các giả định ở Ext 3.
   - Gắn ngân sách Wh cho reasoning. Đây là đòn bẩy carbon lớn nhất dù số tiền nhỏ.
