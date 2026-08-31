# Golden-6 Benchmark — New LLM Judge

## Thiết lập

- Golden subset: `GS-001`, `GS-006`, `GS-008`, `GS-012`, `GS-015`, `GS-018`.
- Bao phủ đủ 3 topic × 2 level.
- Worker: `deepseek-v4-flash`; Judge: `deepseek-v4-pro`.
- Judge mới bỏ `Executability`, thêm `Content Completeness` và `Learning Coverage`.
- Learning Coverage được tính bằng code từ tỷ lệ concept `covered`, không dùng điểm tự khai của LLM.
- Mỗi case tối đa 2 attempt và $0.30.

## Kết quả chính

| Metric | Kết quả | Mục tiêu | Đạt? |
| :--- | ---: | ---: | :---: |
| Execution Pass Rate | **5/6 = 83.3%** | ≥ 80% | Có |
| Final PASS Rate | **4/6 = 66.7%** | ≥ 80% | Không |
| LLM Rubric Score | **4.000/5** | ≥ 3.5 | Có |
| Average Cost / Case | **$0.0406** | ≤ $0.30 | Có |
| Average Time / Case | **89.79s** | ≤ 180s | Có |

## Giải thích bốn nhóm metric

### Execution Pass Rate

Tỷ lệ notebook cuối được chọn của mỗi case mà Executor chạy hết, không crash và không timeout. Đây là gate kỹ thuật độc lập; nó không còn được tính vào điểm LLM.

### Final PASS Rate

Tỷ lệ session kết thúc với quyết định `PASS`. Một case chỉ PASS khi đồng thời: Executor thành công, đủ 8 hard rules, điểm Judge mới trung bình ≥ 3.5/5, và chưa vượt 2 attempt hoặc trần $0.30.

### LLM Rubric Score

Trung bình của 5 tiêu chí: Groundedness, Difficulty Fit, Pedagogical Order, Content Completeness và Learning Coverage. Điểm benchmark lấy từ attempt/notebook cuối được hệ thống chọn.

| Tiêu chí Judge mới | Điểm trung bình |
| :--- | ---: |
| Groundedness | 4.000 |
| Difficulty Fit | 4.000 |
| Pedagogical Order | 4.000 |
| Content Completeness | 3.000 |
| Learning Coverage | 5.000 |

### Cost / Time

Cost là tổng chi phí LLM của toàn session: Research, Curriculum, Notebook Generator và Judge ở mọi attempt. Time là thời gian end-to-end, bao gồm gọi model, sinh notebook, execute và verify.

- Tổng cost: **$0.243799**.
- Cost/case: trung bình **$0.040633**, thấp nhất **$0.030241**, cao nhất **$0.058450**.
- Tổng thời gian: **538.72s** (8.98 phút).
- Time/case: trung bình **89.79s**, thấp nhất **63.41s**, cao nhất **162.67s**.

## Chi tiết từng case

| Case | Topic | Level | Execution | Final | LLM score | Cost | Time |
| :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: |
| GS-001 | logistic_regression | 1 | PASS | FAIL_MAX_RETRY | 4.000 | $0.058450 | 162.67s |
| GS-006 | logistic_regression | 2 | PASS | PASS | 4.000 | $0.033233 | 63.41s |
| GS-008 | decision_tree | 1 | FAIL | FAIL_MAX_RETRY | 4.000 | $0.053303 | 108.44s |
| GS-012 | decision_tree | 2 | PASS | PASS | 4.000 | $0.037060 | 73.60s |
| GS-015 | kmeans | 1 | PASS | PASS | 4.000 | $0.030241 | 64.59s |
| GS-018 | kmeans | 2 | PASS | PASS | 4.000 | $0.031512 | 66.01s |

## Nhận xét

- `GS-001` chạy được ở attempt 2 nhưng vẫn FAIL vì hard rule `no_hardcoded_answers` không đạt.
- `GS-008` vẫn lỗi execution ở attempt 2 và thiếu số lượng visualization tối thiểu, nên FAIL.
- Bốn case còn lại PASS ngay attempt 1.
- Sáu notebook cuối đều nhận cùng vector Judge `(4, 4, 4, 3, 5)`. Rubric mới đã hạ `Content Completeness` xuống 3, nhưng độ phân giải giữa các notebook DeepSeek vẫn thấp; cần human labels/Kappa để biết mức 3 này có chính xác không.
- `Learning Coverage = 5` ở cả sáu case vì Judge xếp ít nhất 90% concept vào `covered`. Điều này cho thấy định nghĩa `covered` có thể vẫn rộng và cần kiểm chuẩn bằng người chấm.

## Lưu ý triển khai

Benchmark dùng adapter eval để đưa trung bình 5 tiêu chí vào contract `VerifierReport` cũ. Production schema/prompt chưa bị thay đổi trong lượt chạy này.

Raw data: `summary.json`, `checkpoint.json`, `judge_attempts.json`.
