# RAG threshold sweep — Logistic Regression

- Concepts: 10 concepts dùng trong pilot DeepSeek
- Embedding: `paraphrase-multilingual-MiniLM-L12-v2`
- KB: 5 files, 69 semantic chunks
- LLM calls/cost: 0 / $0

| Threshold | Coverage | Unique chunks | Chunk chars | Modules | Context chars | Mean similarity | Missing |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 9/10 (90%) | 13 | 4148 | 4/5 | 2418 | 0.5306 | F1-Score |
| 0.35 | 9/10 (90%) | 13 | 4148 | 4/5 | 2418 | 0.5306 | F1-Score |
| 0.40 | 9/10 (90%) | 12 | 3706 | 4/5 | 2418 | 0.5418 | F1-Score |
| 0.45 | 9/10 (90%) | 10 | 3328 | 4/5 | 2418 | 0.5608 | F1-Score |
| 0.50 | 7/10 (70%) | 9 | 2531 | 4/5 | 2036 | 0.5718 | log loss, Recall, F1-Score |
| 0.55 | 5/10 (50%) | 6 | 1888 | 3/5 | 1693 | 0.5941 | logit, log loss, LogisticRegression, Recall, F1-Score |
| 0.60 | 1/10 (10%) | 1 | 503 | 1/5 | 503 | 0.7258 | sigmoid function, logit, log loss, LogisticRegression, Confusion Matrix, Precision, Recall, F1-Score, StandardScaler |
| 0.65 | 1/10 (10%) | 1 | 503 | 1/5 | 503 | 0.7258 | sigmoid function, logit, log loss, LogisticRegression, Confusion Matrix, Precision, Recall, F1-Score, StandardScaler |
| 0.70 | 1/10 (10%) | 1 | 503 | 1/5 | 503 | 0.7258 | sigmoid function, logit, log loss, LogisticRegression, Confusion Matrix, Precision, Recall, F1-Score, StandardScaler |

Nội dung từng chunk và preview được lưu trong `sweep_results.json` để kiểm tra relevance trước khi chốt threshold.

## Kết luận

Không nên hạ threshold một cách độc lập chỉ để tăng coverage:

- `0.45` là điểm gãy tốt nhất theo số liệu thô: tăng từ 7/10 lên 9/10 so với `0.50`, chỉ thêm 1 unique chunk.
- Tuy nhiên coverage 9/10 ở `0.45` có một false association: query `log loss` chọn chunk về Log-Odds (0.4883), trong khi chunk định nghĩa Log Loss đúng chỉ đạt 0.3905 và nằm ngoài top-2.
- `F1-Score` không được retrieve ở mọi threshold đã sweep. Chunk đúng chứa công thức F1 chỉ đạt 0.1920 và đứng thứ 4; vì vậy hạ threshold cũng không giúp khi `top_k=2`.
- Với `Recall`, chunk F1 liên quan đạt 0.4739 và được lấy ở `0.45`; chunk định nghĩa Recall trực tiếp chỉ đạt 0.2695.

Khuyến nghị tạm thời: giữ `0.50` nếu ưu tiên precision. Trước khi chạy ablation chính thức, bổ sung lexical/exact-term boost hoặc alias query (`F1 score`, `F1`, `harmonic mean`; `binary cross-entropy`, `log loss`) rồi sweep lại quanh `0.40-0.50`. Chỉ đổi threshold sang `0.45` sau khi đã sửa bước xếp hạng này.
