# Controlled ablation 2×2 — 12 concepts, threshold 0.45

Canonical ResearchBundle và notebook attempt 1 được chia sẻ theo từng cặp RAG/No-RAG để giảm nhiễu stochastic.
Research LLM reranking không chạy lại; chi phí research của artifact canonical không tính vào run này.

## Canonical concepts

`sigmoid function`, `logit`, `log loss`, `LogisticRegression`, `penalty`, `solver`, `class_weight`, `Confusion Matrix`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`

| Condition | Concepts | Initial hash | Selected hash | Attempts | Chunks | Context chars | Exec | Rules | Exec score | Grounded | Difficulty | Pedagogy | Average | Input tok | Output tok | Sim. system cost | Eval cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rag-verifier | 12 | 0817c9d6cad9 | 0817c9d6cad9 | 1 | 9 | 2109 | True | 8/8 | 5.0 | 4.0 | 4.0 | 4.0 | 4.25 | 20066 | 6865 | $0.020342 | $0.001554 |
| rag-noverifier | 12 | 0817c9d6cad9 | 0817c9d6cad9 | 1 | 9 | 2109 | True | 8/8 | 5.0 | 4.0 | 4.0 | 4.0 | 4.25 | 11837 | 6574 | $0.010941 | $0.001522 |
| norag-verifier | 12 | 17daf2ec36c7 | 17daf2ec36c7 | 1 | 0 | 0 | True | 8/8 | 5.0 | 4.0 | 4.0 | 4.0 | 4.25 | 17848 | 6067 | $0.019153 | $0.001455 |
| norag-noverifier | 12 | 17daf2ec36c7 | 17daf2ec36c7 | 1 | 0 | 0 | True | 8/8 | 5.0 | 4.0 | 4.0 | 4.0 | 4.25 | 10141 | 5821 | $0.010619 | $0.001403 |

## Chi phí API thực tế của controlled run

- Hai shared bases: **$0.021560**
- Verifier/retry branches: **$0.017935**
- Post-hoc evaluation: **$0.005934**
- Tổng: **$0.045429**

`Sim. system cost` là chi phí một condition nếu tính cả shared base tương ứng; không cộng bốn dòng này để tính actual spend vì mỗi base được tái sử dụng hai lần.

## Diễn giải

- Cả 4 condition dùng đúng cùng 12 canonical concepts. Hai condition RAG có cùng initial hash `0817c9d6cad9`; hai condition No-RAG có cùng initial hash `17daf2ec36c7`.
- Cả hai initial notebook đều execution PASS và đạt 8/8 hard rules. Vì Verifier cho PASS ngay attempt 1, nhánh Verifier và No-Verifier trong cùng treatment giữ nguyên một notebook. Trong sample này, Verifier tăng simulated system cost khoảng $0.0085-$0.0094 nhưng không thay đổi output.
- RAG đưa 9 chunks và 2,109 ký tự theory context vào LearningPath; No-RAG đưa 0. Tuy nhiên Groundedness vẫn là 4.0 ở cả hai notebook, nên rubric/Judge hiện chưa phân biệt được tác động này bằng điểm số tổng hợp.
- Notebook RAG nhận feedback chủ yếu về lý thuyết còn thiếu công thức/giải thích hyperparameter. Notebook No-RAG nhận feedback về sự không nhất quán giữa dữ liệu scaled và dữ liệu dùng để train/predict. Đây là khác biệt định tính nhưng chưa phản ánh vào điểm trung bình.
- Judge đã tạo feedback hơi khác nhau khi chấm lại cùng một notebook hash, dù bốn điểm không đổi. Benchmark chính thức nên cache evaluation theo notebook hash hoặc khóa temperature của Judge để giảm nhiễu.

## Phạm vi

Canonical bundle này là fixture của eval; production Research Agent chưa bị hardcode thành đúng 12 concepts. Cách này nhằm cô lập treatment trong ablation, không thay đổi hành vi cho người dùng thật.
