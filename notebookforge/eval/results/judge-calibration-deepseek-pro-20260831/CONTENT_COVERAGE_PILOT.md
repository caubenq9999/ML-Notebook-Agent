# Pilot: Content Completeness & Learning Coverage

## Thiết lập

- Judge: `deepseek-v4-pro`
- Số lần chấm: 1 lần/notebook, tổng 2 API calls
- ResearchBundle: cùng frozen bundle gồm 12 key concepts
- Prompt: `prompts/candidate_content_coverage.txt`
- Thay đổi: bỏ `Executability`; thêm `Content Completeness` và `Learning Coverage`
- Executor/hard rules vẫn là gate riêng, không được tính thành điểm cộng chất lượng.

## Kết quả Judge trả về

| Notebook | Groundedness | Difficulty Fit | Pedagogical Order | Content Completeness | Learning Coverage | Trung bình |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Groq `4d40f588` | 3 | 3 | 3 | 2 | 2 | **2.60** |
| DeepSeek `20260830-v2` | 4 | 4 | 4 | 3 | 4 | **3.80** |

So với rubric cũ, hai notebook từ hòa `4.25` đã tách nhau `1.20` điểm. Kết quả này phù hợp hơn với quan sát thực tế: notebook Groq có code và cấu trúc bề ngoài nhưng phần giảng giải mỏng, thiếu trực giác và thiếu diễn giải output.

## Phân loại concept do Judge trả về

### Groq `4d40f588`

- Covered (6/12): sigmoid function, log loss, LogisticRegression, penalty, solver, ROC-AUC
- Shallow (6/12): logit, class_weight, Confusion Matrix, Precision, Recall, F1-Score
- Missing: không có

### DeepSeek `20260830-v2`

- Covered (12/12): toàn bộ 12 key concepts
- Shallow: không có
- Missing: không có

## Bất nhất phải sửa trước khi đưa vào production

Judge không tuân thủ hoàn toàn công thức Coverage trong prompt:

- Groq có 6/12 concept covered = 50%, theo rubric phải là `3`, nhưng Judge trả `2`.
- DeepSeek có 12/12 concept covered = 100%, theo rubric phải là `5`, nhưng Judge trả `4`.
- Feedback của DeepSeek nói một số concept còn sơ sài, nhưng danh sách lại xếp toàn bộ vào `covered`. Phân loại concept và lập luận chưa hoàn toàn nhất quán.

Vì vậy, kết quả pilot cho thấy **hướng rubric mới tốt hơn trong việc phân biệt notebook**, nhưng chưa nên dùng nguyên output LLM làm gate. Nên để LLM chỉ phân loại concept; chương trình tính `Learning Coverage` tất định từ tỷ lệ covered và validate ba danh sách concept.

## Chi phí

| Notebook | Input tokens | Output tokens | Cost |
| --- | ---: | ---: | ---: |
| Groq `4d40f588` | 8,303 | 355 | $0.012366 |
| DeepSeek `20260830-v2` | 11,884 | 513 | $0.017718 |
| **Tổng** | **20,187** | **868** | **$0.030084** |

Raw result: `content_coverage_pilot.json`.
