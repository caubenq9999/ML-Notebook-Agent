# Pilot: Groq và DeepSeek tạo sinh notebook

## 1. Mục tiêu

Pilot này dùng hai notebook đã sinh để quan sát sơ bộ sự khác biệt giữa hai cấu hình mô hình. Đây chưa phải benchmark chính thức và chưa đủ để khẳng định mô hình nào tốt hơn.

- Groq: session `cli-4d40f588`.
- DeepSeek: session `cli-deepseek-20260830-v2`.

Hai session dùng cùng cấu hình bài học:

- Topic: `logistic_regression`.
- Level: Beginner (`1`).
- Quiz score: `3/5`.
- Thời lượng: `60` phút.
- Số bài tập yêu cầu: `3`.
- Dataset seed: `42`.

## 2. Cấu hình mô hình

| Session | Worker tạo nội dung | Judge chấm notebook |
|---|---|---|
| Groq | `openai/gpt-oss-120b` | `qwen/qwen3.6-27b` |
| DeepSeek | `deepseek-v4-flash` | `deepseek-v4-pro` |

Vì hai session dùng Judge khác nhau, điểm LLM chỉ có giá trị tham khảo. Không nên diễn giải chênh lệch điểm như một phép so sánh tuyệt đối giữa hai Worker.

## 3. Kết quả định lượng

| Chỉ số | Groq | DeepSeek |
|---|---:|---:|
| Decision | PASS | PASS |
| Số attempt | 2/2 | 1/2 |
| Chi phí ước tính | $0.0128 | $0.0285 |
| Thời gian quan sát | khoảng 108 giây | khoảng 55 giây |
| Executability | 5.0 | 5.0 |
| Groundedness | 5.0 | 4.0 |
| Difficulty-fit | 5.0 | 4.0 |
| Pedagogical-order | 4.0 | 4.0 |
| Điểm trung bình | 4.75 | 4.25 |
| Hard rules vòng PASS | 8/8 | 8/8 |
| Số module | 4 | 5 |
| Bài tập được lập kế hoạch | 3 | 4 |
| Tổng cell | 25 | 31 |
| Markdown cell | 12 | 15 |
| Code cell | 13 | 16 |
| Ký tự Markdown | 2,248 | 7,806 |
| Ký tự code | 6,032 | 10,058 |
| Cell TODO | 3 | 4 |
| Cell assert | 3 | 4 |
| Lời gọi biểu đồ | 4 | 4 |

DeepSeek Generator dùng `4,288` input token và `6,227` output token trong lần sinh thành công. Số token chi tiết tương ứng của Groq không được lưu trong artifact nên không đưa vào so sánh trực tiếp.

## 4. Khác biệt quan sát được

### Groq

- Sinh notebook ngắn gọn hơn và bám đúng yêu cầu 3 bài tập.
- Vòng đầu chỉ đạt 5/8 hard rules; hệ thống cần feedback và attempt thứ hai để đạt 8/8.
- Chi phí thấp hơn khoảng 2.2 lần so với session DeepSeek.
- Nội dung lý thuyết ngắn hơn đáng kể: số ký tự Markdown chỉ bằng khoảng 29% notebook DeepSeek.
- Judge ghi nhận thứ tự sư phạm còn điểm cần cải thiện, cụ thể phần tiền xử lý xuất hiện sau khi mô hình đã được sử dụng.

### DeepSeek

- Đạt 8/8 hard rules và PASS ngay attempt đầu tiên.
- Sinh notebook chi tiết hơn: thêm module về common pitfalls, data leakage và decision-threshold tuning.
- File notebook gần gấp đôi kích thước của Groq; phần Markdown dài hơn khoảng 3.5 lần.
- Sinh 4 bài tập dù profile yêu cầu 3. Hard rule hiện chỉ kiểm tra có TODO/assert, chưa kiểm tra chính xác số bài tập theo constraints.
- Chi phí cao hơn, nhưng vẫn thấp hơn nhiều so với trần `$0.30/notebook`.
- Judge có dấu hiệu xem cell TODO cố ý để học viên hoàn thành là lỗi nội dung; đây có thể là false positive cần hiệu chỉnh prompt/rubric.

## 5. Quan sát về Groundedness

Cả hai LearningPath đều có:

- `source_ids` rỗng.
- `theory_context` rỗng.
- Notebook không có URL nguồn.

Do đó, điểm Groundedness `5.0` của Groq và `4.0` của DeepSeek chưa chứng minh notebook thực sự bám ResearchBundle. Trong hai run này, tính năng RAG theory chunks chưa hoạt động vì môi trường thiếu `sentence-transformers`.

## 6. Cập nhật sau pilot: khóa đúng dataset

Prompt Notebook Generator hiện đã bổ sung các ràng buộc nhằm ngăn notebook sử dụng dataset khác với dataset được hệ thống chọn:

- Cell giới thiệu dữ liệu phải ghi đúng thông tin trong `dataset_info`.
- Dataset placeholder phải xuất hiện đúng một lần.
- Model không được tự gọi `read_csv`, `sklearn.datasets`, tải URL hoặc tạo dữ liệu giả.
- Bài toán supervised chỉ được dùng `X_train`, `X_test`, `y_train`, `y_test` do Dataset Injector tạo.
- Prompt cấm bịa dataset và yêu cầu mỗi exercise bám đúng `planned_exercise` trong LearningPath.

Đây là bản vá áp dụng cho các lần sinh sau. Hai artifact trong pilot được giữ nguyên để bảo toàn kết quả thực nghiệm lịch sử. Bản vá đã được kiểm tra ở mức nội dung prompt nhưng vẫn cần một run Decision Tree mới để xác nhận notebook Red Wine không còn nhắc hoặc yêu cầu dùng Iris.

Prompt giúp giảm lỗi sinh sai dataset nhưng chưa phải cổng deterministic. Nếu lỗi này vẫn tái diễn, nên bổ sung hard rule `dataset_consistency` để phát hiện tên dataset không hợp lệ trước khi PASS.

## 7. Kết luận sơ bộ

- DeepSeek phù hợp khi ưu tiên nội dung dày, nhiều giải thích và khả năng đạt hard rules ngay lần đầu.
- Groq phù hợp khi ưu tiên chi phí thấp, notebook ngắn gọn và bám số bài tập yêu cầu.
- Chưa thể chọn mô hình chiến thắng chỉ từ hai notebook này vì cỡ mẫu quá nhỏ, quá trình sinh có tính ngẫu nhiên và Judge không giống nhau.
- Bước tiếp theo nên chạy pilot 6 tổ hợp topic × level với seed và cấu hình cố định, sau đó mới chọn mô hình cho benchmark 20 case.

## 8. Artifacts

Thư mục `notebookforge/outputs/` bị Git bỏ qua mặc định. Khi công bố pilot lên GitHub, chỉ force-add sáu artifact được liệt kê dưới đây; không add toàn bộ outputs của các session khác.

### Groq — `cli-4d40f588`

- [Notebook](../outputs/cli-4d40f588/notebook.ipynb)
- [LearningPath](../outputs/cli-4d40f588/path.json)
- [Quality report](../outputs/cli-4d40f588/quality_report.md)

Lưu ý: quality report cũ ghi `Vòng tốt nhất: #1` do tie-break trước đây chỉ so sánh điểm LLM. Notebook sản phẩm thực tế trùng với attempt 2, là vòng đạt 8/8 hard rules. Lỗi tie-break đã được sửa local sau run này.

### DeepSeek — `cli-deepseek-20260830-v2`

- [Notebook](../outputs/cli-deepseek-20260830-v2/notebook.ipynb)
- [LearningPath](../outputs/cli-deepseek-20260830-v2/path.json)
- [Quality report](../outputs/cli-deepseek-20260830-v2/quality_report.md)
