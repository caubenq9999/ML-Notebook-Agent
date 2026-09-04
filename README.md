# NotebookForge

NotebookForge là hệ thống multi-agent tạo Jupyter Notebook Machine Learning theo hồ sơ người học. Pipeline nghiên cứu kiến thức, thiết kế lộ trình, sinh notebook, thực thi và kiểm định chất lượng trước khi trả kết quả.

## Kiến trúc

```text
LearnerProfile
      ↓
Research Agent → ResearchBundle
      ↓
Curriculum Agent → LearningPath
      ↓
Notebook Generator → .ipynb
      ↓
Executor → ExcRes
      ↓
Verifier → Hard Rules + LLM Judge
      ↓
PASS / RETRY / FAIL
```

Topic hỗ trợ: `logistic_regression`, `decision_tree`, `kmeans`.

Trình độ hỗ trợ:

- `1`: Beginner
- `2`: Intermediate

Cấu hình hiện tại cho phép tối đa **2 attempt**, ngưỡng PASS trung bình **3.5/5** và cost cap **$0.30/session**.

## Cấu trúc thư mục

```text
ML-Notebook-Agent/
├── requirements.txt
└── notebookforge/
    ├── main.py                  # Orchestrator và CLI
    ├── api.py                   # FastAPI backend
    ├── llm_client.py            # LLM, retry, validation, cost tracking
    ├── executor.py              # Chạy notebook bằng nbclient
    ├── schemas.py               # Contract dùng chung
    ├── agents/                  # Research, Curriculum, Generator, Verifier
    ├── prompts/                 # Prompt của các agent
    ├── kb/                      # Knowledge Base của ba topic
    ├── datasets/                # Ba dataset CSV
    ├── tools/                   # KB reader và dataset injector
    ├── ui/                      # Streamlit và report adapter
    ├── eval/                    # Golden set, benchmark, rule checker
    └── tests/                   # Test suite
```

## Yêu cầu

- Python 3.10 trở lên; khuyến nghị Python 3.11.
- API key nếu chạy pipeline thật.
- Internet khi gọi LLM. Executor đặt biến môi trường offline/proxy để hạn chế tải dữ liệu, nhưng đây không phải security sandbox hoàn chỉnh.

CrewAI là tùy chọn; pipeline chính trong `main.py` không cần CrewAI.

## 1. Cài đặt

Mở PowerShell tại thư mục gốc repo:

```powershell
cd D:\MLIoT\final_project\main\ML-Notebook-Agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`sentence-transformers` phục vụ semantic chunking/RAG. Lần chạy Research đầu tiên sẽ tải model `paraphrase-multilingual-MiniLM-L12-v2` về cache local; nếu dependency hoặc model này không khả dụng, pipeline vẫn chạy nhưng `ResearchBundle.theory_chunks` sẽ rỗng và Notebook Generator không nhận được phần lý thuyết RAG.

Nếu PowerShell chặn script activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Đăng ký kernel cho Executor:

```powershell
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

Trên macOS/Linux, activate bằng `source .venv/bin/activate`.

## 2. Cấu hình LLM

Copy `.env.example` thành `.env` tại thư mục gốc rồi điền key. `.env` đã được `.gitignore`; tuyệt đối không commit API key.

```powershell
Copy-Item .env.example .env
```

### Groq — cấu hình mặc định

```dotenv
GROQ_API_KEY=your_groq_api_key
NOTEBOOKFORGE_PROVIDER="groq"
NOTEBOOKFORGE_MODEL="openai/gpt-oss-120b"
NOTEBOOKFORGE_MODEL_JUDGE="qwen/qwen3.6-27b"
```

Worker và Judge nên dùng model khác nhau để giảm self-bias.

### Gemini

```dotenv
GEMINI_API_KEY=your_gemini_api_key
NOTEBOOKFORGE_PROVIDER=gemini
NOTEBOOKFORGE_MODEL=your-supported-gemini-model
NOTEBOOKFORGE_MODEL_JUDGE=your-supported-judge-model
```

### OpenRouter

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
NOTEBOOKFORGE_PROVIDER=openrouter
NOTEBOOKFORGE_MODEL=provider/model-id
NOTEBOOKFORGE_MODEL_JUDGE=another-provider/model-id
```

### DeepSeek official API — DeepSeek V4 Flash

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
NOTEBOOKFORGE_PROVIDER=deepseek
NOTEBOOKFORGE_MODEL=deepseek-v4-flash
NOTEBOOKFORGE_MODEL_JUDGE=deepseek-v4-pro
NOTEBOOKFORGE_DEEPSEEK_THINKING=disabled
NOTEBOOKFORGE_NOTEBOOK_MAX_TOKENS=10000
```

Endpoint mặc định là `https://api.deepseek.com`. Flash làm Worker và Pro làm Judge để hai vai trò không trùng model. `disabled` phù hợp benchmark/pipeline JSON vì giữ temperature có hiệu lực và giảm reasoning token; đổi thành `enabled` để bật thinking mode, khi đó `NOTEBOOKFORGE_NOTEBOOK_REASONING_EFFORT` nhận `low`, `medium`, `high`, `xhigh` hoặc `max`.

Giữ `NOTEBOOKFORGE_NOTEBOOK_MAX_TOKENS=10000` khi sinh notebook thật bằng DeepSeek. Pilot cho thấy mức 5.000 token có thể cắt JSON giữa chừng; mức 10.000 chỉ là trần output, model dừng sớm nếu đã sinh xong.

CostTracker dùng giá peak bảo thủ và đọc `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` từ API. Giá off-peak thực tế bằng một nửa giá peak, nên chi phí dashboard có thể thấp hơn estimate.

### AgentRouter — chỉ dùng khi custom client đã được cấp quyền

```dotenv
AGENTROUTER_API_KEY=your_agentrouter_api_key
AGENTROUTER_BASE_URL=https://agentrouter.org/v1
NOTEBOOKFORGE_PROVIDER=agentrouter
NOTEBOOKFORGE_MODEL=deepseek-v4-flash
NOTEBOOKFORGE_MODEL_JUDGE=glm-5.1
```

Lấy API Token tại `https://agentrouter.org/console/token`; không dùng System Access Token trong phần cài đặt tài khoản. `AGENTROUTER_BASE_URL` có thể bỏ vì code đã dùng endpoint mặc định ở trên. Model ID phải lấy đúng từ trang model/pricing của tài khoản.

**Giới hạn quan trọng:** AgentRouter hiện chặn custom Python/OpenAI SDK bằng lỗi `unauthorized_client` và yêu cầu dùng các harness chính thức như Claude Code, Codex, Cline hoặc OpenCode. Vì NotebookForge là một backend Python tự gọi LLM, cấu hình trên chỉ hoạt động nếu AgentRouter support cấp quyền/whitelist cho custom client. Không giả mạo header của harness để vượt cơ chế này.

Không nên đặt `NOTEBOOKFORGE_MODEL_JUDGE=deepseek-v4-flash`: Worker và Judge trùng model sẽ làm benchmark có nguy cơ self-bias. Ví dụ trên chỉ dùng được khi tài khoản có `glm-5.1`; nếu không, chọn một model Judge khác có trong cùng catalog AgentRouter.

Giá peak `deepseek-v4-flash` trong CostTracker đang được khai báo là `$0.014/M` input cache-hit, `$0.44/M` input cache-miss và `$1.32/M` output. Hãy đối chiếu dashboard trước khi dùng số cost trong báo cáo vì provider có thể áp dụng giá off-peak.

Các provider có thể dùng trực tiếp là `groq`, `gemini`, `openrouter` và `deepseek`. Adapter `agentrouter` đã có trong code nhưng bị chính sách gateway chặn với custom client nếu chưa được whitelist.

Kiểm tra cấu hình mà không gọi LLM:

```powershell
python notebookforge/llm_client.py
```

Sau khi AgentRouter xác nhận đã cấp quyền custom client, có thể kiểm tra model bằng một request rất nhỏ:

```powershell
python notebookforge/llm_client.py --smoke
```

Smoke test chỉ yêu cầu model trả lời `OK`, không sinh notebook và tiêu rất ít token.

### Biến tùy chọn

| Biến | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `NOTEBOOKFORGE_MAX_TOKENS` | `16000` | Output token mặc định của LLM client |
| `NOTEBOOKFORGE_TEMPERATURE` | `0.3` | Temperature mặc định |
| `NOTEBOOKFORGE_LLM_RETRIES` | `3` | Số lần thử lại API/schema |
| `NOTEBOOKFORGE_DEEPSEEK_THINKING` | `disabled` | Bật/tắt thinking mode cho DeepSeek V4 |
| `AGENTROUTER_BASE_URL` | `https://agentrouter.org/v1` | Endpoint OpenAI-compatible của AgentRouter |
| `NOTEBOOKFORGE_CELL_TIMEOUT` | `120` | Timeout mỗi notebook cell, tính bằng giây |
| `NOTEBOOKFORGE_NOTEBOOK_MAX_TOKENS` | `5000` | Token tối đa của Notebook Generator; cấu hình DeepSeek khuyến nghị `10000` |
| `NOTEBOOKFORGE_NOTEBOOK_TPM_BUDGET` | `7600` | Ngân sách TPM của Generator |
| `NOTEBOOKFORGE_NOTEBOOK_MIN_OUTPUT_TOKENS` | `3000` | Output tối thiểu Generator cố gắng dành |
| `NOTEBOOKFORGE_NOTEBOOK_FEEDBACK_MAX_CHARS` | `600` | Feedback tối đa cho attempt sau |
| `NOTEBOOKFORGE_NOTEBOOK_REASONING_EFFORT` | `low` | Reasoning effort của Generator |
| `NOTEBOOKFORGE_UI_ALLOW_MOCK_FALLBACK` | `0` | Cho UI fallback sang dữ liệu mock |

## 3. Smoke test không tốn API

Chạy từ thư mục gốc repo:

```powershell
python notebookforge/main.py `
  --topic logistic_regression `
  --level 1 `
  --quiz-score 3 `
  --duration 60 `
  --exercises 3 `
  --session-id smoke-test `
  --seed 42 `
  --mock
```

`--mock` không kèm tên sẽ mock toàn bộ bốn agent. Executor vẫn là bản thật và vẫn mở kernel để chạy notebook.

> Lưu ý: notebook mock hiện có cell `TODO` cố ý chưa hoàn thành. Lệnh này dùng để kiểm tra wiring, Executor và việc tạo artifact; kết quả cuối có thể là `FAIL_MAX_RETRY` thay vì `PASS`.

Chỉ mock một hoặc nhiều bước:

```powershell
python notebookforge/main.py --mock verifier
python notebookforge/main.py --mock research curriculum
```

Tên hợp lệ: `research`, `curriculum`, `notebook_gen`, `verifier`.

## 4. Chạy pipeline thật bằng CLI

Sau khi cấu hình `.env`:

```powershell
python notebookforge/main.py `
  --topic logistic_regression `
  --level 1 `
  --quiz-score 3 `
  --duration 60 `
  --exercises 3 `
  --seed 42
```

Kết quả nằm tại:

```text
notebookforge/outputs/<session_id>/
├── notebook.ipynb
├── path.json
├── quality_report.md
└── result.json
```

Ba artifact sản phẩm là `notebook.ipynb`, `path.json`, `quality_report.md`; `result.json` phục vụ API và debug.

## 5. Chạy giao diện Web

Cần hai terminal.

### Terminal 1 — FastAPI

```powershell
cd D:\MLIoT\final_project\main\ML-Notebook-Agent\notebookforge
..\.venv\Scripts\Activate.ps1
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

- Swagger: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

### Terminal 2 — Streamlit

```powershell
cd D:\MLIoT\final_project\main\ML-Notebook-Agent
.\.venv\Scripts\Activate.ps1
streamlit run notebookforge/ui/streamlit_app.py
```

UI thường mở tại <http://localhost:8501> và gọi FastAPI tại `http://localhost:8000`.

### Chỉ xem UI bằng mock fallback

```powershell
$env:NOTEBOOKFORGE_UI_ALLOW_MOCK_FALLBACK="1"
streamlit run notebookforge/ui/streamlit_app.py
```

Fallback chỉ phục vụ kiểm tra UI/demo dự phòng, không phải kết quả benchmark.

### Deploy miễn phí trên Streamlit Community Cloud

Bản cloud dùng entrypoint riêng và gọi `main.generate()` trực tiếp trong
cùng process, do đó không cần khởi chạy FastAPI hay port 8000:

```text
notebookforge/ui/streamlit_cloud_app.py
```

Kiểm tra bản cloud trên máy local:

```powershell
$env:APP_PASSWORD="demo-local"
streamlit run notebookforge/ui/streamlit_cloud_app.py
```

Các bước deploy:

1. Push code lên GitHub, nhưng không push `.env` hoặc
   `.streamlit/secrets.toml`.
2. Mở <https://share.streamlit.io>, chọn **Create app** và kết nối repo.
3. Chọn branch cần deploy và Main file path là
   `notebookforge/ui/streamlit_cloud_app.py`.
4. Trong **Advanced settings**, chọn Python 3.11.
5. Mở **App settings > Secrets**, copy mẫu từ
   `.streamlit/secrets.example.toml`, sau đó thay `APP_PASSWORD` và API key
   bằng giá trị thật.
6. Deploy và chờ `requirements.txt` cài xong. Lần Research đầu có thể
   chậm hơn do tải embedding model.

Cloud entrypoint mặc định yêu cầu `APP_PASSWORD` để tránh người lạ
dùng hết API credit. Chỉ khi chủ động đặt secret sau thì app mới cho
phép truy cập công khai không cần mật khẩu:

```toml
NOTEBOOKFORGE_ALLOW_PUBLIC = true
```

Mỗi cloud instance chỉ chạy một job sinh notebook tại một thời điểm để
giảm nguy cơ hết RAM. Artifact trên Community Cloud là tạm thời; người
dùng nên tải notebook xuống ngay sau khi run hoàn tất.

## 6. Gọi API thủ công

Sau khi FastAPI chạy, tạo job mock để kiểm tra contract và cơ chế polling:

```powershell
$body = @{
  topic = "logistic_regression"
  level = 1
  quiz_score = 3
  duration_minutes = 60
  num_exercises = 3
  use_mock = $true
} | ConvertTo-Json

$job = Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/generate" `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod "http://127.0.0.1:8000/report/$($job.session_id)"
```

Job mock vẫn đi qua Executor thật; do notebook mock có `TODO`, trạng thái cuối có thể là `error`. Muốn demo một lượt `PASS`, cần dùng pipeline thật hoặc cập nhật mock notebook thành bản chạy sạch.

| Method | Endpoint | Chức năng |
| :--- | :--- | :--- |
| `POST` | `/generate` | Tạo job nền, trả `session_id` |
| `GET` | `/report/{session_id}` | Poll trạng thái và lấy kết quả |
| `GET` | `/artifacts/{session_id}/{artifact_name}` | Tải notebook/path/report |
| `GET` | `/health` | Kiểm tra server |

API lưu trạng thái session trong RAM. Restart server sẽ mất danh sách job, nhưng artifact đã ghi trên đĩa vẫn còn.

## 7. Chạy test

Test suite không gọi LLM và không yêu cầu pytest:

```powershell
cd D:\MLIoT\final_project\main\ML-Notebook-Agent\notebookforge
..\.venv\Scripts\Activate.ps1
python tests/run_all.py
```

Test Executor mở kernel thật nên có thể lâu hơn các nhóm test khác.

Chạy riêng Verifier/Evaluation:

```powershell
python -m unittest eval.test_sprint22 -v
```

Kiểm tra contract hiện tại:

```powershell
python tests/contract_check.py --here
```

## 8. Evaluation và Benchmark

Chạy các lệnh sau trong thư mục `notebookforge`.

Preflight, không gọi pipeline:

```powershell
python -m eval.harness --preflight
```

Chạy một case:

```powershell
python -m eval.harness `
  --case GS-001 `
  --checkpoint eval/results/checkpoint.json `
  --output eval/results/benchmark.md `
  --json-output eval/results/benchmark.json
```

Chạy 20 case và resume case đã hoàn thành:

```powershell
python -m eval.harness `
  --checkpoint eval/results/checkpoint.json `
  --resume `
  --output eval/results/benchmark.md `
  --json-output eval/results/benchmark.json
```

Kiểm tra hard rules của một notebook:

```powershell
python -m eval.check_rules outputs/<session_id>/notebook.ipynb `
  --topic logistic_regression `
  --level 1 `
  --detail
```

Benchmark thật gọi LLM và có thể tốn chi phí. Luôn chạy preflight và một case trước khi chạy toàn bộ golden set.

## 9. Điều kiện PASS

Notebook chỉ PASS khi đồng thời:

1. Executor chạy thành công.
2. Tất cả hard rules đạt.
3. Trung bình bốn điểm LLM đạt ít nhất `3.5/5`.
4. Verifier trả `PASS`.

Bốn tiêu chí LLM: Executability, Groundedness, Difficulty-fit và Pedagogical-order.

Tám hard rules:

- `has_instructions`: Beginner có ít nhất 8 markdown cell; Intermediate có ít nhất 10.
- `has_todo`
- `has_assert`
- `no_hardcoded_answers`
- `has_train_test_split`
- `has_visualization`: có ít nhất 2 lời gọi vẽ thực sự.
- `has_demo_per_module`
- `min_cells_by_level`: Beginner có ít nhất 12 cell; Intermediate có ít nhất 18.

## 10. Troubleshooting

### Thiếu `nbformat` hoặc dependency khác

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `No such kernel named python3`

```powershell
python -m ipykernel install --user --name python3 --display-name "Python 3"
```

### Thiếu API key

Kiểm tra `.env`, provider và model:

```powershell
python notebookforge/llm_client.py
```

### Streamlit không kết nối được FastAPI

1. Kiểm tra FastAPI còn chạy.
2. Mở <http://127.0.0.1:8000/health>.
3. Xác nhận port 8000 không bị ứng dụng khác dùng.
4. Chạy UI sau khi backend sẵn sàng.

### API trả `409 Session đã tồn tại`

Dùng `session_id` mới. API không nhận hai job có cùng session trong một lần chạy server.

### Output bị cắt hoặc JSON không hợp lệ

- Tăng `NOTEBOOKFORGE_MAX_TOKENS` hoặc `NOTEBOOKFORGE_NOTEBOOK_MAX_TOKENS` có kiểm soát.
- Kiểm tra TPM/RPM của provider.
- Theo dõi cost trước khi chạy nhiều case.

## Lưu ý vận hành

- Không commit `.env` hoặc API key.
- `outputs/` và `output_notebooks/` đã được Git bỏ qua.
- CostTracker dùng bảng giá khai báo trong `llm_client.py`; cập nhật bảng giá trước khi đưa cost vào báo cáo chính thức.
- UI đang cố định backend tại `http://localhost:8000`.
- API store nằm trong RAM, phù hợp demo nhưng chưa phù hợp lưu trữ lâu dài.
- Nên chạy mock smoke test, sau đó một case thật, cuối cùng mới chạy benchmark nhiều case.
