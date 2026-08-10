"""
crew_setup.py - NotebookForge (Nhóm 19)
=======================================
Chủ sở hữu: HOÀNG. Khai báo Agent + Task của CrewAI ở một chỗ.

ĐỌC KỸ CHỖ NÀY - dễ hiểu nhầm:
main.py KHÔNG dùng CrewAI. Vòng lặp retry của mình tự viết bằng Python thuần,
vì cần kiểm soát chính xác 3 điều kiện dừng và phải đếm được tiền từng attempt -
Crew chạy tuần tự thì không chen được cost guard vào giữa.

File này tồn tại cho 2 mục đích:
  1. Ai muốn dùng CrewAI bên trong module của mình (vd Trí cho Research Agent
     gọi tool kb_reader) thì lấy sẵn Agent đã cấu hình đúng model + prompt.
  2. Có bản demo "Crew chạy tuần tự" để so sánh trong báo cáo cuối kỳ.

Persona (role/goal/backstory) để ở đây, còn nội dung việc cụ thể vẫn nằm trong
prompts/*.txt của từng người - đúng nguyên tắc "prompt là code, quản lý bằng git".

Chưa cài crewai cũng import file này được; chỉ khi gọi build_agent() mới báo lỗi.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import MODEL, MODEL_JUDGE, PROVIDER  # noqa: E402

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# CrewAI đi qua litellm nên model phải có tiền tố nhà cung cấp: "groq/llama-3.3-70b-versatile"
CREW_MODEL = os.getenv("NOTEBOOKFORGE_CREW_MODEL", f"{PROVIDER}/{MODEL}")
CREW_MODEL_JUDGE = os.getenv("NOTEBOOKFORGE_CREW_MODEL_JUDGE", f"{PROVIDER}/{MODEL_JUDGE}")


# ---------------------------------------------------------------------------
# Persona của 4 agent
# ---------------------------------------------------------------------------

AGENT_SPECS: dict[str, dict[str, Any]] = {
    "research": {
        "owner": "TRÍ",
        "prompt_file": "research.txt",
        "role": "Chuyên viên nghiên cứu tài liệu Machine Learning",
        "goal": (
            "Với một topic ML, tìm ra các khái niệm cốt lõi và gắn MỖI khái niệm "
            "với một nguồn cụ thể trong kb/. Khái niệm nào không có nguồn thì khai "
            "báo thẳng vào unresolved_concepts, tuyệt đối không bịa."
        ),
        "backstory": (
            "Bạn là trợ giảng đã đọc nát giáo trình. Bạn thà nói 'phần này tài liệu "
            "chưa nhắc tới' còn hơn viết một câu nghe hay mà không có nguồn - vì "
            "notebook sinh ra từ nội dung bịa sẽ dạy sai cho người học."
        ),
    },
    "curriculum": {
        "owner": "HỢP",
        "prompt_file": "curriculum.txt",
        "role": "Người thiết kế lộ trình học",
        "goal": (
            "Biến ResearchBundle thành LearningPath vừa sức học viên: đúng level_final, "
            "tổng thời lượng và số bài tập bám sát constraints, khái niệm đi từ dễ đến khó."
        ),
        "backstory": (
            "Bạn dạy ML nhiều năm và biết rõ: nhồi 5 khái niệm vào 60 phút thì học "
            "viên bỏ giữa chừng. Bạn luôn cắt bớt để phần còn lại được học tử tế."
        ),
    },
    "notebook_gen": {
        "owner": "HỢP",
        "prompt_file": "notebook_gen.txt",
        "role": "Người viết notebook bài tập",
        "goal": (
            "Từ LearningPath sinh ra file .ipynb chạy được: mỗi bài tập có markdown "
            "hướng dẫn, có chỗ TODO cho học viên, có assert để tự kiểm tra, có tách "
            "train/test, và KHÔNG lộ đáp án sẵn."
        ),
        "backstory": (
            "Bạn tin rằng học viên chỉ nhớ thứ họ tự gõ ra. Notebook của bạn luôn "
            "chừa chỗ trống đúng chỗ quan trọng nhất. Nếu vòng trước có feedback, "
            "bạn sửa đúng cái được chỉ ra, không viết lại từ đầu."
        ),
    },
    "verifier": {
        "owner": "HUY",
        "prompt_file": "verifier.txt",
        "role": "Giám khảo chất lượng notebook",
        "goal": (
            "Chấm notebook trên 4 tiêu chí (executability, groundedness, difficulty_fit, "
            "pedagogical_order) thang 1-5, và viết feedback đủ cụ thể để vòng sau sửa được."
        ),
        "backstory": (
            "Bạn chấm thẳng tay. Điểm 62% kèm phân tích vì sao fail có giá trị hơn 95% "
            "không giải thích được. Feedback của bạn luôn chỉ đúng cell nào, sai cái gì, "
            "sửa thế nào - không nói chung chung kiểu 'cần cải thiện'."
        ),
    },
}


def load_prompt(name: str) -> str:
    """Đọc prompts/<name>.txt. Chưa có file thì trả chuỗi rỗng (không làm sập)."""
    spec = AGENT_SPECS.get(name)
    filename = spec["prompt_file"] if spec else f"{name}.txt"
    path = PROMPTS_DIR / filename
    if not path.is_file():
        print(f"[crew] chưa có {path} - dùng prompt rỗng", file=sys.stderr)
        return ""
    return path.read_text(encoding="utf-8")


def crew_available() -> bool:
    """Có cài crewai hay chưa - dùng để rẽ nhánh mà không phải try/except."""
    try:
        import crewai  # noqa: F401
    except ImportError:
        return False
    return True


def _require_crew():
    try:
        import crewai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Chưa cài CrewAI. Chạy: pip install crewai\n"
            "(main.py không cần CrewAI - chỉ file này cần.)"
        ) from exc
    return crewai


def build_llm(model: str = CREW_MODEL, temperature: float = 0.3):
    """LLM cho agent. Mặc định là model worker; verifier tự lấy model judge."""
    crewai = _require_crew()
    return crewai.LLM(model=model, temperature=temperature)


def build_agent(name: str, *, tools: list | None = None, verbose: bool = True):
    """Tạo crewai.Agent theo persona đã khai ở AGENT_SPECS.

        agent = build_agent("research", tools=[kb_reader_tool])

    Riêng verifier chạy model KHÁC worker - đề cương mục 2.2, giảm self-bias.
    """
    if name not in AGENT_SPECS:
        raise KeyError(f"Không có agent '{name}'. Chọn: {list(AGENT_SPECS)}")

    crewai = _require_crew()
    spec = AGENT_SPECS[name]
    model = CREW_MODEL_JUDGE if name == "verifier" else CREW_MODEL
    return crewai.Agent(
        role=spec["role"],
        goal=spec["goal"],
        backstory=spec["backstory"],
        llm=build_llm(model),
        tools=tools or [],
        verbose=verbose,
        allow_delegation=False,  # 4 agent chạy theo pipeline cố định, không tự gọi nhau
    )


def build_task(name: str, description: str, expected_output: str, agent=None):
    """Tạo crewai.Task. description nên nối thêm nội dung prompts/<name>.txt."""
    crewai = _require_crew()
    return crewai.Task(
        description=description,
        expected_output=expected_output,
        agent=agent or build_agent(name),
    )


def build_crew(tasks: list, verbose: bool = True):
    """Gom task thành Crew chạy tuần tự."""
    crewai = _require_crew()
    return crewai.Crew(
        agents=[t.agent for t in tasks],
        tasks=tasks,
        process=crewai.Process.sequential,
        verbose=verbose,
    )


__all__ = [
    "AGENT_SPECS",
    "CREW_MODEL",
    "CREW_MODEL_JUDGE",
    "PROMPTS_DIR",
    "load_prompt",
    "crew_available",
    "build_llm",
    "build_agent",
    "build_task",
    "build_crew",
]


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    print(f"crewai đã cài  : {crew_available()}")
    print(f"model worker   : {CREW_MODEL}")
    print(f"model judge    : {CREW_MODEL_JUDGE}")
    print(f"thư mục prompt : {PROMPTS_DIR}")
    print()
    for key, spec in AGENT_SPECS.items():
        has_prompt = (PROMPTS_DIR / spec["prompt_file"]).is_file()
        mark = "có" if has_prompt else "CHƯA CÓ"
        model = CREW_MODEL_JUDGE if key == "verifier" else CREW_MODEL
        print(f"  {key:<13} ({spec['owner']:<5}) prompt {spec['prompt_file']:<18} {mark:<8} {model}")
