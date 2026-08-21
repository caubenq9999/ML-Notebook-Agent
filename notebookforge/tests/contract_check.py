"""
tests/contract_check.py - NotebookForge (Nhóm 19)
=================================================
Chủ sở hữu: HOÀNG. Công cụ review PR: kiểm 1 nhánh có đúng contract không.

    # Kiểm 1 nhánh (tự checkout ra thư mục tạm, không đụng code đang làm dở)
    python tests/contract_check.py feat/research-agent

    # Kiểm tất cả các nhánh
    python tests/contract_check.py --all

    # Kiểm code đang có trong thư mục hiện tại
    python tests/contract_check.py --here

    # Gọi hàm thật (TỐN QUOTA LLM) - mặc định chỉ kiểm tĩnh
    python tests/contract_check.py feat/research-agent --live

Kiểm 6 điều, đúng theo quy tắc trong Hướng dẫn Sprint 2 phần 5:
    1. Chỉ sửa file trong vùng của mình
    2. Không đụng schemas.py / tests/mocks.py (chỉ Hoàng được sửa)
    3. Không commit .env hoặc file rác
    4. Đúng chữ ký hàm đã chốt ở Phần 3
    5. Prompt nằm trong prompts/*.txt, không hardcode trong .py
    6. Không import trực tiếp code của người khác - mọi thứ đi qua schema
    (+ với --live: gọi hàm thật, kiểm output có đúng schema không)

Exit code 0 = qua hết, 1 = có lỗi chặn PR.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent  # .../ML-Notebook-Agent
PKG = "notebookforge"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# ---------------------------------------------------------------------------
# Ai sở hữu cái gì - lấy từ Hướng dẫn Sprint 2 phần 1 + 6 chữ ký ở phần 3
# ---------------------------------------------------------------------------

BRANCHES: dict[str, dict] = {
    "feat/research-agent": {
        "owner": "TRÍ",
        "zone": [
            f"{PKG}/agents/research.py",
            f"{PKG}/tools/kb_reader.py",
            f"{PKG}/kb/",
            f"{PKG}/prompts/research.txt",
        ],
        "module": "agents.research",
        "func": "run_research",
        "params": ["topic"],
        "returns": "ResearchBundle",
        "prompt": f"{PKG}/prompts/research.txt",
    },
    "feat/curriculum-agent": {
        "owner": "HỢP",
        "zone": [
            f"{PKG}/agents/curriculum.py",
            f"{PKG}/agents/notebook_gen.py",
            f"{PKG}/prompts/curriculum.txt",
            f"{PKG}/prompts/notebook_gen.txt",
        ],
        "module": "agents.curriculum",
        "func": "run_curriculum",
        "params": ["bundle", "profile"],
        "returns": "LearningPath",
        "prompt": f"{PKG}/prompts/curriculum.txt",
    },
    "feat/notebook_gen-agent": {
        "owner": "HỢP",
        "zone": [f"{PKG}/agents/notebook_gen.py", f"{PKG}/prompts/notebook_gen.txt"],
        "module": "agents.notebook_gen",
        "func": "run_notebook_gen",
        "params": ["path", "profile", "attempt", "prior_feedback"],
        "returns": "str",
        "prompt": f"{PKG}/prompts/notebook_gen.txt",
    },
    "feat/verifier": {
        "owner": "HUY",
        "zone": [
            f"{PKG}/agents/verifier.py",
            f"{PKG}/eval/",
            f"{PKG}/prompts/verifier.txt",
        ],
        "module": "agents.verifier",
        "func": "run_verifier",
        "params": ["nb_path", "exc", "bundle"],
        "returns": "VerifierReport",
        "prompt": f"{PKG}/prompts/verifier.txt",
    },
    "feat/dataset-injector": {
        "owner": "NAM",
        "zone": [f"{PKG}/tools/dataset_injector.py"],
        "module": "tools.dataset_injector",
        "func": "get_dataset_code",
        "params": ["topic", "seed"],
        "returns": "str",
        "prompt": None,
    },
    "feat/ui": {
        "owner": "NAM",
        "zone": [f"{PKG}/ui/"],
        "module": None,  # Streamlit app, không có hàm contract để gọi
        "func": None,
        "params": [],
        "returns": None,
        "prompt": None,
    },
}

# Chỉ Hoàng được sửa - ai đụng là chặn PR
LOCKED = [f"{PKG}/schemas.py", f"{PKG}/tests/mocks.py", f"{PKG}/main.py",
          f"{PKG}/llm_client.py", f"{PKG}/executor.py", f"{PKG}/api.py",
          f"{PKG}/crew_setup.py"]

# File không bao giờ được commit
FORBIDDEN = [".env", "myenv/", "__pycache__/", ".executed.ipynb"]

# Module thuộc contract giữa các thành viên. Import module của NGƯỜI KHÁC là sai,
# nhưng import module của CHÍNH MÌNH thì bình thường (vd Trí gọi tools.kb_reader
# của chính Trí) - nên phải đối chiếu với vùng sở hữu, không chặn cứng cả danh sách.
PEER_MODULES = {"agents.research", "agents.curriculum", "agents.notebook_gen",
                "agents.verifier", "tools.kb_reader", "tools.dataset_injector"}


# Ngoại lệ hợp lệ: đề cương mục 2.3 vẽ pipeline "[3] NOTEBOOK GEN ... + dataset
# injection (sklearn)", tức notebook_gen LÀ nơi gọi get_dataset_code. main.py không
# gọi hàm này, nên nếu cấm luôn thì không ai gọi được - hàm thành code chết.
ALLOWED_DEPS: dict[str, set[str]] = {
    "agents.notebook_gen": {"tools.dataset_injector"},
}


def _own_modules(spec: dict) -> set[str]:
    """Module nằm trong vùng sở hữu của nhánh, cộng các phụ thuộc hợp lệ."""
    own = set(ALLOWED_DEPS.get(spec.get("module") or "", set()))
    for z in spec["zone"]:
        if z.endswith(".py"):
            mod = z[len(PKG) + 1 : -3].replace("/", ".")
            own.add(mod)
        else:  # cả thư mục, vd notebookforge/eval/
            prefix = z[len(PKG) + 1 :].rstrip("/").replace("/", ".")
            own.update(m for m in PEER_MODULES if m.startswith(prefix + "."))
    return own


class Result:
    def __init__(self) -> None:
        self.blocking: list[str] = []
        self.warnings: list[str] = []
        self.passed: list[str] = []

    def block(self, msg: str) -> None:
        self.blocking.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)


def _git(*args: str, cwd: Path | None = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=str(cwd or REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# 6 phép kiểm
# ---------------------------------------------------------------------------


def check_files(spec: dict, changed: list[str], res: Result) -> None:
    """1 + 2 + 3: file có nằm trong vùng của mình không."""
    if not changed:
        res.warn("Nhánh không có file nào khác main - đã push chưa?")
        return

    outside = []
    for f in changed:
        if any(f == lk for lk in LOCKED):
            res.block(f"Sửa file KHOÁ (chỉ Hoàng được sửa): {f}")
            continue
        if any(bad in f for bad in FORBIDDEN):
            res.block(f"Commit nhầm file không được phép: {f}")
            continue
        if not any(f.startswith(z) for z in spec["zone"]):
            outside.append(f)

    if outside:
        for f in outside:
            # Sai thư mục là lỗi hay gặp nhất: quên tiền tố notebookforge/
            if not f.startswith(f"{PKG}/") and any(
                f == z[len(PKG) + 1 :] or f.startswith(z[len(PKG) + 1 :].rstrip("/"))
                for z in spec["zone"]
            ):
                res.block(
                    f"SAI THƯ MỤC: '{f}' phải nằm trong '{PKG}/{f}'. "
                    f"Ở ngoài thì main.py không import được."
                )
            else:
                res.block(f"File ngoài vùng sở hữu: {f}")
    else:
        res.ok(f"{len(changed)} file, tất cả trong vùng của {spec['owner']}")


def check_prompt(spec: dict, root: Path, res: Result) -> None:
    """5: prompt phải nằm trong prompts/*.txt, không hardcode trong .py."""
    if not spec["prompt"]:
        return
    p = root / spec["prompt"]
    if not p.is_file():
        res.block(f"Thiếu file prompt: {spec['prompt']} (nguyên tắc 'prompt là code')")
        return
    if not p.read_text(encoding="utf-8", errors="replace").strip():
        res.block(f"File prompt rỗng: {spec['prompt']}")
        return
    res.ok(f"Prompt nằm đúng chỗ: {spec['prompt']} ({p.stat().st_size} bytes)")

    # Chuỗi dài trong .py thường là prompt bị hardcode
    py = root / spec["zone"][0]
    if py.is_file() and py.suffix == ".py":
        src = py.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r'"""(.*?)"""', src, re.DOTALL):
            body = m.group(1)
            if len(body) > 400 and re.search(r"\b(Bạn là|You are|JSON|schema)\b", body):
                res.warn(
                    f"{spec['zone'][0]}: có chuỗi {len(body)} ký tự trông giống prompt "
                    f"hardcode - nên chuyển vào {spec['prompt']}"
                )
                break


def check_no_peer_imports(spec: dict, root: Path, res: Result, changed: list[str]) -> None:
    """6: không ai được import trực tiếp module của người khác."""
    mine = _own_modules(spec)
    bad: list[str] = []
    for rel in changed:
        f = root / rel
        if f.suffix != ".py" or not f.is_file():
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            res.block(f"{rel}: file không parse được - {exc}")
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                if n in PEER_MODULES and n not in mine:
                    bad.append(f"{rel} import {n}")
    if bad:
        for b in bad:
            res.block(f"Import thẳng code người khác (phải đi qua schema): {b}")
    else:
        res.ok("Không import trực tiếp code của ai")


def check_signature(spec: dict, root: Path, res: Result, live: bool) -> None:
    """4 (+ --live): đúng chữ ký hàm, và gọi thử xem có trả đúng schema không."""
    if not spec["module"]:
        res.ok("Nhánh UI - không có hàm contract để kiểm")
        return

    pkg_dir = str(root / PKG)
    sys.path.insert(0, pkg_dir)
    for mod in list(sys.modules):
        if mod.startswith(("agents", "tools", "schemas")):
            del sys.modules[mod]
    try:
        module = importlib.import_module(spec["module"])
    except Exception as exc:  # noqa: BLE001
        res.block(f"Không import được {spec['module']}: {type(exc).__name__}: {exc}")
        sys.path.remove(pkg_dir)
        return

    func = getattr(module, spec["func"], None)
    if func is None:
        res.block(f"Thiếu hàm {spec['func']}() trong {spec['module']}")
        sys.path.remove(pkg_dir)
        return

    sig = inspect.signature(func)
    got = list(sig.parameters)
    want = spec["params"]
    required = [n for n, p in sig.parameters.items() if p.default is inspect.Parameter.empty]

    if got[: len(want)] != want:
        # Sai tên hoặc sai thứ tự tham số -> main.py gọi là gãy.
        res.block(
            f"Sai chữ ký: {spec['func']}({', '.join(got)}) "
            f"- đã chốt là {spec['func']}({', '.join(want)})"
        )
    elif [n for n in required if n not in want]:
        # Thừa tham số mà BẮT BUỘC -> main.py gọi thiếu, vẫn gãy.
        # (Tham số trong `want` mà có default thì không sao: main.py vẫn truyền đủ.)
        res.block(
            f"Tham số thừa mà không có giá trị mặc định: "
            f"{[n for n in required if n not in want]}. "
            f"main.py chỉ gọi {spec['func']}({', '.join(want)})."
        )
    elif len(got) > len(want):
        # Thừa nhưng có mặc định -> main.py vẫn gọi được, chỉ nhắc.
        res.ok(f"Chữ ký tương thích: {spec['func']}({', '.join(got)})")
        res.warn(
            f"Có thêm tham số tuỳ chọn {got[len(want):]} - main.py không truyền, "
            f"nên phần đó sẽ luôn chạy ở giá trị mặc định"
        )
    else:
        res.ok(f"Chữ ký đúng: {spec['func']}({', '.join(got)})")

    ret = str(sig.return_annotation).replace("'", "")
    if spec["returns"] and spec["returns"] not in ret:
        res.warn(f"Thiếu/sai type hint kết quả: '{ret}' (chờ '{spec['returns']}')")

    if live:
        _run_live(spec, func, res)

    sys.path.remove(pkg_dir)


def _run_live(spec: dict, func, res: Result) -> None:
    """Gọi hàm thật với mock input, kiểm output có đúng schema không."""
    import schemas
    from tests import mocks

    args_by_func = {
        "run_research": lambda: (mocks.MOCK_PROFILE.topic,),
        "run_curriculum": lambda: (mocks.MOCK_BUNDLE, mocks.MOCK_PROFILE),
        "run_notebook_gen": lambda: (mocks.MOCK_PATH, mocks.MOCK_PROFILE, 1, None),
        "run_verifier": lambda: (
            mocks.write_mock_notebook(), mocks.MOCK_EXC_OK, mocks.MOCK_BUNDLE),
        "get_dataset_code": lambda: (mocks.MOCK_PROFILE.topic, mocks.MOCK_PROFILE.dataset_seed),
    }
    make_args = args_by_func.get(spec["func"])
    if make_args is None:
        return

    print(f"    (đang gọi {spec['func']} thật, có thể mất vài chục giây...)")
    try:
        out = func(*make_args())
    except Exception as exc:  # noqa: BLE001
        res.block(f"Gọi {spec['func']}() thật thì lỗi: {type(exc).__name__}: {exc}")
        return

    want = spec["returns"]
    if want == "str":
        if not isinstance(out, str) or not out.strip():
            res.block(f"{spec['func']}() trả về {type(out).__name__}, chờ str không rỗng")
        else:
            res.ok(f"Chạy thật OK -> str ({len(out)} ký tự)")
        return

    expected = getattr(schemas, want, None)
    if expected and isinstance(out, expected):
        res.ok(f"Chạy thật OK -> {want} hợp lệ")
    else:
        res.block(f"{spec['func']}() trả về {type(out).__name__}, chờ {want}")


# ---------------------------------------------------------------------------
# Chạy
# ---------------------------------------------------------------------------


def review(branch: str, live: bool, here: bool) -> bool:
    spec = BRANCHES.get(branch)
    if spec is None:
        print(f"Không biết nhánh '{branch}'. Đã khai báo: {list(BRANCHES)}")
        return False

    print(f"\n{'=' * 68}")
    print(f"  {branch}   (chủ sở hữu: {spec['owner']})")
    print("=" * 68)

    res = Result()
    tmp: Path | None = None

    if here:
        root = REPO
        changed = [f for f in _git("diff", "--name-only", "origin/main").splitlines() if f]
    else:
        ref = f"origin/{branch}"
        if not _git("rev-parse", "--verify", "--quiet", ref):
            print(f"  Không thấy {ref}. Chạy `git fetch origin` trước.")
            return False
        changed = [f for f in _git("diff", "--name-only", f"origin/main...{ref}").splitlines() if f]
        # Dựng cây từ origin/main RỒI đắp file của nhánh lên - mô phỏng đúng trạng
        # thái SAU KHI MERGE. Nếu checkout thẳng nhánh thì thiếu code chung đã có
        # trên main (llm_client.py) lẫn code người khác, sinh ra lỗi báo oan.
        tmp = Path(tempfile.mkdtemp(prefix="review-"))
        _git("worktree", "add", "--detach", str(tmp), "origin/main")
        for f in changed:
            _git("checkout", ref, "--", f, cwd=tmp)
        root = tmp

    try:
        last = _git("log", "-1", "--format=%h %an: %s",
                    f"origin/{branch}" if not here else "HEAD")
        print(f"  commit cuối: {last}\n")

        check_files(spec, changed, res)
        check_prompt(spec, root, res)
        check_no_peer_imports(spec, root, res, changed)
        check_signature(spec, root, res, live)
    finally:
        if tmp:
            _git("worktree", "remove", "--force", str(tmp))
            shutil.rmtree(tmp, ignore_errors=True)

    for m in res.passed:
        print(f"  [OK]    {m}")
    for m in res.warnings:
        print(f"  [LƯU Ý] {m}")
    for m in res.blocking:
        print(f"  [CHẶN]  {m}")

    verdict = not res.blocking
    print(f"\n  => {'MERGE ĐƯỢC' if verdict else 'CHƯA MERGE ĐƯỢC'}"
          f" ({len(res.blocking)} lỗi chặn, {len(res.warnings)} lưu ý)")
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm 1 nhánh có đúng contract không")
    parser.add_argument("branch", nargs="?", help="vd: feat/research-agent")
    parser.add_argument("--all", action="store_true", help="Kiểm tất cả nhánh đã khai báo")
    parser.add_argument("--here", action="store_true", help="Kiểm thư mục hiện tại")
    parser.add_argument("--live", action="store_true", help="Gọi hàm thật (tốn quota LLM)")
    args = parser.parse_args(argv)

    if args.all:
        results = {b: review(b, args.live, False) for b in BRANCHES}
        print(f"\n{'=' * 68}\n  TỔNG KẾT\n{'=' * 68}")
        for b, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {b}")
        return 0 if all(results.values()) else 1

    if not args.branch and not args.here:
        parser.error("Cần tên nhánh, hoặc --all, hoặc --here")

    return 0 if review(args.branch or "feat/research-agent", args.live, args.here) else 1


if __name__ == "__main__":
    raise SystemExit(main())
