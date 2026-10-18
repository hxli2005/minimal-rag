"""Official-format export of the frozen pure-BM25 Enterprise 500 answers (no new generation)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/overnight-20261012"))
from export_official_history import main  # noqa: E402

main(folders=(("dev", ROOT / "runs/code-tools-20261017/A"), ("test", ROOT / "runs/search-description-20261016/enterprise-test")),
     output=ROOT / "runs/frozen-20261017/enterprise-500-answers.jsonl",
     evidence_path=ROOT / "evidence/frozen-enterprise-export-20261017.json",
     meta={"kind": "frozen_official_format_export", "date": "2026-10-17", "variant": "frozen-20261017",
           "generation_model": "gpt-6.1-sol high (existing answers, no new generation)"})
