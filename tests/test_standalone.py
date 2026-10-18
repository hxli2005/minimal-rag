import json
from pathlib import Path

import pytest

from minimal_rag.cli import main
from minimal_rag.applications.qa import cli


def test_offline_demo_and_user_corpus_work_without_textbook(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "offline_scripted_demo"
    assert report["result"]["status"] == "finished"
    assert {e["name"] for e in report["events"] if e["role"] == "tool"} == {"search", "read"}
    source = tmp_path / "documents.jsonl"
    source.write_text(json.dumps({"id": "a", "title": "Troubleshooting", "text": "Connection timeout: check the endpoint.", "source": "test://a"}) + "\n")
    index = tmp_path / "data/corpus.sqlite"
    assert main(["index", str(source), "--output", str(index)]) == 0
    capsys.readouterr()
    assert main(["search", "--index", str(index), "--query", "timeout"]) == 0
    assert json.loads(capsys.readouterr().out)["hits"][0]["document_id"] == "a"


def test_ask_requires_opt_in_before_accessing_corpus_or_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli(["--index", "absent.sqlite", "--question", "test"])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
