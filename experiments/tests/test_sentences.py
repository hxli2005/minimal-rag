import pytest

from minimal_rag_experiments.sentences import split_sentences


@pytest.mark.parametrize("text,expected", [
    ("Dr. Chen upgraded v2.3.1. Restart at 3.5 seconds.", ["Dr. Chen upgraded v2.3.1.", "Restart at 3.5 seconds."]),
    ("See https://docs.example.com/a?q=2.3 and team@example.com. Then retry.",
     ["See https://docs.example.com/a?q=2.3 and team@example.com.", "Then retry."]),
    ("Use e.g. DNS checks. The U.S. team owns it.", ["Use e.g. DNS checks.", "The U.S. team owns it."]),
    ('He said “Stop.” Then restart.', ['He said “Stop.”', 'Then restart.']),
    ("先检查版本1.2。再重试！成功了吗？", ["先检查版本1.2。", "再重试！", "成功了吗？"]),
    ("  Heading\r\n\r\n  - retry once\r\n1. Keep the marker. Done.\n",
     ["Heading", "- retry once", "1. Keep the marker.", "Done."]),
    ("| version | result |\n| v1.2 | OK. Ready. |\n", ["| version | result |", "| v1.2 | OK. Ready. |"]),
    ("emoji 🧭 works.\n另一行", ["emoji 🧭 works.", "另一行"]),
    ("  \n\t\r\n", []),
])
def test_source_spans_and_boundaries(text, expected):
    units = split_sentences(text)
    assert [u.text for u in units] == expected
    assert all(text[u.offset:u.end_offset] == u.text for u in units)
    assert all(a.end_offset <= b.offset for a, b in zip(units, units[1:]))
    # Splitting must not lose any non-whitespace source character.
    assert "".join(c for c in text if not c.isspace()) == "".join(c for u in units for c in u.text if not c.isspace())


def test_long_lines_are_not_silently_truncated_and_types_are_honest():
    text = "Unpunctuated log " * 2000 + "\nDone."
    units = split_sentences(text)
    assert len(units[0].text) > 2000
    assert [u.unit_type for u in units] == ["line", "sentence"]
