# Third-party reference code and data

The unmodified HotpotQA evaluation script at
`experiments/src/minimal_rag_experiments/_vendor/hotpot_evaluate_v1.py` was extracted with the existing
local benchmark assets. Upstream:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py .
Its original authors retain their rights. The loader replaces only the optional
`ujson` import with Python's standard `json` module at runtime.

Downloaded model weights and benchmark datasets under `models/` and `data/`
are ignored by Git. Their upstream terms apply independently; copying local
assets is not a license grant or a publication of them.

The eleven original document records packaged in
`src/minimal_rag/fixtures/northwind.jsonl` come from the released synthetic-company
corpus `onyx-dot-app/EnterpriseRAG-Bench`, revision
`69916e31c68aa5963c00248fd7f0bc12d04fd235`.
The dataset README declares MIT licensing. Attribution, selection queries and
record hashes are included in `northwind-manifest.json`. Upstream:
https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench .
The accompanying decision replay and Chinese brief are authored examples, not
upstream gold answers or measured model outputs.

The rendered report and trace in `examples/incident-handoff/`, and the screenshot
in `docs/assets/incident-demo.jpg`, are derived from that same authored replay and
released synthetic-company corpus. They retain the original source attribution
and are not new benchmark results.

Architectural references include ReAct and A-RAG
(https://github.com/Ayanami0730/arag). No A-RAG implementation was copied in this
extraction. Project-wide publication licensing has not been selected.

The archived model-generated report and observed passages in `examples/real-run/`
also come from the released EnterpriseRAG-Bench synthetic corpus above. Its answer
and tool decisions are real recorded model outputs, not authored replay. The
source artifacts and byte-level copy hashes are listed in `provenance.json`.

`experiments/overnight-20261012/official-source/answer_evaluation.py` contains
unaltered EnterpriseRAG-Bench evaluation prompt constants from revision
`d36685e273713975ee20299bbf1ab64165575b3c`, copyright 2026 DanswerAI, Inc.,
under the MIT license. The upstream license and pinned source URLs are beside
the file. The local compatibility runner uses sol and frozen references, so it
does not reproduce the full upstream evaluation or establish an official score.
