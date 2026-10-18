"""Validate the complete retrieved Enterprise index before starting B groups."""
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import time

import numpy as np

RUN = Path(__file__).resolve().parents[2] / "runs/semantic-tool-20261016"
INDEX = RUN / "a800-results/enterprise-index"


def main():
    manifest_path = INDEX / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["index"]["complete"] and manifest["index"]["documents"] == 511962
    shards = sorted(manifest["shards"], key=lambda item: item["file"])
    assert [item["file"] for item in shards] == [f"shard-{i:05d}.npz" for i in range(125)]
    seen, maximum_error = set(), 0.0
    started = time.time()
    with gzip.open(RUN / "enterprise.jsonl.gz", "rt") as source:
        for item in shards:
            path = INDEX / item["file"]
            assert path.stat().st_size == item["bytes"]
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
            expected = [json.loads(line)["id"] for line in islice(source, 4096)]
            with np.load(path, allow_pickle=False) as data:
                ids, vectors = data["ids"].tolist(), data["vectors"]
                assert ids == expected, item["file"]
                assert vectors.shape == (len(expected), 4096) and vectors.dtype == np.float32
                assert np.isfinite(vectors).all()
                error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1)))
                assert error < 1e-5, (item["file"], error)
                maximum_error = max(maximum_error, error)
                assert not seen.intersection(ids)
                seen.update(ids)
        assert not source.readline()
    assert len(seen) == 511962
    result = dict(checked_at=time.time(), seconds=time.time()-started, complete_index=True,
                  documents=len(seen), shards=len(shards), dimensions=4096, dtype="float32",
                  all_file_hashes_match=True, ids_match_original_ranges=True,
                  unique_ids=True, finite=True, max_unit_norm_error=maximum_error,
                  manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    (RUN / "enterprise-index-validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
