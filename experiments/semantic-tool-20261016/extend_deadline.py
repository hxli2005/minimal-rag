"""Resume the unchanged encoder at the next completed shard with a later deadline."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--deadline", type=float, required=True)
args = parser.parse_args()
prior = json.loads(Path("full-process.json").read_text())
initial = json.loads(Path("enterprise-index/state.json").read_text())
state = {"pid": os.getpid(), "old_pid": prior["pid"], "started": time.time(),
         "initial_documents": initial["documents"], "deadline": args.deadline}


def save(status, **fields):
    state.update(status=status, updated=time.time(), **fields)
    tmp = Path("extend-deadline-state.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace("extend-deadline-state.json")


try:
    save("waiting_for_completed_shard")
    while time.time() < state["started"] + 3600:
        index = json.loads(Path("enterprise-index/state.json").read_text())
        full = json.loads(Path("full-state.json").read_text())
        if index.get("complete"):
            save("already_complete"); break
        if full.get("status") == "failed":
            raise RuntimeError("Pipeline failed before the checkpoint; inspect without restarting")
        if index["documents"] > initial["documents"]:
            assert json.loads(Path("full-process.json").read_text())["pid"] == prior["pid"]
            Path("full-before-deadline-extension.json").write_text(json.dumps(full, indent=2) + "\n")
            os.killpg(prior["pid"], signal.SIGTERM)
            pids = f"{prior['pid']},{full['processes']['encode']}"
            for _ in range(50):
                statuses = subprocess.run(["ps", "-p", pids, "-o", "stat="], capture_output=True, text=True).stdout.split()
                if all(s.startswith("Z") for s in statuses):
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError("Old encoder is still active; not starting a duplicate")
            with Path("full.log").open("a") as log:
                process = subprocess.Popen(["/root/miniconda3/bin/python", "-u", "full_index.py", "--prepared-corpus",
                                            "--deadline", str(args.deadline)], stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            Path("full-process.json").write_text(json.dumps({"pid": process.pid, "started": time.time(),
                "deadline": args.deadline, "reason": "Extended protection window at a completed shard; identical encoder settings",
                "resumed_documents": index["documents"]}, indent=2) + "\n")
            save("complete", new_pid=process.pid, resumed_documents=index["documents"])
            break
        time.sleep(0.2)
    else:
        raise TimeoutError("No completed shard within one hour")
except Exception as error:
    save("failed", error=repr(error))
    raise
