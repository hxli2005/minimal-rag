"""Bounded model/environment preparation followed by the fixed throughput sample."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    state = {"started": time.time(), "deadline": args.deadline, "stages": []}
    processes = []

    def save():
        Path("preparation-state.json").write_text(json.dumps(state, indent=2)+"\n")

    def start(name, command):
        state["stage"] = name; save()
        with open(name + ".log", "w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
        return process

    def wait(name, process):
        state["stage"] = name; save()
        code = process.wait(timeout=max(1, args.deadline-time.time()))
        state["stages"].append({"stage": name, "returncode": code, "finished": time.time()}); save()
        if code: raise RuntimeError(f"{name} failed: {code}")

    try:
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", ".gpu-venv"], check=True)
        download = start("download", [sys.executable, "-u", "download.py"])
        environment = start("environment", [".gpu-venv/bin/python", "-m", "pip", "install", "--no-cache-dir",
                                            "transformers==4.53.2", "numpy==1.26.4"])
        wait("environment", environment); wait("download", download)
        probe = start("probe", [".gpu-venv/bin/python", "-u", "gpu.py", "encode", "--model", "models/embedding",
                                "--input", "enterprise-probe.jsonl.gz", "--output", "probe",
                                "--deadline", str(args.deadline)])
        wait("probe", probe)
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=type(error).__name__, detail=str(error))
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        state["finished"] = time.time(); save()
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
