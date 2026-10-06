#!/usr/bin/env python3
"""
Lightweight WandB logger for bash-driven pipelines.
"""

from __future__ import annotations

import argparse
import json
import os
import time

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--event", type=str, required=True)
    parser.add_argument("--status", type=str, required=True)
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--job_type", type=str, default="pipeline")
    parser.add_argument("--mode", type=str, default="online")
    parser.add_argument("--config_json", type=str, default="")
    parser.add_argument("--metrics_json", type=str, default="")
    parser.add_argument("--file", action="append", default=[])
    return parser.parse_args()


def _load_json(path_or_json: str) -> dict:
    if not path_or_json:
        return {}
    if os.path.isfile(path_or_json):
        with open(path_or_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path_or_json)


def main() -> None:
    args = _parse_args()
    if wandb is None:
        print("[WARN] wandb is not installed; skipping wandb log event.")
        return

    config = _load_json(args.config_json) if args.config_json else {}
    metrics = _load_json(args.metrics_json) if args.metrics_json else {}

    init_kwargs = {
        "project": args.project,
        "name": args.run_name,
        "id": args.run_id,
        "resume": "allow",
        "job_type": args.job_type,
        "config": config,
        "mode": args.mode,
    }
    if args.entity:
        init_kwargs["entity"] = args.entity

    try:
        run = wandb.init(**init_kwargs)
        payload = {
            "pipeline/event": args.event,
            "pipeline/status": args.status,
            "pipeline/timestamp": time.time(),
        }
        payload.update(metrics)
        wandb.log(payload)

        # Intentionally disable file/media/artifact upload for faster pipeline logging.
        # Keep args.file in CLI for backward compatibility, but do not upload those files.

        run.finish()
    except Exception as e:
        print(f"[WARN] wandb logging skipped: {e}")
        return


if __name__ == "__main__":
    main()
