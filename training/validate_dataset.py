"""Validate an explicitly approved instruction dataset without training it."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SECRET_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b")


def _normalise_messages(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    clean = []
    for message in messages:
        if not isinstance(message, dict):
            return None
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str) or not content.strip():
            return None
        clean.append({"role": role, "content": " ".join(content.split())})
    if not any(message["role"] == "assistant" for message in clean):
        return None
    return clean


def validate(path: Path):
    report = {"rows": 0, "valid": 0, "invalid": 0, "duplicates": 0, "sensitive": 0,
              "train": 0, "validation": 0}
    cleaned, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        report["rows"] += 1
        try:
            messages = _normalise_messages(json.loads(line))
        except json.JSONDecodeError:
            messages = None
        if not messages:
            report["invalid"] += 1
            continue
        serialised = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        if digest in seen:
            report["duplicates"] += 1
            continue
        seen.add(digest)
        if SECRET_RE.search(serialised):
            report["sensitive"] += 1
            continue
        cleaned.append({"messages": messages, "_split": "validation" if int(digest[:8], 16) % 10 == 0 else "train"})
        report["valid"] += 1
    report["train"] = sum(row["_split"] == "train" for row in cleaned)
    report["validation"] = sum(row["_split"] == "validation" for row in cleaned)
    return report, cleaned


def main():
    parser = argparse.ArgumentParser(description="Validate an approved SFT JSONL dataset; it never trains or uploads.")
    parser.add_argument("--input", required=True, help="Approved local JSONL path")
    parser.add_argument("--write-cleaned", help="Optional output JSONL path after owner review")
    args = parser.parse_args()
    report, cleaned = validate(Path(args.input))
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.write_cleaned:
        target = Path(args.write_cleaned)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n" for row in cleaned), encoding="utf-8")


if __name__ == "__main__":
    main()
