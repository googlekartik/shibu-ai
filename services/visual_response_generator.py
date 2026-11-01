"""VisualResponseGenerator: derive charts/flowcharts/diagrams.

Derivations use only numbers and steps already in the structured
blocks. A chart is produced only when at least two numeric values
with labels exist; a flowchart only when there are at least two
steps. Anything else is omitted rather than guessed. Image
galleries are produced only from source items that actually carry
an image URL; Shibu AI's current sources carry none, so no gallery
is fabricated.
"""
from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|percent|million|billion|thousand)?")


def _chart_from_blocks(blocks: list) -> dict | None:
    labels, values = [], []
    for block in blocks:
        if block.get("kind") == "table":
            for row in block.get("rows", []):
                if not row:
                    continue
                match = _NUMBER_RE.search(str(row[0]))
                if match:
                    try:
                        value = float(match.group(1).replace(",", "."))
                    except ValueError:
                        continue
                    label = re.sub(r"\d+(?:[.,]\d+)?\s*(%|percent|million|billion|thousand)?", "", str(row[0])).strip(" :,-")
                    labels.append((label or f"Item {len(labels) + 1}")[:40])
                    values.append(value)
    if len(values) < 2:
        return None
    return {"kind": "chart", "chart_type": "bar", "title": "Figures from the sources",
            "labels": labels[:8], "values": values[:8]}


def _flowchart_from_blocks(blocks: list, content_type: str) -> dict | None:
    if content_type not in ("process_steps", "algorithm", "roadmap", "explain"):
        return None
    for block in blocks:
        if block.get("kind") == "steps" and len(block.get("items", [])) >= 2:
            nodes = []
            for i, item in enumerate(block["items"][:10]):
                node_type = "decision" if re.search(r"\bif\b", item, re.I) else "process"
                if i == 0:
                    node_type = "start"
                elif i == len(block["items"]) - 1:
                    node_type = "end"
                nodes.append({"label": item[:120], "type": node_type})
            return {"kind": "flowchart", "title": "Workflow", "nodes": nodes}
    return None


def _gallery_from_sources(sources: list) -> dict | None:
    images = []
    for source in sources or []:
        url = source.get("image_url") or source.get("image")
        if url:
            images.append({"url": url, "caption": source.get("source_name", "Image")})
    if not images:
        return None
    return {"kind": "gallery", "images": images[:6]}


def generate_visuals(blocks: list, intent: dict, sources: list = None) -> list:
    """Return the visual blocks to append after the structured blocks."""
    visuals = []
    content_type = intent.get("content_type", "explain")
    if content_type == "statistics":
        chart = _chart_from_blocks(blocks)
        if chart:
            visuals.append(chart)
    flowchart = _flowchart_from_blocks(blocks, content_type)
    if flowchart:
        visuals.append(flowchart)
    gallery = _gallery_from_sources(sources or [])
    if gallery:
        visuals.append(gallery)
    return visuals
