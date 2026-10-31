"""Configuration-level tests; they do not create paid services or GPU jobs."""
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _training_validator():
    path = ROOT / "training" / "validate_dataset.py"
    spec = importlib.util.spec_from_file_location("shibu_training_validator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_validation_removes_duplicates_and_secret_like_text(tmp_path):
    data = tmp_path / "approved.jsonl"
    valid = {"messages": [{"role": "user", "content": "Explain gravity."},
                          {"role": "assistant", "content": "Gravity attracts masses."}]}
    sensitive = {"messages": [{"role": "user", "content": "Use sk-12345678901234567890"},
                                {"role": "assistant", "content": "No."}]}
    data.write_text("\n".join(json.dumps(row) for row in (valid, valid, sensitive)) + "\n", encoding="utf-8")
    report, rows = _training_validator().validate(data)
    assert report["valid"] == 1 and report["duplicates"] == 1 and report["sensitive"] == 1
    assert rows[0]["messages"][0]["content"] == "Explain gravity."


def test_cloud_training_config_is_template_with_manual_promotion_only():
    config = (ROOT / "training" / "configs" / "cloud_lora_sft.yaml").read_text(encoding="utf-8")
    assert "automatic: false" in config
    assert "allow_private_user_content: false" in config
    assert "quantization: int4_if_supported" in config


def test_render_uses_hosted_only_production_process():
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "gunicorn" in render
    assert "CLOUD_ONLY" in render
    assert "HF_USE_LOCAL" in render
    assert "/api/health" in render
