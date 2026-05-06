"""Generic client for local vision models via mlx_vlm CLI.

Provides a subprocess wrapper around `python -m mlx_vlm generate` for
reliable, prompt-engineered extraction from pre-cropped payslip blocks.

Validated model: mlx-community/Qwen3.5-9B-MLX-4bit (Qwen 3.5 9B 4-bit)
Hardware: Apple Silicon M-series (tested on M1 Pro 16GB)
"""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-MLX-4bit"


DEFAULT_MAX_TOKENS = 800
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 600  # seconds — per-crop inference only (download handled separately)


def _model_is_cached(model_name: str) -> bool:
    """Check if the HuggingFace model is already downloaded locally."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
    return model_cache.exists() and any(model_cache.iterdir())


def ensure_model_downloaded(model: str = DEFAULT_MODEL, timeout: int = 3600) -> None:
    """Download the model if not cached. Blocks until complete.

    Uses a 1x1 dummy image to trigger the download via mlx_vlm without
    doing meaningful inference. This separates the download step from
    actual vision extraction so timeouts don't overlap.
    """
    if _model_is_cached(model):
        logger.info("Model already cached locally.")
        return

    logger.info(
        "Model not cached. Starting download of ~9 GB from HuggingFace. "
        "This takes 10–25 minutes on first run depending on your connection. Please wait..."
    )

    # Create a minimal 1x1 PNG to satisfy mlx_vlm's --image requirement
    dummy_image = Path(__file__).parent / "_dummy_1x1.png"
    if not dummy_image.exists():
        from PIL import Image
        Image.new("RGB", (1, 1), color="white").save(dummy_image)

    cmd = [
        sys.executable, "-m", "mlx_vlm", "generate",
        "--model", model,
        "--max-tokens", "1",
        "--prompt", "hi",
        "--image", str(dummy_image),
    ]

    import threading
    import time

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Heartbeat thread: logs every 60 sec so users know the app hasn't frozen
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        elapsed = 0
        while not heartbeat_stop.wait(timeout=60):
            elapsed += 1
            logger.info("Download in progress... %d minute(s) elapsed. Please wait.", elapsed)

    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()

    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        heartbeat_stop.set()
        process.kill()
        process.wait()
        raise RuntimeError(f"Model download timed out after {timeout}s")
    finally:
        heartbeat_stop.set()

    if process.returncode != 0:
        err = process.stderr.strip() if process.stderr else "Unknown error"
        raise RuntimeError(f"Model download failed: {err}")

    logger.info("Model download complete. Ready to process payslips.")


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences and <think> blocks from model output."""
    # Strip <think>...</think> blocks (Qwen reasoning tags)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_json_block(text: str) -> Optional[str]:
    """Extract the outermost JSON object from unstructured text.

    Finds the first '{' and tracks brace depth to find its matching '}'.
    This ensures we get the top-level object, not a nested inner object.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    # Not valid JSON, keep looking
                    start = text.find("{", i + 1)
                    if start == -1:
                        break
    return None


def run_vision_extraction(
    image_path: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run mlx_vlm on an image with a text prompt and return parsed JSON.

    Args:
        image_path: Path to the image (PNG/JPG) to analyze.
        prompt: The vision prompt. Should instruct the model to return ONLY JSON.
        model: mlx_vlm model identifier.
        max_tokens: Generation token limit.
        temperature: 0.0 for deterministic output.
        timeout: Subprocess timeout in seconds.

    Returns:
        Parsed JSON dict. Returns {"_error": str} on failure so callers
        can decide whether to fallback.
    """
    try:
        cmd = [
            sys.executable, "-m", "mlx_vlm", "generate",
            "--model", model,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--prompt", prompt,
            "--image", image_path,
        ]
        logger.info("Running vision extraction on %s", image_path)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            err = result.stderr.strip() or "mlx_vlm exited with non-zero code"
            logger.error("Vision model failed: %s", err)
            return {"_error": f"mlx_vlm failed: {err}"}

        output = result.stdout

        # Strip prompt echo — keep only the assistant's response
        # mlx_vlm output format: prompt...<|im_end|>\n<|im_start|>assistant\n<think>...</think>\n{JSON}...
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker in output:
            output = output.split(assistant_marker, 1)[-1]

        # Strip <think> blocks before JSON extraction
        output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)

        # Find the JSON block
        json_match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
        if json_match:
            raw_json = json_match.group(1)
        else:
            # Fallback: find the last complete { ... } block
            # Use a greedy scan from the end to find valid JSON
            candidate = _extract_json_block(output)
            if candidate is None:
                logger.error("No JSON found in vision model output")
                return {"_error": "No JSON found in model output"}
            raw_json = candidate

        raw_json = _strip_fences(raw_json)
        parsed = json.loads(raw_json)
        return parsed

    except subprocess.TimeoutExpired:
        logger.error("Vision model timed out after %ds", timeout)
        return {"_error": f"Timeout after {timeout}s"}
    except json.JSONDecodeError as exc:
        logger.error("Vision model returned invalid JSON: %s", exc)
        return {"_error": f"Invalid JSON: {exc}"}
    except Exception as exc:
        logger.exception("Unexpected error in vision extraction")
        return {"_error": str(exc)}
