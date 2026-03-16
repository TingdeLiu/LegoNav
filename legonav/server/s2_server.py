#!/usr/bin/env python3
"""
Wheeltec S2 Server — Qwen3-VL 视觉语言导航解析服务

在 GPU 服务器上运行，接收 Jetson 端发来的 RGB 图像 + 导航指令，
返回目标像素坐标 (u, v) 和导航控制符号序列。

API:
  GET  /health          → {"status": "ok", "model": "..."}
  POST /s2_step         → {"target", "point_2d_norm", "point_2d_pixel", "navigation", "raw"}

启动:
  python wheeltec_s2_server.py \
      --model_path Qwen/Qwen3-VL-7B-Instruct \
      --port 8890 \
      --device auto

依赖 (在 internnav conda 环境中安装):
  pip install flask transformers>=4.57.0 qwen-vl-utils
  pip install flash-attn --no-build-isolation  # 可选，无则自动降级
"""

import argparse
import io
import json
import re
import sys
import traceback

from flask import Flask, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
# Role
You are a high-precision robot visual navigation and task decomposition system. Decompose the user's compound navigation instruction into multiple sequential atomic tasks and output them as a JSON array.

# Step 1: Task Decomposition
Split the user instruction into independent atomic tasks by action boundaries, in sequential order.
Atomic task types (task field):
- `pixel_point`: visual target localization
- `move`: movement or rotation action

# Step 2: Output Format
Output strictly one JSON array. Each element corresponds to one atomic task.

## pixel_point format
{"task": "pixel_point", "target": "<target name>", "point_2d": [x, y]}
- Coordinate range [0, 1000], x is horizontal, y is vertical
- Must be based on actual image content; return [null, null] if target cannot be located
- Only output this task type when the instruction contains a navigation target

## move format
{"task": "move", "action": "<symbol>", "number": <integer>}

| action | meaning | unit |
|--------|---------|------|
| ← | turn left | per 15° |
| → | turn right | per 15° |
| ↑ | move forward | per 0.5m |
| ↓ | move backward | per 0.5m |
| stop | stop | number fixed to 1 |

# Decomposition Rules
- **Turn instruction** → 1 move task (turn left 60° → action: "←", number: 4)
- **"Go to target" instruction** → 1 pixel_point task ONLY, do NOT append any move task
- **Explicit distance instruction** → 1 move task (forward 2m → action: "↑", number: 4)
- **Rotate in place** → 1 move task (clockwise 360° → action: "→", number: 24; counterclockwise 360° → action: "←", number: 24)
- **Stop instruction** → 1 move task (action: "stop", number: 1)

# Constraints
- Output only the raw JSON array, no explanatory text, no preamble, no closing remarks, no markdown code block markers
- Array element order must strictly follow the execution order of the instruction
- Visual coordinates must be based on actual image content, refuse hallucination

---

## Example Validation

**Input**: `Turn left 60 degrees, then go to the black chair, then rotate counterclockwise in place for one full turn`

**Expected output**:
```json
[
  {"task": "move", "action": "←", "number": 4},
  {"task": "pixel_point", "target": "black chair", "point_2d": [320, 680]},
  {"task": "move", "action": "←", "number": 24}
]
```\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Globals
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = None
processor = None
cfg = None  # argparse namespace, set in main()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str, device: str) -> None:
    global model, processor

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"[S2] Loading processor from {model_path} …", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device)

    # Try flash_attention_2, fall back to sdpa (PyTorch scaled-dot-product)
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, attn_implementation=attn_impl, **load_kwargs
            )
            print(f"[S2] Loaded with attn_implementation={attn_impl}", flush=True)
            break
        except Exception as exc:
            print(f"[S2] {attn_impl} unavailable: {exc}", flush=True)
    else:
        raise RuntimeError("Could not load model with any attention implementation.")

    model.eval()
    print("[S2] Model ready.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(image_bytes: bytes, instruction: str) -> str:
    """Call Qwen3-VL and return the raw text output."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                    "resized_width": cfg.resize_w,
                    "resized_height": cfg.resize_h,
                },
                {"type": "text", "text": instruction},
            ],
        },
    ]

    # Standard Qwen3-VL inference path via process_vision_info
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )

    target_device = next(model.parameters()).device
    inputs = inputs.to(target_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Strip prompt tokens, decode only newly generated tokens
    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────
_NAV_PATTERN = re.compile(r"[←→↑↓]+|stop|start")
_JSON_PATTERN = re.compile(r"\{[^{}]+\}", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
    m = _CODE_FENCE.search(text)
    return m.group(1) if m else text


def _extract_json_array(text: str):
    """Find and parse the first JSON array in text (handles nested structures)."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _norm_to_pixel(nx: float, ny: float):
    """Convert [0, 1000] normalised coords to clamped pixel (u, v)."""
    u = max(0, min(cfg.image_width - 1,  int(nx / 1000.0 * cfg.image_width)))
    v = max(0, min(cfg.image_height - 1, int(ny / 1000.0 * cfg.image_height)))
    return u, v


def parse_output(raw: str) -> dict:
    """
    Parse model output in either the new JSON-array format or the legacy two-line format.

    New format (JSON array of task objects):
      [
        {"task": "move", "action": "←", "number": 4},
        {"task": "pixel_point", "target": "black chair", "point_2d": [710, 220]}
      ]

    Legacy format (two lines):
      {"target": "chair", "point_2d": [320, 240]}
      ↑↑←

    Returns:
      target          – string or None
      point_2d_norm   – [x, y] in [0, 1000] or None
      point_2d_pixel  – [u, v] in actual image pixels or None
      navigation      – concatenated nav symbols, e.g. "↑↑←←←←" or "stop"
      raw             – original model output (for debugging)
    """
    target = None
    point_2d_norm = None
    point_2d_pixel = None
    navigation = ""

    # Strip markdown code fences that the model may add despite instructions
    clean = _strip_code_fence(raw)

    # ── New format: JSON array of tasks ──────────────────────────────────────
    tasks = _extract_json_array(clean)
    if isinstance(tasks, list):
        nav_parts = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_type = task.get("task")

            if task_type == "pixel_point":
                norm = task.get("point_2d")
                if isinstance(norm, (list, tuple)) and len(norm) == 2 and norm[0] is not None:
                    nx, ny = float(norm[0]), float(norm[1])
                    u, v = _norm_to_pixel(nx, ny)
                    task["point_2d_pixel"] = [u, v]   # 供 pipeline 直接使用，无需再转换
                else:
                    task["point_2d_pixel"] = None
                if target is None:
                    target = task.get("target")
                    if task.get("point_2d_pixel") is not None:
                        point_2d_norm = [int(nx), int(ny)]
                        point_2d_pixel = task["point_2d_pixel"]

            elif task_type == "move":
                action = task.get("action", "")
                number = max(1, int(task.get("number", 1)))
                nav_parts.append("stop" if action == "stop" else action * number)

        navigation = "".join(nav_parts)
        return {
            "raw": raw,
            "tasks": tasks,          # 原始任务列表，供 pipeline 顺序执行
            "target": target,
            "point_2d_norm": point_2d_norm,
            "point_2d_pixel": point_2d_pixel,
            "navigation": navigation,
        }

    # ── Fallback: legacy two-line format ─────────────────────────────────────
    json_match = _JSON_PATTERN.search(clean)
    if json_match:
        try:
            data = json.loads(json_match.group())
            target = data.get("target")
            norm = data.get("point_2d")
            if isinstance(norm, (list, tuple)) and len(norm) == 2:
                nx, ny = float(norm[0]), float(norm[1])
                point_2d_norm = [int(nx), int(ny)]
                u, v = _norm_to_pixel(nx, ny)
                point_2d_pixel = [u, v]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    nav_parts = _NAV_PATTERN.findall(clean)
    navigation = "".join(nav_parts)

    return {
        "raw": raw,
        "target": target,
        "point_2d_norm": point_2d_norm,
        "point_2d_pixel": point_2d_pixel,
        "navigation": navigation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": cfg.model_path if cfg else "not loaded"})


@app.route("/s2_step", methods=["POST"])
def s2_step():
    """
    POST /s2_step
    Form fields:
      image       – image file (JPEG / PNG)
      instruction – natural language navigation instruction

    Response JSON:
      target          – detected target name or null
      point_2d_norm   – [x, y] in [0, 1000] or null
      point_2d_pixel  – [u, v] in pixel coords (cfg.image_width × cfg.image_height) or null
      navigation      – nav symbol string, e.g. "↑↑←←" / "stop" / ""
      raw             – raw model text output (for debugging)
    """
    if model is None:
        return jsonify({"error": "model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "missing form field: image"}), 400

    instruction = request.form.get("instruction", "").strip()
    if not instruction:
        return jsonify({"error": "missing form field: instruction"}), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"error": "image file is empty"}), 400

    try:
        raw_output = run_inference(image_bytes, instruction)
        result = parse_output(raw_output)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc), "traceback": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def _round32(n: int) -> int:
    """Round up to the nearest multiple of 32 (required by Qwen3-VL)."""
    return ((n + 31) // 32) * 32


def main():
    global cfg

    parser = argparse.ArgumentParser(
        description="Wheeltec S2 Server — Qwen3-VL navigation instruction parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", default="Qwen/Qwen3-VL-7B-Instruct",
        help="HuggingFace model ID or local checkpoint directory",
    )
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--device", default="auto",
        help="device_map value: 'auto', 'cuda:0', 'cuda:1', 'cpu', …",
    )
    # Camera / image dimensions (used for pixel coordinate conversion)
    parser.add_argument("--image_width",  type=int, default=1280,
                        help="Robot camera width in pixels  (Gemini 336L: 1280 | Astra S: 640)")
    parser.add_argument("--image_height", type=int, default=720,
                        help="Robot camera height in pixels (Gemini 336L: 720  | Astra S: 480)")
    # Resolution fed to the model (multiples of 32)
    # Gemini 336L: 640×360 keeps 16:9 aspect ratio
    # Astra S:     640×480 keeps 4:3 aspect ratio
    parser.add_argument("--resize_w", type=int, default=640,
                        help="Image width passed to Qwen3-VL (rounded up to multiple of 32)")
    parser.add_argument("--resize_h", type=int, default=360,
                        help="Image height passed to Qwen3-VL (Gemini 336L: 360 | Astra S: 480)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate per request")

    cfg = parser.parse_args()

    # Enforce multiples of 32
    cfg.resize_w = _round32(cfg.resize_w)
    cfg.resize_h = _round32(cfg.resize_h)

    print(f"[S2] Config: {vars(cfg)}", flush=True)

    load_model(cfg.model_path, cfg.device)

    print(f"[S2] Listening on http://{cfg.host}:{cfg.port}", flush=True)
    # threaded=False: single-GPU model is not thread-safe; requests queue naturally
    app.run(host=cfg.host, port=cfg.port, threaded=False)


if __name__ == "__main__":
    main()
