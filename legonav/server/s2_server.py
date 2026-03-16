#!/usr/bin/env python3
"""
LegoNav S2 Server — 视觉语言导航解析服务

支持两种推理后端：
  local  (默认) — 本机 GPU 加载 Qwen-VL 模型（Qwen2-VL / Qwen2.5-VL / Qwen3-VL）
  api           — 调用外部视觉语言模型 API（OpenAI 兼容协议）
                  通过 --provider 指定服务商：
                    openai  → GPT-4o / GPT-4.1 等
                    gemini  → Gemini 2.5 Pro / 2.0 Flash 等
                    kimi    → Kimi VL (Moonshot AI)
                    qwen    → Qwen-VL-Max / Qwen2.5-VL 等 (DashScope)
                    custom  → 任意 OpenAI 兼容接口，需配合 --api_base_url

API:
  GET  /health          → {"status": "ok", "model": "...", "backend": "...", "provider": "..."}
  POST /s2_step         → {"target", "point_2d_norm", "point_2d_pixel", "navigation", "raw"}

启动示例:
  # 本地 GPU (Qwen2.5-VL-7B)
  python s2_server.py --model_path Qwen/Qwen2.5-VL-7B-Instruct

  # OpenAI GPT-4o
  python s2_server.py --backend api --provider openai \\
      --model_path gpt-4o --api_key sk-xxx

  # Google Gemini 2.5 Pro
  python s2_server.py --backend api --provider gemini \\
      --model_path gemini-2.5-pro --api_key AIzaSy-xxx

  # Kimi 2.5 VL (Moonshot)
  python s2_server.py --backend api --provider kimi \\
      --model_path moonshot-v1-vision --api_key sk-xxx

  # Qwen-VL-Max (DashScope)
  python s2_server.py --backend api --provider qwen \\
      --model_path qwen-vl-max --api_key sk-xxx

  # 自定义 OpenAI 兼容接口
  python s2_server.py --backend api --provider custom \\
      --model_path your-model-id \\
      --api_base_url https://your-endpoint/v1 --api_key sk-xxx

  # API Key 也可通过环境变量传入（按 provider 自动识别）
  OPENAI_API_KEY=sk-xxx python s2_server.py --backend api --provider openai --model_path gpt-4o

依赖:
  pip install flask transformers>=4.51.0 qwen-vl-utils accelerate pillow numpy
  pip install openai          # api 模式必须
  pip install flash-attn --no-build-isolation  # 可选，local 模式加速
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import traceback

from flask import Flask, jsonify, request

# ─────────────────────────────────────────────────────────────────────────────
# 外部服务商配置
# ─────────────────────────────────────────────────────────────────────────────
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "models": [
            "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.1-mini",
            "o1", "o3-mini",
        ],
        "note": "需要 OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
        "models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        "note": "需要 GEMINI_API_KEY",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-vision",
        "models": [
            "moonshot-v1-vision",
            "kimi-vl-a3b-thinking",
            "kimi-latest",
        ],
        "note": "需要 MOONSHOT_API_KEY (Moonshot AI / Kimi)",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-vl-max",
        "models": [
            "qwen-vl-max",
            "qwen-vl-max-latest",
            "qwen-vl-plus",
            "qwen-vl-plus-latest",
            "qwen2.5-vl-72b-instruct",
            "qwen2.5-vl-7b-instruct",
            "qvq-max",
            "qvq-72b-preview",
        ],
        "note": "需要 DASHSCOPE_API_KEY",
    },
    "custom": {
        "base_url": None,  # 必须通过 --api_base_url 指定
        "env_key": "API_KEY",
        "default_model": "",
        "models": [],
        "note": "任意 OpenAI 兼容接口，需提供 --api_base_url",
    },
}

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
api_client = None   # openai.OpenAI instance (api 模式)
cfg = None          # argparse namespace, set in main()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — local backend
# ─────────────────────────────────────────────────────────────────────────────
def _detect_model_class(model_path: str):
    """根据模型名称自动选择对应的 transformers 类。"""
    name = model_path.lower()
    if "qwen3" in name:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if "qwen2.5" in name or "qwen2_5" in name or "qwen2-5" in name:
        from transformers import Qwen2_5VLForConditionalGeneration
        return Qwen2_5VLForConditionalGeneration
    if "qwen2" in name:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    print(f"[S2] 未识别模型系列 '{model_path}'，默认使用 Qwen2_5VLForConditionalGeneration", flush=True)
    from transformers import Qwen2_5VLForConditionalGeneration
    return Qwen2_5VLForConditionalGeneration


def load_model(model_path: str, device: str) -> None:
    global model, processor

    import torch
    from transformers import AutoProcessor

    ModelClass = _detect_model_class(model_path)
    print(f"[S2] Model class: {ModelClass.__name__}", flush=True)
    print(f"[S2] Loading processor from {model_path} …", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)

    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device)
    for attn_impl in ("flash_attention_2", "sdpa"):
        try:
            model = ModelClass.from_pretrained(
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
# API client init — api backend
# ─────────────────────────────────────────────────────────────────────────────
def init_api_client(api_key: str, base_url: str) -> None:
    global api_client
    try:
        import openai
    except ImportError:
        raise ImportError("openai 包未安装，请运行: pip install openai")

    api_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    print(f"[S2] API client ready. provider={cfg.provider} base_url={base_url}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference — local backend
# ─────────────────────────────────────────────────────────────────────────────
def run_inference_local(image_bytes: bytes, instruction: str) -> str:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(next(model.parameters()).device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Inference — api backend
# ─────────────────────────────────────────────────────────────────────────────
def _image_to_data_url(image_bytes: bytes) -> str:
    """将图像字节转为 base64 data URL。"""
    if image_bytes[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def run_inference_api(image_bytes: bytes, instruction: str) -> str:
    """通过 OpenAI 兼容接口调用外部视觉语言模型。"""
    data_url = _image_to_data_url(image_bytes)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": instruction},
            ],
        },
    ]

    response = api_client.chat.completions.create(
        model=cfg.model_path,
        messages=messages,
        max_tokens=cfg.max_new_tokens,
        temperature=0,
        stream=False,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Unified inference entry
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(image_bytes: bytes, instruction: str) -> str:
    if cfg.backend == "api":
        return run_inference_api(image_bytes, instruction)
    return run_inference_local(image_bytes, instruction)


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────
_NAV_PATTERN = re.compile(r"[←→↑↓]+|stop|start")
_JSON_PATTERN = re.compile(r"\{[^{}]+\}", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    m = _CODE_FENCE.search(text)
    return m.group(1) if m else text


def _extract_json_array(text: str):
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
    u = max(0, min(cfg.image_width - 1,  int(nx / 1000.0 * cfg.image_width)))
    v = max(0, min(cfg.image_height - 1, int(ny / 1000.0 * cfg.image_height)))
    return u, v


def parse_output(raw: str) -> dict:
    target = None
    point_2d_norm = None
    point_2d_pixel = None
    navigation = ""

    clean = _strip_code_fence(raw)

    # ── 新格式：JSON 任务数组 ─────────────────────────────────────────────────
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
                    task["point_2d_pixel"] = [u, v]
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
            "tasks": tasks,
            "target": target,
            "point_2d_norm": point_2d_norm,
            "point_2d_pixel": point_2d_pixel,
            "navigation": navigation,
        }

    # ── 旧版兼容：两行格式 ────────────────────────────────────────────────────
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
    ready = (model is not None) if cfg.backend == "local" else (api_client is not None)
    return jsonify({
        "status": "ok" if ready else "loading",
        "backend": cfg.backend if cfg else "unknown",
        "provider": cfg.provider if cfg and cfg.backend == "api" else "local",
        "model": cfg.model_path if cfg else "not loaded",
    })


@app.route("/s2_step", methods=["POST"])
def s2_step():
    if cfg.backend == "local" and model is None:
        return jsonify({"error": "model not loaded"}), 503
    if cfg.backend == "api" and api_client is None:
        return jsonify({"error": "api client not initialized"}), 503

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
    return ((n + 31) // 32) * 32


def _build_provider_help() -> str:
    lines = []
    for name, conf in PROVIDER_CONFIGS.items():
        models = ", ".join(conf["models"][:3]) + ("…" if len(conf["models"]) > 3 else "")
        lines.append(f"  {name:8s}: {conf['note']}; 示例模型: {models or '见 --api_base_url'}")
    return "\n" + "\n".join(lines)


def main():
    global cfg

    parser = argparse.ArgumentParser(
        description="LegoNav S2 Server — 视觉语言导航服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"支持的 provider:{_build_provider_help()}",
    )

    # ── 后端 ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--backend", default="local", choices=["local", "api"],
        help="推理后端: local=本机 GPU, api=外部 API",
    )

    # ── API 模式专属 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--provider", default="openai",
        choices=list(PROVIDER_CONFIGS.keys()),
        help="API 服务商 (仅 --backend api 时生效)",
    )
    parser.add_argument(
        "--api_key", default=None,
        help=(
            "API Key（优先级高于环境变量）。"
            "各服务商默认读取的环境变量: "
            "openai→OPENAI_API_KEY, gemini→GEMINI_API_KEY, "
            "kimi→MOONSHOT_API_KEY, qwen→DASHSCOPE_API_KEY, custom→API_KEY"
        ),
    )
    parser.add_argument(
        "--api_base_url", default=None,
        help="覆盖服务商默认 base_url（custom 模式必填）",
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_path", default=None,
        help=(
            "local: HuggingFace 模型 ID 或本地路径 (默认 Qwen/Qwen2.5-VL-7B-Instruct); "
            "api: 模型名称，默认使用各 provider 的推荐模型"
        ),
    )

    # ── 服务器 ────────────────────────────────────────────────────────────────
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", default="0.0.0.0")

    # ── local 模式专属 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--device", default="auto",
        help="device_map: 'auto', 'cuda:0', 'cpu'（仅 local 模式）",
    )

    # ── 图像尺寸 ──────────────────────────────────────────────────────────────
    parser.add_argument("--image_width",  type=int, default=1280,
                        help="机器人相机宽度 (Gemini 336L: 1280 | Astra S: 640)")
    parser.add_argument("--image_height", type=int, default=720,
                        help="机器人相机高度 (Gemini 336L: 720  | Astra S: 480)")
    parser.add_argument("--resize_w", type=int, default=640,
                        help="送入本地模型的图像宽度（local 模式）")
    parser.add_argument("--resize_h", type=int, default=360,
                        help="送入本地模型的图像高度（local 模式）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数")

    cfg = parser.parse_args()
    cfg.resize_w = _round32(cfg.resize_w)
    cfg.resize_h = _round32(cfg.resize_h)

    # ── 填充 model_path 默认值 ────────────────────────────────────────────────
    if cfg.model_path is None:
        if cfg.backend == "local":
            cfg.model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            cfg.model_path = PROVIDER_CONFIGS[cfg.provider]["default_model"]

    print(f"[S2] Config: {vars(cfg)}", flush=True)

    # ── 初始化推理后端 ────────────────────────────────────────────────────────
    if cfg.backend == "local":
        load_model(cfg.model_path, cfg.device)
    else:
        provider_conf = PROVIDER_CONFIGS[cfg.provider]

        # 解析 API Key
        api_key = cfg.api_key or os.environ.get(provider_conf["env_key"])
        if not api_key:
            print(
                f"[S2] 错误: provider={cfg.provider} 需要 API Key。\n"
                f"  通过 --api_key 传入，或设置环境变量 {provider_conf['env_key']}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 解析 base_url
        base_url = cfg.api_base_url or provider_conf["base_url"]
        if not base_url:
            print(
                "[S2] 错误: custom provider 必须提供 --api_base_url",
                file=sys.stderr,
            )
            sys.exit(1)

        init_api_client(api_key, base_url)

    print(f"[S2] Listening on http://{cfg.host}:{cfg.port}", flush=True)
    app.run(host=cfg.host, port=cfg.port, threaded=(cfg.backend == "api"))


if __name__ == "__main__":
    main()
