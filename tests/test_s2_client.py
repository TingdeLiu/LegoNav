#!/usr/bin/env python3
"""
S2 服务器测试客户端

用法 (在 Jetson 或任意机器上):
  # 使用本地图片文件
  python test_s2_client.py \
      --image /path/to/test.jpg \
      --instruction "Go to the red chair" \
      --host 192.168.1.100 --port 8890

  # 使用随机噪声图（纯连通性测试，不验证识别结果）
  python test_s2_client.py --random \
      --instruction "Go to the door" \
      --host 192.168.1.100 --port 8890
"""

import argparse
import io
import json
import sys

import numpy as np
import requests
from PIL import Image


def make_random_image(width: int = 640, height: int = 480) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def load_image(path: str) -> bytes:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def call_s2(host: str, port: int, image_bytes: bytes, instruction: str) -> dict:
    url = f"http://{host}:{port}/s2_step"
    resp = requests.post(
        url,
        files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
        data={"instruction": instruction},
        timeout=60,
    )
    if not resp.ok:
        # Print server-side error details before raising
        try:
            body = resp.json()
            print(f"\n[Server ERROR {resp.status_code}]")
            print(f"  error    : {body.get('error', '(no error field)')}")
            tb = body.get("traceback", "")
            if tb:
                print(f"  traceback:\n{tb}")
        except Exception:
            print(f"\n[Server ERROR {resp.status_code}] raw body: {resp.text[:2000]}")
        resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="S2 server test client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--instruction", default="Go to the red chair")
    parser.add_argument("--image", default=None, help="Path to a real image file")
    parser.add_argument("--random", action="store_true",
                        help="Use random noise image (connectivity test only)")
    args = parser.parse_args()

    # ── Health check ──────────────────────────────────────────────────────────
    health_url = f"http://{args.host}:{args.port}/health"
    try:
        h = requests.get(health_url, timeout=5)
        h.raise_for_status()
        print(f"[Health] {h.json()}")
    except Exception as exc:
        print(f"[Health] FAILED: {exc}")
        sys.exit(1)

    # ── Prepare image ─────────────────────────────────────────────────────────
    if args.random:
        image_bytes = make_random_image()
        print("[Image] Using random 640×480 noise image")
    elif args.image:
        image_bytes = load_image(args.image)
        print(f"[Image] Loaded from {args.image}")
    else:
        print("[Error] Provide --image <path> or --random")
        sys.exit(1)

    # ── Call /s2_step ─────────────────────────────────────────────────────────
    print(f"\n[Request] instruction = \"{args.instruction}\"")
    import time
    t0 = time.time()
    result = call_s2(args.host, args.port, image_bytes, args.instruction)
    elapsed = time.time() - t0

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n[Result]  (latency: {elapsed:.2f}s)")
    print(f"  raw            : {result.get('raw', '')!r}")
    print(f"  target         : {result.get('target')}")
    print(f"  point_2d_norm  : {result.get('point_2d_norm')}   (scale [0,1000])")
    print(f"  point_2d_pixel : {result.get('point_2d_pixel')}  (image pixels)")
    print(f"  navigation     : {result.get('navigation')!r}")

    if "error" in result:
        print(f"\n[ERROR] {result['error']}")
        if "traceback" in result:
            print(result["traceback"])


if __name__ == "__main__":
    main()
