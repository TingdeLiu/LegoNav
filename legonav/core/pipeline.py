#!/usr/bin/env python3
"""
LegoNav Pipeline — S2 (Qwen3-VL) + S1 (NavDP pixelgoal) 联合推理

S2 服务器 (wheeltec_s2_server.py) 分析图像+指令，输出像素目标或旋转指令；
S1 服务器 (navdp_server.py) 接收像素目标，输出轨迹。

数据流:
    RGB (BGR numpy) + Depth (float32, 米) + 导航指令
        │
        ▼  POST /s2_step
    S2 Qwen3-VL
        │ {"target", "point_2d_pixel", "navigation", ...}
        │
        ├─ pixel_goal ──▶ POST /pixelgoal_step ──▶ S1 NavDP ──▶ trajectory (m)
        ├─ turn       ──▶ rotation_rad (直接返回旋转角度，跳过 S1)
        └─ stop       ──▶ stop 信号

用法:
    # 独立测试（不需要真实机器人）
    python legonav_pipeline.py \
        --s2_host 127.0.0.1 --s2_port 8890 \
        --s1_host 127.0.0.1 --s1_port 8901 \
        --image /path/to/test.jpg \
        --instruction "Go to the red chair"
"""

import io
import json
import math

import cv2
import numpy as np
import requests
from PIL import Image

from legonav.clients.navdp_client import NavDPClient

# Gemini 336L 相机内参 (1280×720) — 当前默认相机
# fx=607.45, fy=607.40, cx=639.19, cy=361.75
GEMINI_336L_INTRINSIC = np.array([
    [607.45, 0.0, 639.19, 0.0],
    [0.0, 607.40, 361.75, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# Astra S 相机内参 (640×480) — 备用，切换时将下方默认值改为此常量
# fx=570.3, fy=570.3, cx=319.5, cy=239.5
ASTRA_S_INTRINSIC = np.array([
    [570.3, 0.0, 319.5, 0.0],
    [0.0, 570.3, 239.5, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# S2 导航符号 → 角度
_DEG_PER_ARROW = 15.0  # 每个箭头对应 15°


class LegoNavPipeline:
    """
    LegoNav 双系统推理管线。

    S1 支持两种模式：
      - 服务器模式（默认）：通过 HTTP 调用远端 NavDP 服务器
      - 本地模式：直接在当前进程加载 NavDP 模型（端侧部署，无网络依赖）

    参数:
        s2_host / s2_port   : S2 Qwen3-VL 服务器地址
        s1_host / s1_port   : S1 NavDP 服务器地址（本地模式时忽略）
        s1_client           : 若传入，直接使用该客户端（NavDPLocalClient 或自定义）；
                              为 None 时自动创建 NavDPClient(s1_host, s1_port)
        camera_intrinsic    : (4,4) 相机内参矩阵（默认 Gemini 336L 1280×720）
        s2_timeout          : S2 HTTP 超时秒数
        s1_timeout          : S1 HTTP 超时秒数（本地模式时不使用）
    """

    def __init__(
        self,
        s2_host: str = "127.0.0.1",
        s2_port: int = 8890,
        s1_host: str = "127.0.0.1",
        s1_port: int = 8901,
        s1_client=None,
        camera_intrinsic: np.ndarray = None,
        s2_timeout: float = 30.0,
        s1_timeout: float = 10.0,
    ):
        self.s2_url = f"http://{s2_host}:{s2_port}/s2_step"
        self.s2_timeout = s2_timeout

        # s1_client 优先：支持本地推理（NavDPLocalClient）或自定义实现
        self.navdp = s1_client if s1_client is not None else NavDPClient(host=s1_host, port=s1_port)
        self.s1_timeout = s1_timeout

        self.camera_intrinsic = (
            camera_intrinsic if camera_intrinsic is not None else GEMINI_336L_INTRINSIC
        )
        self.instruction: str = ""
        self._s1_initialized: bool = False
        self._task_queue: list = []         # 顺序任务队列（S2 填充，逐个弹出执行）
        self._stop_threshold: float = -3.0  # NavDP Critic 到达阈值
        self._tasks_loaded: bool = False    # 本 episode 是否已加载过任务

    # ──────────────────────────────────────────────────────────────────────────
    # Episode control
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, instruction: str, stop_threshold: float = -3.0) -> None:
        """新 episode 开始时调用：设置指令，重置 S1 记忆队列。"""
        self.instruction = instruction
        self._task_queue = []
        self._stop_threshold = stop_threshold
        self._tasks_loaded = False
        self.navdp.reset(
            camera_intrinsic=self.camera_intrinsic,
            batch_size=1,
            stop_threshold=stop_threshold,
        )
        self._s1_initialized = True
        print(f"[LegoNav] Reset. instruction='{instruction}'", flush=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Main step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, rgb_bgr: np.ndarray, depth_m: np.ndarray) -> dict:
        """
        单步推理。

        参数:
            rgb_bgr  : (H, W, 3) uint8, BGR 格式（与 OpenCV 一致）
            depth_m  : (H, W) float32, 深度单位为米

        返回 dict，key "mode" 决定后续控制方式：
            mode="trajectory"
                trajectory   : np.ndarray (1, 24, 3)，相对位移轨迹（米）
                               轨迹坐标系: x=前, y=左, z=上
                all_trajectory: np.ndarray (1, N, 24, 3)，所有候选轨迹
                values       : np.ndarray (1, N)，Critic 评分
                s2           : S2 原始响应 dict

            mode="rotate"
                rotation_rad : float，正值=逆时针(左转)，负值=顺时针(右转)
                s2           : S2 原始响应 dict

            mode="stop"
                s2           : S2 原始响应 dict

            mode="error"
                message      : str
                s2           : S2 原始响应 dict（若有）
        """
        if not self._s1_initialized:
            return {"mode": "error", "message": "call reset() before step()"}

        # ── 所有任务已完成 ────────────────────────────────────────────────────
        if self._tasks_loaded and not self._task_queue:
            return {"mode": "stop"}

        # ── 队列为空时查询 S2 填充任务 ────────────────────────────────────────
        if not self._task_queue:
            try:
                s2_result = self._call_s2(rgb_bgr, self.instruction)
            except Exception as exc:
                return {"mode": "error", "message": f"S2 request failed: {exc}"}
            self._populate_task_queue(s2_result)
            self._tasks_loaded = True
            if not self._task_queue:
                return {
                    "mode": "error",
                    "message": "S2 returned no actionable tasks",
                    "s2": s2_result,
                }

        task = self._task_queue[0]

        # ── stop 任务 ─────────────────────────────────────────────────────────
        if task["type"] == "stop":
            self._task_queue.pop(0)
            return {"mode": "stop"}

        # ── move 任务（旋转）──────────────────────────────────────────────────
        if task["type"] == "move":
            rad = task["rotation_rad"]
            self._task_queue.pop(0)
            print(
                f"[LegoNav] task=move | {task['symbols']!r} → {math.degrees(rad):.1f}° "
                f"| remaining={len(self._task_queue)}",
                flush=True,
            )
            return {"mode": "rotate", "rotation_rad": rad}

        # ── pixel_point 任务（目标导航）───────────────────────────────────────
        if task["type"] == "pixel_point":
            target = task["target"]
            pixel  = task.get("pixel")   # 队列填充时存入的初始像素坐标

            # 目标可见：直接调 S1，无需刷新 S2
            # NavDP memory queue 能处理机器人移动后的坐标偏差
            if pixel is not None:
                try:
                    traj, all_traj, values = self._call_s1_pixelgoal(
                        rgb_bgr, depth_m, pixel
                    )
                except Exception as exc:
                    return {"mode": "error", "message": f"S1 request failed: {exc}"}

                # NavDP Critic 判定到达 → 弹出任务，下一步自动执行后续任务
                if float(values.max()) < self._stop_threshold:
                    self._task_queue.pop(0)
                    print(
                        f"[LegoNav] task=pixel_point done (NavDP Critic) | target={target!r} "
                        f"| remaining={len(self._task_queue)}",
                        flush=True,
                    )

                return {
                    "mode": "trajectory",
                    "trajectory": traj,
                    "all_trajectory": all_traj,
                    "values": values,
                    "target": target,
                    "pixel": pixel,
                }

            # 目标初始不可见 → 固定搜索旋转，依赖 NavDP 能力完成导航
            print(
                f"[LegoNav] task=pixel_point | target={target!r} pixel=None → search rotate",
                flush=True,
            )
            return {"mode": "rotate", "rotation_rad": math.radians(15)}

        return {"mode": "error", "message": f"Unknown task type: {task['type']}"}

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _call_s2(self, rgb_bgr: np.ndarray, instruction: str) -> dict:
        """将 BGR numpy 图像转 JPEG，POST 到 S2 服务器。"""
        # BGR → RGB，编码为 JPEG
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)

        resp = requests.post(
            self.s2_url,
            files={"image": ("frame.jpg", buf.getvalue(), "image/jpeg")},
            data={"instruction": instruction},
            timeout=self.s2_timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _call_s1_pixelgoal(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        pixel: list,
    ):
        """调用 NavDP /pixelgoal_step，返回 (traj, all_traj, values)。"""
        pixel_goals = np.array([[pixel[0], pixel[1]]], dtype=np.float32)   # (1, 2)
        rgb_batch   = rgb_bgr[np.newaxis]                                  # (1, H, W, 3)
        depth_batch = depth_m[np.newaxis, :, :, np.newaxis]               # (1, H, W, 1)
        return self.navdp.pixelgoal_step(pixel_goals, rgb_batch, depth_batch)

    # ──────────────────────────────────────────────────────────────────────────
    # Task queue management
    # ──────────────────────────────────────────────────────────────────────────

    def _populate_task_queue(self, s2_result: dict) -> None:
        """将 S2 响应中的结构化任务列表解析为顺序任务队列。

        支持新格式（含 tasks 字段）和旧版服务器兼容（无 tasks 字段）。
        """
        raw_tasks = s2_result.get("tasks") or []
        queue: list = []

        for task in raw_tasks:
            task_type = task.get("task")
            if task_type == "pixel_point":
                target = task.get("target")
                if target:
                    queue.append({
                        "type": "pixel_point",
                        "target": target,
                        "pixel": task.get("point_2d_pixel"),  # None 表示初始不可见
                    })
            elif task_type == "move":
                action = task.get("action", "")
                number = max(1, int(task.get("number", 1)))
                if action == "stop":
                    queue.append({"type": "stop"})
                elif action in ("←", "→"):
                    symbols = action * number
                    queue.append({
                        "type": "move",
                        "symbols": symbols,
                        "rotation_rad": _parse_rotation(symbols),
                    })
                # ↑ ↓ 暂不实现，跳过

        # 旧版服务器兼容：无 tasks 字段时退化为单任务逻辑
        if not queue:
            queue = self._legacy_task_queue(s2_result)

        self._task_queue = queue
        print(
            f"[LegoNav] task queue ({len(queue)}): {[t['type'] for t in queue]}",
            flush=True,
        )

    def _legacy_task_queue(self, s2_result: dict) -> list:
        """兼容旧版服务器（无 tasks 字段）：从折叠字段重建单任务队列。"""
        nav    = s2_result.get("navigation", "")
        pixel  = s2_result.get("point_2d_pixel")
        target = s2_result.get("target")

        if "stop" in nav:
            return [{"type": "stop"}]
        if pixel is not None and target:
            return [{"type": "pixel_point", "target": target}]
        if nav and any(c in nav for c in ("←", "→")):
            return [{"type": "move", "symbols": nav, "rotation_rad": _parse_rotation(nav)}]
        if target:
            return [{"type": "pixel_point", "target": target}]
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_rotation(nav_str: str) -> float:
    """
    将 S2 导航符号串解析为旋转弧度。
    返回值: 正 = 左转(逆时针), 负 = 右转(顺时针)
    """
    left  = nav_str.count("←")
    right = nav_str.count("→")
    return math.radians((left - right) * _DEG_PER_ARROW)


def traj_to_first_waypoint(traj: np.ndarray) -> tuple:
    """
    从轨迹中提取第一步的线速度/角速度参考值（用于 MPC/PID 控制）。

    参数:
        traj: (1, T, 3) 轨迹，坐标系 x=前, y=左

    返回:
        (vx, vy_left)  线速度(前向, m/step)，横向偏移(左, m/step)
    """
    wp = traj[0, 0]      # 第一个轨迹点 [dx, dy, dz]
    return float(wp[0]), float(wp[1])


def health_check(s2_host: str, s2_port: int, s1_host: str, s1_port: int) -> bool:
    """检查 S2 和 S1 服务器是否在线。"""
    ok = True
    for name, url in [
        ("S2", f"http://{s2_host}:{s2_port}/health"),
        ("S1", f"http://{s1_host}:{s1_port}/navigator_reset"),
    ]:
        try:
            if name == "S2":
                r = requests.get(url, timeout=3)
                r.raise_for_status()
                print(f"  [{name}] OK  {r.json()}")
            else:
                # S1 没有 /health，尝试 /navigator_reset 用虚假 intrinsic
                # 只测连通性，用 GET（会返回 405，但说明服务在线）
                r = requests.get(f"http://{s1_host}:{s1_port}/", timeout=3)
                print(f"  [S1] reachable (status {r.status_code})")
        except requests.exceptions.ConnectionError:
            print(f"  [{name}] OFFLINE — {url}")
            ok = False
        except Exception as exc:
            print(f"  [{name}] {exc}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# CLI 快速测试
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_inputs(width=640, height=480):
    rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    depth = np.ones((height, width), dtype=np.float32) * 2.0
    return rgb, depth


def _load_inputs(image_path: str, width=640, height=480):
    img = Image.open(image_path).convert("RGB").resize((width, height))
    rgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    depth = np.ones((height, width), dtype=np.float32) * 2.0  # 假深度 2m
    return rgb, depth


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="LegoNav Pipeline 快速测试")
    parser.add_argument("--s2_host", default="127.0.0.1")
    parser.add_argument("--s2_port", type=int, default=8890)
    parser.add_argument("--s1_host", default="127.0.0.1")
    parser.add_argument("--s1_port", type=int, default=8901)
    parser.add_argument("--instruction", default="Go to the red chair")
    parser.add_argument("--image", default=None, help="真实图片路径")
    parser.add_argument("--random", action="store_true",
                        help="使用随机噪声图像（连通性测试，不验证识别结果）")
    parser.add_argument("--skip_s1", action="store_true",
                        help="跳过 S1 调用（仅测试 S2，NavDP 未启动时使用）")
    args = parser.parse_args()

    if not args.image and not args.random:
        parser.error("请提供 --image <路径> 或 --random")

    print("=" * 55)
    print("  LegoNav Pipeline — 快速测试")
    print("=" * 55)

    # ── 健康检查 ──────────────────────────────────────────────────────────────
    print("\n[1] 服务器连通性检查 …")
    health_check(args.s2_host, args.s2_port, args.s1_host, args.s1_port)

    # ── 初始化管线 ────────────────────────────────────────────────────────────
    pipeline = LegoNavPipeline(
        s2_host=args.s2_host, s2_port=args.s2_port,
        s1_host=args.s1_host, s1_port=args.s1_port,
    )

    if args.skip_s1:
        # 跳过 S1 reset（NavDP 未启动时使用）
        pipeline.instruction = args.instruction
        pipeline._s1_initialized = True
        print("[!] --skip_s1 模式：跳过 NavDP reset，S1 调用将会失败（预期）")
    else:
        print("\n[2] Reset pipeline …")
        pipeline.reset(args.instruction)

    # ── 准备图像 ──────────────────────────────────────────────────────────────
    if args.random:
        rgb, depth = _make_fake_inputs()
        print("\n[3] 图像来源: 随机噪声 640×480（S2 目标检测结果仅供连通性参考）")
    else:
        rgb, depth = _load_inputs(args.image)
        print(f"\n[3] 图像来源: {args.image}")

    # ── 单步推理 ──────────────────────────────────────────────────────────────
    print(f"\n[4] 执行 step(), instruction='{args.instruction}' …")
    t0 = time.time()
    result = pipeline.step(rgb, depth)
    elapsed = time.time() - t0

    # ── 打印结果 ──────────────────────────────────────────────────────────────
    print(f"\n[Result]  (总耗时 {elapsed:.2f}s)")
    print(f"  mode          : {result['mode']}")

    s2 = result.get("s2", {})
    print(f"  S2 target     : {s2.get('target')}")
    print(f"  S2 pixel_norm : {s2.get('point_2d_norm')}")
    print(f"  S2 pixel_px   : {s2.get('point_2d_pixel')}")
    print(f"  S2 navigation : {s2.get('navigation')!r}")
    print(f"  S2 raw        : {s2.get('raw', '')!r}")

    if result["mode"] == "trajectory":
        traj = result["trajectory"]
        print(f"  S1 traj shape : {traj.shape}")
        print(f"  S1 traj[0,0]  : {traj[0, 0]}  (first waypoint, meters)")
        print(f"  S1 values max : {result['values'].max():.3f}")
    elif result["mode"] == "rotate":
        deg = math.degrees(result["rotation_rad"])
        print(f"  rotation      : {result['rotation_rad']:.4f} rad  ({deg:.1f}°)")
    elif result["mode"] == "error":
        print(f"  [ERROR] {result.get('message')}")
