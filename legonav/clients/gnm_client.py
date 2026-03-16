"""
GNM (General Navigation Model) S1 HTTP 客户端

论文: GNM: A General Navigation Model to Drive Any Robot (ICRA 2023)
项目: https://github.com/robodhruv/visualnav-transformer

GNM 接收当前 RGB 图像序列 + 目标图像，输出前向路标点（waypoints）。
本客户端同时兼容 **ViNT** 模型（相同服务端协议，模型更强）。

服务端启动 (visualnav-transformer):
  cd visualnav-transformer
  python deployment/src/navigate.py \
      --model gnm  \          # 或 --model vint
      --dir <goal_image_dir> \
      --port 8047

HTTP 协议 (服务端需实现):
  POST /reset
      body: {}
      response: {"algo": "gnm"}

  POST /navigate
      files:
        obs  : JPEG  当前观测帧（最新一帧，服务端维护历史队列）
        goal : JPEG  目标参考图（imagegoal 模式）
      data:
        goal_mode : "image"（图像目标）或 "none"（自主探索）
      response:
        {"waypoints": [[x1,y1], [x2,y2], ...],   # K 个路标点，相机坐标系
         "temporal_dist": 5.2}                    # 估计到达步数（可选）
"""

import base64
import io

import cv2
import numpy as np
import requests

from legonav.clients.base_client import BaseS1Client


class GNMClient(BaseS1Client):
    """GNM / ViNT S1 HTTP 客户端。

    Args:
        host         : 服务器地址（默认 127.0.0.1）
        port         : 服务器端口（默认 8047）
        timeout      : HTTP 超时秒数
        waypoint_T   : 轨迹输出步数（填充/截断到此长度，默认 8）
        stop_dist    : 到达判断距离（米），路标第一点 < stop_dist 时触发停止
    """

    algo_name = "gnm"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8047,
        timeout: float = 10.0,
        waypoint_T: int = 8,
        stop_dist: float = 0.3,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.waypoint_T = waypoint_T
        self.stop_dist = stop_dist
        self._stop_threshold = -3.0  # 由 reset() 设置

    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        camera_intrinsic: np.ndarray,
        batch_size: int = 1,
        stop_threshold: float = -3.0,
    ) -> str:
        self._stop_threshold = stop_threshold
        try:
            resp = requests.post(
                f"{self.base_url}/reset", json={}, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("algo", self.algo_name)
        except Exception as exc:
            print(f"[GNMClient] reset failed: {exc}", flush=True)
            return self.algo_name

    # ──────────────────────────────────────────────────────────────────────────

    def _navigate(
        self,
        rgb_bgr: np.ndarray,
        goal_bgr: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """调用服务端 /navigate，返回标准三元组。"""
        B = rgb_bgr.shape[0]

        # 只取 batch[0]（GNM 单帧推理，服务端自维护历史）
        _, obs_enc = cv2.imencode(".jpg", rgb_bgr[0])
        files: dict = {"obs": ("obs.jpg", obs_enc.tobytes(), "image/jpeg")}
        data: dict = {}

        if goal_bgr is not None:
            _, goal_enc = cv2.imencode(".jpg", goal_bgr[0])
            files["goal"] = ("goal.jpg", goal_enc.tobytes(), "image/jpeg")
            data["goal_mode"] = "image"
        else:
            data["goal_mode"] = "none"

        resp = requests.post(
            f"{self.base_url}/navigate",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        waypoints = np.array(result["waypoints"], dtype=np.float32)  # (K, 2)
        if waypoints.ndim == 1:
            waypoints = waypoints[np.newaxis]  # 保证 (K, 2)

        # 广播到 batch 维度 (B, K, 2)
        waypoints_batch = np.tile(waypoints[np.newaxis], (B, 1, 1))

        traj = self._waypoints_to_trajectory(waypoints_batch, T=self.waypoint_T)

        # 到达检测：第一个路标点距离 < stop_dist 时给出负分
        first_dist = float(np.linalg.norm(waypoints[0]))
        score = -10.0 if first_dist < self.stop_dist else 0.0
        all_traj = traj[:, np.newaxis]
        values   = np.full((B, 1), score, dtype=np.float32)

        return traj, all_traj, values

    # ──────────────────────────────────────────────────────────────────────────

    def pixelgoal_step(self, pixel_goals, rgb_images, depth_images):
        """像素目标 → 从深度图估计前向距离，退化为 nogoal 模式（GNM 无像素目标概念）。"""
        return self._navigate(rgb_images, goal_bgr=None)

    def imagegoal_step(self, goal_images, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=goal_images)

    def nogoal_step(self, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=None)

    def pointgoal_step(self, point_goals, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=None)
