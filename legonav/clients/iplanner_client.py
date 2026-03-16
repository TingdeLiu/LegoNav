"""
iPlanner S1 HTTP 客户端

论文: iPlanner: Imperative Path Planning (RSS 2023)
项目: https://github.com/ZhuangYanDLUT/iPlanner

iPlanner 是一个端到端的局部路径规划器，直接从 RGB 图像预测无碰撞轨迹。
输入: RGB 图像 + 目标点（相机坐标系）
输出: 局部路径（若干路标点，相机坐标系 [x=前, y=左]）

服务端 HTTP 协议 (需自行实现 Flask/FastAPI 包装 iPlanner):
  POST /reset
      body: {}
      response: {"algo": "iplanner"}

  POST /plan
      files: image (JPEG RGB)
      data:
        goal_data: JSON {"x": 3.0, "y": 0.5, "z": 0.0}   # 目标，相机坐标系
      response:
        {"trajectory": [[x1,y1,z1], [x2,y2,z2], ...],    # K 个路标点
         "cost": 0.12}                                    # 规划代价（可选）

参考部署:
  - https://github.com/ZhuangYanDLUT/iPlanner#deploy
  - iPlanner 原生支持 ROS；HTTP 包装参考 examples/iPlanner_deploy.py
"""

import json

import cv2
import numpy as np
import requests

from legonav.clients.base_client import BaseS1Client


class iPlannerClient(BaseS1Client):
    """iPlanner S1 HTTP 客户端。

    Args:
        host       : 服务器地址
        port       : 服务器端口（默认 8903）
        timeout    : HTTP 超时秒数
        waypoint_T : 输出轨迹步数
        stop_dist  : 到达距离阈值（米），轨迹第一点 < stop_dist 时触发停止
    """

    algo_name = "iplanner"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8903,
        timeout: float = 5.0,
        waypoint_T: int = 8,
        stop_dist: float = 0.3,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.waypoint_T = waypoint_T
        self.stop_dist = stop_dist

    def reset(
        self,
        camera_intrinsic: np.ndarray,
        batch_size: int = 1,
        stop_threshold: float = -3.0,
    ) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/reset", json={}, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json().get("algo", self.algo_name)
        except Exception as exc:
            print(f"[iPlannerClient] reset failed: {exc}", flush=True)
            return self.algo_name

    def _plan(
        self,
        rgb_bgr: np.ndarray,
        goal_xyz: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = rgb_bgr.shape[0]

        _, img_enc = cv2.imencode(".jpg", rgb_bgr[0])
        resp = requests.post(
            f"{self.base_url}/plan",
            files={"image": ("image.jpg", img_enc.tobytes(), "image/jpeg")},
            data={"goal_data": json.dumps(goal_xyz)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        pts = np.array(result["trajectory"], dtype=np.float32)  # (K, 2 or 3)
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        if pts.shape[1] == 2:
            pts = np.concatenate([pts, np.zeros((len(pts), 1), dtype=np.float32)], axis=1)

        pts_batch = np.tile(pts[np.newaxis, :, :2], (B, 1, 1))
        traj = self._waypoints_to_trajectory(pts_batch, T=self.waypoint_T)

        first_dist = float(np.linalg.norm(pts[0, :2]))
        score = -10.0 if first_dist < self.stop_dist else 0.0
        return self._wrap_single_trajectory_scored(traj, score, B)

    @staticmethod
    def _wrap_single_trajectory_scored(traj, score, B):
        all_traj = traj[:, np.newaxis]
        values   = np.full((B, 1), score, dtype=np.float32)
        return traj, all_traj, values

    # ──────────────────────────────────────────────────────────────────────────

    def pixelgoal_step(self, pixel_goals, rgb_images, depth_images):
        """像素目标 → 通过深度图估算 3D 目标点。"""
        if depth_images.ndim == 4:
            d = depth_images[0, :, :, 0]
        else:
            d = depth_images[0]
        H, W = d.shape
        u, v = int(pixel_goals[0, 0]), int(pixel_goals[0, 1])
        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))
        depth = float(d[v, u])
        if depth < 0.1 or depth > 20.0:
            depth = 2.0
        goal = {"x": depth, "y": -(u - W / 2) / W * depth, "z": 0.0}
        return self._plan(rgb_images, goal)

    def pointgoal_step(self, point_goals, rgb_images, depth_images):
        goal = {
            "x": float(point_goals[0, 0]),
            "y": float(point_goals[0, 1]),
            "z": float(point_goals[0, 2]) if point_goals.shape[1] > 2 else 0.0,
        }
        return self._plan(rgb_images, goal)

    def nogoal_step(self, rgb_images, depth_images):
        return self._plan(rgb_images, {"x": 5.0, "y": 0.0, "z": 0.0})

    def imagegoal_step(self, goal_images, rgb_images, depth_images):
        # iPlanner 不支持图像目标，退化为 nogoal
        return self.nogoal_step(rgb_images, depth_images)
