"""
ViPlanner S1 HTTP 客户端

论文: ViPlanner: Visual Semantic Imperative Learning for Local Navigation (ICRA 2024)
项目: https://github.com/leggedrobotics/viplanner

ViPlanner 结合视觉语义信息（RGB + 可选语义分割）与目标点，
生成适用于室外非结构化地形的局部轨迹。

与 iPlanner 接口相似，但额外支持语义图像输入。

服务端 HTTP 协议:
  POST /reset
      body: {}
      response: {"algo": "viplanner"}

  POST /plan
      files:
        image    : JPEG (RGB)
        semantic : JPEG (可选，语义分割伪彩图)
        depth    : PNG uint16 (可选)
      data:
        goal_data: JSON {"x": 3.0, "y": 0.5, "z": 0.0}
      response:
        {"trajectory": [[x1,y1,z1], ...],
         "cost": 0.05}

参考:
  - https://github.com/leggedrobotics/viplanner#deployment
"""

import json

import cv2
import numpy as np
import requests

from legonav.clients.base_client import BaseS1Client


class ViPlannerClient(BaseS1Client):
    """ViPlanner S1 HTTP 客户端。

    Args:
        host          : 服务器地址
        port          : 服务器端口（默认 8904）
        timeout       : HTTP 超时秒数
        waypoint_T    : 输出轨迹步数
        stop_dist     : 到达距离阈值（米）
        send_depth    : 是否发送深度图（服务端支持时开启）
        send_semantic : 是否发送语义分割图（需要外部语义分割前处理）
    """

    algo_name = "viplanner"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8904,
        timeout: float = 10.0,
        waypoint_T: int = 8,
        stop_dist: float = 0.3,
        send_depth: bool = False,
        send_semantic: bool = False,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.waypoint_T = waypoint_T
        self.stop_dist = stop_dist
        self.send_depth = send_depth
        self.send_semantic = send_semantic

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
            print(f"[ViPlannerClient] reset failed: {exc}", flush=True)
            return self.algo_name

    def _plan(
        self,
        rgb_bgr: np.ndarray,
        depth_images: np.ndarray | None,
        goal_xyz: dict,
        semantic: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = rgb_bgr.shape[0]

        _, img_enc = cv2.imencode(".jpg", rgb_bgr[0])
        files: dict = {"image": ("image.jpg", img_enc.tobytes(), "image/jpeg")}

        if self.send_depth and depth_images is not None:
            d = depth_images[0, :, :, 0] if depth_images.ndim == 4 else depth_images[0]
            d_u16 = np.clip(d * 10000, 0, 65535).astype(np.uint16)
            _, d_enc = cv2.imencode(".png", d_u16)
            files["depth"] = ("depth.png", d_enc.tobytes(), "image/png")

        if self.send_semantic and semantic is not None:
            _, sem_enc = cv2.imencode(".jpg", semantic[0])
            files["semantic"] = ("semantic.jpg", sem_enc.tobytes(), "image/jpeg")

        resp = requests.post(
            f"{self.base_url}/plan",
            files=files,
            data={"goal_data": json.dumps(goal_xyz)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        pts = np.array(result["trajectory"], dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[np.newaxis]
        if pts.shape[1] == 2:
            pts = np.concatenate([pts, np.zeros((len(pts), 1), dtype=np.float32)], axis=1)

        pts_batch = np.tile(pts[np.newaxis, :, :2], (B, 1, 1))
        traj = self._waypoints_to_trajectory(pts_batch, T=self.waypoint_T)

        first_dist = float(np.linalg.norm(pts[0, :2]))
        score = -10.0 if first_dist < self.stop_dist else 0.0
        all_traj = traj[:, np.newaxis]
        values   = np.full((B, 1), score, dtype=np.float32)
        return traj, all_traj, values

    # ──────────────────────────────────────────────────────────────────────────

    def pixelgoal_step(self, pixel_goals, rgb_images, depth_images):
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
        return self._plan(rgb_images, depth_images, goal)

    def pointgoal_step(self, point_goals, rgb_images, depth_images):
        goal = {
            "x": float(point_goals[0, 0]),
            "y": float(point_goals[0, 1]),
            "z": float(point_goals[0, 2]) if point_goals.shape[1] > 2 else 0.0,
        }
        return self._plan(rgb_images, depth_images, goal)

    def nogoal_step(self, rgb_images, depth_images):
        return self._plan(rgb_images, depth_images, {"x": 5.0, "y": 0.0, "z": 0.0})

    def imagegoal_step(self, goal_images, rgb_images, depth_images):
        return self.nogoal_step(rgb_images, depth_images)
