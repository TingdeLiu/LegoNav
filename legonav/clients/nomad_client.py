"""
NoMaD (No Maps required Diffusion) S1 HTTP 客户端

论文: NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration (ICRA 2024)
项目: https://github.com/robodhruv/visualnav-transformer

NoMaD 基于扩散策略，支持有目标（imagegoal）和无目标（exploration）两种模式。
服务端协议与 GNM/ViNT 基本一致，额外支持 goal_mask 参数以切换模式。

服务端启动:
  python deployment/src/navigate.py --model nomad --port 8048

HTTP 协议 (在 GNM 基础上扩展):
  POST /navigate
      files: obs (JPEG), goal (JPEG, 可选)
      data:
        goal_mode  : "image" | "none"
        num_samples: 扩散采样数（默认 8，越多越慢但越好）
      response:
        {"waypoints": [[x1,y1], ...],     # 最优轨迹路标点
         "all_waypoints": [[[x,y],...], ...],  # N 组候选路标 (可选)
         "temporal_dist": 4.1}
"""

import cv2
import numpy as np
import requests

from legonav.clients.base_client import BaseS1Client


class NoMaDClient(BaseS1Client):
    """NoMaD S1 HTTP 客户端。

    Args:
        host         : 服务器地址
        port         : 服务器端口（默认 8048，与 GNM/ViNT 区分）
        timeout      : HTTP 超时秒数
        waypoint_T   : 输出轨迹步数
        num_samples  : 扩散采样数（越多轨迹质量越高，延迟越大）
        stop_dist    : 到达距离阈值（米）
    """

    algo_name = "nomad"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8048,
        timeout: float = 15.0,
        waypoint_T: int = 8,
        num_samples: int = 8,
        stop_dist: float = 0.3,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.waypoint_T = waypoint_T
        self.num_samples = num_samples
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
            print(f"[NoMaDClient] reset failed: {exc}", flush=True)
            return self.algo_name

    def _navigate(
        self,
        rgb_bgr: np.ndarray,
        goal_bgr: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        B = rgb_bgr.shape[0]

        _, obs_enc = cv2.imencode(".jpg", rgb_bgr[0])
        files: dict = {"obs": ("obs.jpg", obs_enc.tobytes(), "image/jpeg")}
        data = {"num_samples": self.num_samples}

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

        # 最优路标点
        best_wps = np.array(result["waypoints"], dtype=np.float32)  # (K, 2)
        if best_wps.ndim == 1:
            best_wps = best_wps[np.newaxis]

        # 候选轨迹（NoMaD 扩散采样，可选字段）
        raw_all = result.get("all_waypoints")
        if raw_all:
            all_wps = np.array(raw_all, dtype=np.float32)   # (N, K, 2)
            N = all_wps.shape[0]
            all_traj = np.zeros((B, N, self.waypoint_T, 3), dtype=np.float32)
            for i in range(N):
                wps_b = np.tile(all_wps[i][np.newaxis], (B, 1, 1))
                all_traj[:, i] = self._waypoints_to_trajectory(wps_b, T=self.waypoint_T)
            values = np.zeros((B, N), dtype=np.float32)
        else:
            all_traj = None
            values   = None

        best_wps_b = np.tile(best_wps[np.newaxis], (B, 1, 1))
        traj = self._waypoints_to_trajectory(best_wps_b, T=self.waypoint_T)

        if all_traj is None:
            traj, all_traj, values = self._wrap_single_trajectory(traj)

        # 到达检测
        first_dist = float(np.linalg.norm(best_wps[0]))
        if first_dist < self.stop_dist:
            values[:] = -10.0

        return traj, all_traj, values

    def pixelgoal_step(self, pixel_goals, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=None)

    def imagegoal_step(self, goal_images, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=goal_images)

    def nogoal_step(self, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=None)

    def pointgoal_step(self, point_goals, rgb_images, depth_images):
        return self._navigate(rgb_images, goal_bgr=None)
