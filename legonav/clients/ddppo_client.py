"""
DD-PPO (Decentralized Distributed Proximal Policy Optimization) S1 HTTP 客户端

论文: DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames (ICLR 2020)
来源: Habitat Challenge 官方 baseline (Meta AI)
权重: https://github.com/facebookresearch/habitat-lab/tree/main/habitat-baselines

DD-PPO 在 Habitat 仿真器中训练，输出离散导航动作（stop/forward/left/right）。
本客户端将离散动作转换为 LegoNav 轨迹格式。

停止判断: action=STOP（0）时返回 values=-10，触发 pipeline 停止。

服务端 HTTP 协议 (需自行包装 Habitat policy):
  POST /reset
      body: {"goal_distance": 1.5, "goal_heading": 0.0}
      response: {"algo": "ddppo"}

  POST /step
      files: image (JPEG RGB 256×256), depth (PNG uint16 256×256)
      data:
        goal_data: JSON {"distance": 1.5, "heading": 0.3}  # 到目标的极坐标
      response:
        {"action": 1,               # 0=STOP 1=FORWARD 2=LEFT 3=RIGHT
         "distance_to_goal": 1.2,   # 剩余距离（可选）
         "heading_to_goal": 0.15}   # 剩余方向（可选）

参考实现:
  - habitat-lab/habitat_baselines/agents/ppo_agents.py
  - 可包装为 Flask server 使用本客户端
"""

import json

import cv2
import numpy as np
import requests

from legonav.clients.base_client import BaseS1Client

# Habitat 标准离散动作
_STOP    = 0
_FORWARD = 1
_LEFT    = 2
_RIGHT   = 3


class DDPPOClient(BaseS1Client):
    """DD-PPO S1 HTTP 客户端。

    Args:
        host         : 服务器地址
        port         : 服务器端口（默认 8902）
        timeout      : HTTP 超时秒数
        step_m       : FORWARD 动作对应的前进距离（米，默认 0.25）
        turn_deg     : LEFT/RIGHT 动作对应的转角（度，默认 15）
        waypoint_T   : 轨迹步数（离散动作转轨迹时重复填充的长度）
        image_size   : 送入 DD-PPO 模型的图像尺寸（默认 256）
    """

    algo_name = "ddppo"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8902,
        timeout: float = 5.0,
        step_m: float = 0.25,
        turn_deg: float = 15.0,
        waypoint_T: int = 8,
        image_size: int = 256,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.step_m = step_m
        self.turn_rad = float(np.deg2rad(turn_deg))
        self.waypoint_T = waypoint_T
        self.image_size = image_size
        self._goal_distance: float = 1.5
        self._goal_heading: float = 0.0

    def reset(
        self,
        camera_intrinsic: np.ndarray,
        batch_size: int = 1,
        stop_threshold: float = -3.0,
    ) -> str:
        try:
            resp = requests.post(
                f"{self.base_url}/reset",
                json={
                    "goal_distance": self._goal_distance,
                    "goal_heading":  self._goal_heading,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("algo", self.algo_name)
        except Exception as exc:
            print(f"[DDPPOClient] reset failed: {exc}", flush=True)
            return self.algo_name

    # ──────────────────────────────────────────────────────────────────────────

    def _call_step(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
        goal_data: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """通用推理调用，返回标准三元组。"""
        B = rgb_bgr.shape[0]
        trajs, all_trajs, vals = [], [], []

        for i in range(B):
            img = cv2.resize(rgb_bgr[i], (self.image_size, self.image_size))
            # Depth 转 uint16 PNG
            if depth_m.ndim == 4:
                d = depth_m[i, :, :, 0]
            else:
                d = depth_m[i]
            d_resized = cv2.resize(d, (self.image_size, self.image_size))
            d_u16 = np.clip(d_resized * 10000, 0, 65535).astype(np.uint16)
            _, img_enc = cv2.imencode(".jpg", img)
            _, d_enc   = cv2.imencode(".png", d_u16)

            resp = requests.post(
                f"{self.base_url}/step",
                files={
                    "image": ("image.jpg", img_enc.tobytes(), "image/jpeg"),
                    "depth": ("depth.png", d_enc.tobytes(),  "image/png"),
                },
                data={"goal_data": json.dumps(goal_data)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
            action = int(result.get("action", _FORWARD))

            traj_i = self._action_to_trajectory(
                action, step_m=self.step_m, turn_rad=self.turn_rad, T=self.waypoint_T
            )  # (1, T, 3)

            # STOP 动作 → 负分，触发 pipeline 停止
            score = -10.0 if action == _STOP else 0.0
            trajs.append(traj_i[0])
            all_trajs.append(traj_i)
            vals.append([[score]])

        traj     = np.stack(trajs,     axis=0)          # (B, T, 3)
        all_traj = np.concatenate(all_trajs, axis=0)    # (B, 1, T, 3)
        values   = np.array(vals, dtype=np.float32).squeeze(-1)  # (B, 1)
        return traj, all_traj, values

    # ──────────────────────────────────────────────────────────────────────────

    def pixelgoal_step(self, pixel_goals, rgb_images, depth_images):
        """像素目标 → 从深度图估算距离/方向，调用 step。"""
        if depth_images.ndim == 4:
            depth_images = depth_images[:, :, :, 0]
        goal_data = self._pixel_to_polar(pixel_goals, depth_images, rgb_images)
        return self._call_step(rgb_images, depth_images, goal_data)

    def pointgoal_step(self, point_goals, rgb_images, depth_images):
        """三维点目标 → 转换为极坐标距离/方向。"""
        dist    = float(np.linalg.norm(point_goals[0, :2]))
        heading = float(np.arctan2(point_goals[0, 1], point_goals[0, 0]))
        return self._call_step(rgb_images, depth_images,
                               {"distance": dist, "heading": heading})

    def nogoal_step(self, rgb_images, depth_images):
        return self._call_step(rgb_images, depth_images,
                               {"distance": 5.0, "heading": 0.0})

    def imagegoal_step(self, goal_images, rgb_images, depth_images):
        # DD-PPO 不支持图像目标，退化为 nogoal
        return self.nogoal_step(rgb_images, depth_images)

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _pixel_to_polar(pixel_goals, depth_images, rgb_images):
        """将像素坐标 + 深度图转为极坐标 (distance, heading)。"""
        H, W = depth_images.shape[1], depth_images.shape[2]
        u, v = int(pixel_goals[0, 0]), int(pixel_goals[0, 1])
        u = max(0, min(W - 1, u))
        v = max(0, min(H - 1, v))
        dist = float(depth_images[0, v, u])
        if dist < 0.1 or dist > 20.0:
            dist = 2.0
        heading = float(np.arctan2(u - W / 2, W / 2))
        return {"distance": dist, "heading": heading}
