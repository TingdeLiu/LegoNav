"""
NavDP S1 HTTP 客户端工具

提供 Python API 调用 NavDP Server 端点，兼容 NavDP 项目的 HTTP 协议。

使用示例:
    from navdp_client import NavDPClient

    client = NavDPClient(port=8888)
    client.reset(camera_intrinsic)

    # 点目标导航
    traj, all_traj, values = client.pointgoal_step(goals, images, depths)

    # 无目标探索
    traj, all_traj, values = client.nogoal_step(images, depths)
"""

import io
import json
import time

import cv2
import numpy as np
import requests

from legonav.clients.base_client import BaseS1Client


class NavDPClient(BaseS1Client):
    """NavDP S1 HTTP 客户端

    Args:
        host: Server 地址 (default: '127.0.0.1')
        port: Server 端口 (default: 8901)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8901):
        self.base_url = f"http://{host}:{port}"

    def reset(
        self,
        camera_intrinsic: np.ndarray,
        batch_size: int = 1,
        stop_threshold: float = -3.0,
    ) -> str:
        """重置导航器

        Args:
            camera_intrinsic: 相机内参矩阵 (4×4)
            batch_size: 批次大小
            stop_threshold: 停止阈值

        Returns:
            算法名称 (str)
        """
        resp = requests.post(
            f"{self.base_url}/navigator_reset",
            json={
                "intrinsic": camera_intrinsic.tolist(),
                "stop_threshold": stop_threshold,
                "batch_size": batch_size,
            },
        )
        return resp.json()["algo"]

    def reset_env(self, env_id: int) -> str:
        """重置指定环境"""
        resp = requests.post(
            f"{self.base_url}/navigator_reset_env",
            json={"env_id": env_id},
        )
        return resp.json()["algo"]

    @staticmethod
    def _encode_images(rgb_images: np.ndarray, depth_images: np.ndarray):
        """编码 RGB 和 Depth 图像为 HTTP 请求格式

        Args:
            rgb_images: (B, H, W, 3) uint8 BGR
            depth_images: (B, H, W) or (B, H, W, 1) float32 深度（米）

        Returns:
            files dict for requests.post
        """
        # RGB: 垂直拼接所有 batch -> JPEG
        concat_rgb = np.concatenate([img for img in rgb_images], axis=0)
        _, rgb_encoded = cv2.imencode(".jpg", concat_rgb)
        rgb_bytes = io.BytesIO(rgb_encoded.tobytes())

        # Depth: 垂直拼接 -> uint16 PNG (×10000)
        if len(depth_images.shape) == 4:
            depth_images = depth_images[:, :, :, 0]
        concat_depth = np.concatenate([d for d in depth_images], axis=0)
        depth_uint16 = np.clip(concat_depth * 10000.0, 0, 65535.0).astype(np.uint16)
        _, depth_encoded = cv2.imencode(".png", depth_uint16)
        depth_bytes = io.BytesIO(depth_encoded.tobytes())

        return {
            "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
            "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
        }

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ):
        """点目标导航推理

        Args:
            point_goals: (B, 2) 目标点 [x, y]
            rgb_images: (B, H, W, 3) BGR 图像
            depth_images: (B, H, W) 或 (B, H, W, 1) 深度图

        Returns:
            trajectory: (B, T, 3) 最优轨迹
            all_trajectory: (B, N, T, 3) 所有候选轨迹
            all_values: (B, N) critic 评分
        """
        files = self._encode_images(rgb_images, depth_images)
        data = {
            "goal_data": json.dumps({
                "goal_x": point_goals[:, 0].tolist(),
                "goal_y": point_goals[:, 1].tolist(),
            }),
            "depth_time": time.time(),
            "rgb_time": time.time(),
        }
        resp = requests.post(f"{self.base_url}/pointgoal_step", files=files, data=data)
        result = resp.json()
        return (
            np.array(result["trajectory"]),
            np.array(result["all_trajectory"]),
            np.array(result["all_values"]),
        )

    def nogoal_step(self, rgb_images: np.ndarray, depth_images: np.ndarray):
        """无目标探索推理

        Args:
            rgb_images: (B, H, W, 3) BGR 图像
            depth_images: (B, H, W) 或 (B, H, W, 1) 深度图

        Returns:
            trajectory, all_trajectory, all_values
        """
        files = self._encode_images(rgb_images, depth_images)
        data = {"depth_time": time.time(), "rgb_time": time.time()}
        resp = requests.post(f"{self.base_url}/nogoal_step", files=files, data=data)
        result = resp.json()
        return (
            np.array(result["trajectory"]),
            np.array(result["all_trajectory"]),
            np.array(result["all_values"]),
        )

    def imagegoal_step(
        self,
        goal_images: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ):
        """图像目标导航推理

        Args:
            goal_images: (B, H, W, 3) 目标 BGR 图像
            rgb_images: (B, H, W, 3) 当前 BGR 图像
            depth_images: (B, H, W) 或 (B, H, W, 1) 深度图

        Returns:
            trajectory, all_trajectory, all_values
        """
        files = self._encode_images(rgb_images, depth_images)

        # 编码目标图像
        concat_goal = np.concatenate([img for img in goal_images], axis=0)
        _, goal_encoded = cv2.imencode(".jpg", concat_goal)
        goal_bytes = io.BytesIO(goal_encoded.tobytes())
        files["goal"] = ("goal.jpg", goal_bytes.getvalue(), "image/jpeg")

        data = {"depth_time": time.time(), "rgb_time": time.time()}
        resp = requests.post(f"{self.base_url}/imagegoal_step", files=files, data=data)
        result = resp.json()
        return (
            np.array(result["trajectory"]),
            np.array(result["all_trajectory"]),
            np.array(result["all_values"]),
        )

    def pixelgoal_step(
        self,
        pixel_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ):
        """像素目标导航推理

        Args:
            pixel_goals: (B, 2) 目标像素坐标 [x, y]
            rgb_images: (B, H, W, 3) BGR 图像
            depth_images: (B, H, W) 或 (B, H, W, 1) 深度图

        Returns:
            trajectory, all_trajectory, all_values
        """
        files = self._encode_images(rgb_images, depth_images)
        data = {
            "goal_data": json.dumps({
                "goal_x": pixel_goals[:, 0].tolist(),
                "goal_y": pixel_goals[:, 1].tolist(),
            }),
            "depth_time": time.time(),
            "rgb_time": time.time(),
        }
        resp = requests.post(f"{self.base_url}/pixelgoal_step", files=files, data=data)
        result = resp.json()
        return (
            np.array(result["trajectory"]),
            np.array(result["all_trajectory"]),
            np.array(result["all_values"]),
        )
