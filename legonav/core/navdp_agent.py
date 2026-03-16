"""
NavDP S1 Agent — 独立推理封装层

复用 NavDP 项目的 NavDP_Policy 类，提供记忆队列管理、RGBD 预处理和轨迹可视化。
使用方式请参考 docs/navdp_s1_standalone_guide.md
"""

import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps as cm

# ---------------------------------------------------------------------------
# 自动定位 NavDP 项目路径，将其 baselines/navdp 加入 sys.path
# 用户也可通过环境变量 NAVDP_ROOT 显式指定
# 默认值: 与 LegoNav 同级的 NavDP/ 目录 (C:\GitHub\NavDP)
# ---------------------------------------------------------------------------
_NAVDP_ROOT = os.environ.get(
    "NAVDP_ROOT",
    str(Path(__file__).resolve().parent.parent.parent.parent / "NavDP"),
)
_NAVDP_BASELINE_DIR = os.path.join(_NAVDP_ROOT, "baselines", "navdp")
if _NAVDP_BASELINE_DIR not in sys.path:
    sys.path.insert(0, _NAVDP_BASELINE_DIR)

from policy_network import NavDP_Policy  # noqa: E402


class NavDPAgent:
    """NavDP S1 导航代理

    封装 NavDP_Policy（来自 NavDP 项目），管理 RGBD 历史记忆队列、
    输入预处理和轨迹投影可视化。支持 5 种导航任务模式：
    pointgoal / nogoal / imagegoal / pixelgoal / mixgoal

    Args:
        camera_intrinsic: 相机内参矩阵 (4×4 numpy array)
        checkpoint: 模型权重路径 (.ckpt)
        image_size: 模型输入图像尺寸 (default: 224)
        memory_size: 历史帧记忆数量 (default: 8)
        predict_size: 预测轨迹点数 (default: 24)
        temporal_depth: Transformer 解码器层数 (default: 16)
        heads: 注意力头数 (default: 8)
        token_dim: Token 维度 (default: 384)
        device: 计算设备 (default: 'cuda:0')
    """

    def __init__(
        self,
        camera_intrinsic: np.ndarray,
        checkpoint: str,
        image_size: int = 224,
        memory_size: int = 8,
        predict_size: int = 24,
        temporal_depth: int = 16,
        heads: int = 8,
        token_dim: int = 384,
        device: str = "cuda:0",
    ):
        self.camera_intrinsic = camera_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size

        # 加载 NavDP_Policy 网络
        self.policy = NavDP_Policy(
            image_size, memory_size, predict_size, temporal_depth, heads, token_dim, device=device
        )
        state_dict = torch.load(checkpoint, map_location=device)
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.to(device)
        self.policy.eval()

        # 运行状态
        self.batch_size = 1
        self.stop_threshold = -3.0
        self.memory_queue = []

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------
    def reset(self, batch_size: int = 1, stop_threshold: float = -3.0):
        """重置代理状态（新 episode 时调用）"""
        self.batch_size = batch_size
        self.stop_threshold = stop_threshold
        self.memory_queue = [[] for _ in range(batch_size)]

    def reset_env(self, env_id: int):
        """重置指定环境的记忆队列"""
        self.memory_queue[env_id] = []

    # ------------------------------------------------------------------
    # 输入预处理
    # ------------------------------------------------------------------
    def process_image(self, images: np.ndarray) -> np.ndarray:
        """处理 RGB 图像：缩放 + 填充到 image_size × image_size，归一化到 [0,1]

        Args:
            images: (B, H, W, C) uint8 BGR 图像

        Returns:
            (B, image_size, image_size, C) float32 归一化图像
        """
        assert len(images.shape) == 4
        H, W = images.shape[1], images.shape[2]
        prop = self.image_size / max(H, W)
        result = []
        for img in images:
            resized = cv2.resize(img, (-1, -1), fx=prop, fy=prop)
            pad_w = max((self.image_size - resized.shape[1]) // 2, 0)
            pad_h = max((self.image_size - resized.shape[0]) // 2, 0)
            padded = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
            resized = cv2.resize(padded, (self.image_size, self.image_size))
            result.append(resized.astype(np.float32) / 255.0)
        return np.array(result)

    def process_depth(self, depths: np.ndarray) -> np.ndarray:
        """处理深度图：缩放 + 填充，过滤异常值

        Args:
            depths: (B, H, W, 1) float32 深度图（单位：米）

        Returns:
            (B, image_size, image_size, 1) float32 处理后深度图
        """
        assert len(depths.shape) == 4
        depths = depths.copy()
        depths[depths == np.inf] = 0
        H, W = depths.shape[1], depths.shape[2]
        prop = self.image_size / max(H, W)
        result = []
        for depth in depths:
            resized = cv2.resize(depth, (-1, -1), fx=prop, fy=prop)
            pad_w = max((self.image_size - resized.shape[1]) // 2, 0)
            pad_h = max((self.image_size - resized.shape[0]) // 2, 0)
            padded = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
            resized = cv2.resize(padded, (self.image_size, self.image_size))
            resized[resized > 5.0] = 0
            resized[resized < 0.1] = 0
            result.append(resized[:, :, np.newaxis])
        return np.array(result)

    def process_pointgoal(self, goals: np.ndarray) -> np.ndarray:
        """裁剪目标点到合理范围 [-10, 10]，X >= 0"""
        clip_goals = goals.clip(-10, 10)
        clip_goals[:, 0] = np.clip(clip_goals[:, 0], 0, 10)
        return clip_goals

    def process_pixel(self, pixel_coords: np.ndarray, input_images: np.ndarray) -> np.ndarray:
        """生成像素目标掩码"""
        result = []
        H, W, C = input_images.shape[1], input_images.shape[2], input_images.shape[3]
        prop = self.image_size / max(H, W)
        for pixel_coord, input_image in zip(pixel_coords, input_images):
            panel = np.zeros_like(input_image, dtype=np.uint8)
            px, py = int(pixel_coord[0]), int(pixel_coord[1])
            min_x, min_y = px - 10, py - 10
            max_x, max_y = px + 10, py + 10
            if min_x <= 0:
                panel[:, 0:10] = 255
            elif min_y <= 0:
                panel[0:10, :] = 255
            elif max_x >= panel.shape[1]:
                panel[:, panel.shape[1] - 10 :] = 255
            elif max_y >= panel.shape[0]:
                panel[panel.shape[0] - 10 :, :] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel.shape[1] and max_y < panel.shape[0]:
                panel[min_y:max_y, min_x:max_x] = 255
            resized = cv2.resize(panel, (-1, -1), fx=prop, fy=prop, interpolation=cv2.INTER_NEAREST)
            pad_w = max((self.image_size - resized.shape[1]) // 2, 0)
            pad_h = max((self.image_size - resized.shape[0]) // 2, 0)
            padded = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
            resized = cv2.resize(padded, (self.image_size, self.image_size))
            result.append(resized.astype(np.float32) / 255.0)
        return np.array(result).mean(axis=-1)

    # ------------------------------------------------------------------
    # 记忆队列更新（内部方法）
    # ------------------------------------------------------------------
    def _update_memory(self, process_images: np.ndarray) -> np.ndarray:
        """更新记忆队列并返回 (B, memory_size, H, W, C) 输入"""
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                img = np.array(self.memory_queue[i])
                img = np.pad(img, ((self.memory_size - img.shape[0], 0), (0, 0), (0, 0), (0, 0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])
                img = np.array(self.memory_queue[i])
            input_images.append(img)
        return np.array(input_images)

    # ------------------------------------------------------------------
    # 轨迹投影可视化
    # ------------------------------------------------------------------
    def project_trajectory(
        self, images: np.ndarray, n_trajectories: np.ndarray, n_values: np.ndarray
    ) -> np.ndarray:
        """将 3D 轨迹投影到 2D 图像上进行可视化

        Args:
            images: 原始 RGB 图像 (B, H, W, C)
            n_trajectories: 所有候选轨迹 (B, N, T, 3)
            n_values: 轨迹评分 (B, N)

        Returns:
            水平拼接的可视化图像
        """
        masks = []
        for i in range(images.shape[0]):
            mask = np.array(images[i])
            for waypoints, value in zip(n_trajectories[i, :, :, 0:2], n_values[i]):
                norm_val = np.clip(-value * 0.1, 0, 1)
                color = np.array(cm.get_cmap("jet")(norm_val)[0:3]) * 255.0
                pts = np.zeros((waypoints.shape[0], 3)) - 0.2
                pts[:, 0:2] = waypoints
                pts[:, 1] = -pts[:, 1]
                cam_z = (
                    images[0].shape[0]
                    - 1
                    - self.camera_intrinsic[1][1] * pts[:, 2] / (pts[:, 0] + 1e-8)
                    - self.camera_intrinsic[1][2]
                )
                cam_x = self.camera_intrinsic[0][0] * pts[:, 1] / (pts[:, 0] + 1e-8) + self.camera_intrinsic[0][2]
                for j in range(cam_x.shape[0] - 1):
                    try:
                        if cam_x[j] > 0 and cam_z[j] > 0 and cam_x[j + 1] > 0 and cam_z[j + 1] > 0:
                            mask = cv2.line(
                                mask,
                                (int(cam_x[j]), int(cam_z[j])),
                                (int(cam_x[j + 1]), int(cam_z[j + 1])),
                                color.astype(np.uint8).tolist(),
                                5,
                            )
                    except Exception:
                        pass
            masks.append(mask)
        return np.concatenate(masks, axis=1)

    # ------------------------------------------------------------------
    # 推理接口：5 种导航任务模式
    # ------------------------------------------------------------------
    def step_pointgoal(self, goals: np.ndarray, images: np.ndarray, depths: np.ndarray):
        """点目标导航

        Args:
            goals: 目标点坐标 (B, 3)，相机坐标系 (x=前, y=左, z=上)
            images: BGR 图像 (B, H, W, 3)
            depths: 深度图 (B, H, W, 1)

        Returns:
            (best_trajectory, all_trajectories, critic_values, vis_image)
        """
        proc_imgs = self.process_image(images)
        proc_deps = self.process_depth(depths)
        input_imgs = self._update_memory(proc_imgs)
        input_goals = self.process_pointgoal(goals)

        all_traj, all_vals, good_traj, _ = self.policy.predict_pointgoal_action(
            input_goals, input_imgs, proc_deps
        )

        if all_vals.max() < self.stop_threshold:
            good_traj[:, :, :, 0] *= 0.0
            good_traj[:, :, :, 1] = np.sign(good_traj[:, :, :, 1].mean())

        vis = self.project_trajectory(images, all_traj, all_vals)
        return good_traj[:, 0], all_traj, all_vals, vis

    def step_nogoal(self, images: np.ndarray, depths: np.ndarray):
        """无目标探索

        Args:
            images: BGR 图像 (B, H, W, 3)
            depths: 深度图 (B, H, W, 1)

        Returns:
            (best_trajectory, all_trajectories, critic_values, vis_image)
        """
        proc_imgs = self.process_image(images)
        proc_deps = self.process_depth(depths)
        input_imgs = self._update_memory(proc_imgs)

        all_traj, all_vals, good_traj, _ = self.policy.predict_nogoal_action(input_imgs, proc_deps)

        if all_vals.max() < self.stop_threshold:
            good_traj[:, :, :, 0] *= 0.0
            good_traj[:, :, :, 1] = np.sign(good_traj[:, :, :, 1].mean())

        vis = self.project_trajectory(images, all_traj, all_vals)
        return good_traj[:, 0], all_traj, all_vals, vis

    def step_imagegoal(self, goal_images: np.ndarray, images: np.ndarray, depths: np.ndarray):
        """图像目标导航

        Args:
            goal_images: 目标 BGR 图像 (B, H, W, 3)
            images: 当前 BGR 图像 (B, H, W, 3)
            depths: 深度图 (B, H, W, 1)

        Returns:
            (best_trajectory, all_trajectories, critic_values, vis_image)
        """
        proc_imgs = self.process_image(images)
        proc_deps = self.process_depth(depths)
        input_imgs = self._update_memory(proc_imgs)
        input_goals = self.process_image(goal_images)

        all_traj, all_vals, good_traj, _ = self.policy.predict_imagegoal_action(
            input_goals, input_imgs, proc_deps
        )

        if all_vals.max() < self.stop_threshold:
            good_traj[:, :, :, 0] *= 0.0
            good_traj[:, :, :, 1] = np.sign(good_traj[:, :, :, 1].mean())

        vis = self.project_trajectory(images, all_traj, all_vals)
        return good_traj[:, 0], all_traj, all_vals, vis

    def step_pixelgoal(self, pixel_goals: np.ndarray, images: np.ndarray, depths: np.ndarray):
        """像素目标导航

        Args:
            pixel_goals: 目标像素坐标 (B, 2) [x, y]
            images: BGR 图像 (B, H, W, 3)
            depths: 深度图 (B, H, W, 1)

        Returns:
            (best_trajectory, all_trajectories, critic_values, vis_image)
        """
        proc_imgs = self.process_image(images)
        proc_deps = self.process_depth(depths)
        input_imgs = self._update_memory(proc_imgs)
        input_goals = self.process_pixel(pixel_goals, images)

        all_traj, all_vals, good_traj, _ = self.policy.predict_pixelgoal_action(
            input_goals, input_imgs, proc_deps
        )

        if all_vals.max() < self.stop_threshold:
            good_traj[:, :, :, 0] *= 0.0
            good_traj[:, :, :, 1] = np.sign(good_traj[:, :, :, 1].mean())

        vis = self.project_trajectory(images, all_traj, all_vals)
        return good_traj[:, 0], all_traj, all_vals, vis

    def step_mixgoal(
        self,
        point_goals: np.ndarray,
        image_goals: np.ndarray,
        images: np.ndarray,
        depths: np.ndarray,
    ):
        """混合目标导航（点目标 + 图像目标）

        Args:
            point_goals: 目标点坐标 (B, 3)
            image_goals: 目标 BGR 图像 (B, H, W, 3)
            images: 当前 BGR 图像 (B, H, W, 3)
            depths: 深度图 (B, H, W, 1)

        Returns:
            (best_trajectory, all_trajectories, critic_values, vis_image)
        """
        proc_imgs = self.process_image(images)
        proc_deps = self.process_depth(depths)
        input_imgs = self._update_memory(proc_imgs)
        input_pointgoal = self.process_pointgoal(point_goals)
        input_imagegoal = self.process_image(image_goals)

        all_traj, all_vals, good_traj, _ = self.policy.predict_ip_action(
            input_pointgoal, input_imagegoal, input_imgs, proc_deps
        )

        if all_vals.max() < self.stop_threshold:
            good_traj[:, :, :, 0] *= 0.0
            good_traj[:, :, :, 1] = np.sign(good_traj[:, :, :, 1].mean())

        vis = self.project_trajectory(images, all_traj, all_vals)
        return good_traj[:, 0], all_traj, all_vals, vis
