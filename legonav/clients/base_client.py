"""
S1 客户端抽象基类

所有 S1 导航策略客户端（NavDP、GNM、ViNT、NoMad、DD-PPO、iPlanner、ViPlanner）
均继承此基类，保证接口统一，可在 LegoNavPipeline 中无缝替换。

输出约定
---------
所有 *_step() 方法返回三元组 (trajectory, all_trajectory, values)：
  trajectory     : np.ndarray (B, T, 3)   最优轨迹，相对位移 [dx, dy, dz]（米）
                   坐标系: x=前, y=左, z=上（与 NavDP / LegoNav 一致）
  all_trajectory : np.ndarray (B, N, T, 3) 所有候选轨迹
  values         : np.ndarray (B, N)       轨迹评分（越高越好）
                   对于没有 Critic 的模型，填充 0.0
                   对于 NavDP Critic，值通常为负数（如 -3.0 表示到达）

停止判断
---------
LegoNavPipeline 使用 values.max() < stop_threshold 判断是否到达目标。
- NavDP:         values 由 Critic 网络输出，stop_threshold 推荐 -3.0
- 其他模型:       values=0.0，停止判断依赖计步器或外部触发
  可将 stop_threshold 设为 -99 以禁用自动停止，由上层逻辑控制
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseS1Client(ABC):
    """S1 导航策略客户端抽象基类。"""

    # ── 子类应覆盖此属性，用于日志 ────────────────────────────────────────────
    algo_name: str = "base"

    # ──────────────────────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def reset(
        self,
        camera_intrinsic: np.ndarray,
        batch_size: int = 1,
        stop_threshold: float = -3.0,
    ) -> str:
        """新 episode 开始时调用。

        参数:
            camera_intrinsic : 相机内参矩阵 (4×4) float64
            batch_size       : 并行环境数，LegoNav 单机场景固定为 1
            stop_threshold   : 到达判断阈值（values.max() < threshold → 停止）

        返回:
            算法名称字符串（用于日志）
        """

    # ──────────────────────────────────────────────────────────────────────────
    # 推理接口
    # ──────────────────────────────────────────────────────────────────────────

    def pixelgoal_step(
        self,
        pixel_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """像素目标导航推理。

        参数:
            pixel_goals  : (B, 2) 图像像素坐标 [u, v]
            rgb_images   : (B, H, W, 3) uint8 BGR
            depth_images : (B, H, W) 或 (B, H, W, 1) float32 深度（米）

        返回:
            trajectory, all_trajectory, values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 pixelgoal_step"
        )

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """三维点目标导航推理。

        参数:
            point_goals  : (B, 3) 相机坐标系下目标 [x=前, y=左, z=上]（米）
            rgb_images   : (B, H, W, 3) uint8 BGR
            depth_images : (B, H, W) 或 (B, H, W, 1) float32

        返回:
            trajectory, all_trajectory, values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 pointgoal_step"
        )

    def imagegoal_step(
        self,
        goal_images: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """图像目标导航推理。

        参数:
            goal_images  : (B, H, W, 3) uint8 BGR 目标场景参考图
            rgb_images   : (B, H, W, 3) uint8 BGR 当前观测
            depth_images : (B, H, W) 或 (B, H, W, 1) float32

        返回:
            trajectory, all_trajectory, values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 imagegoal_step"
        )

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """无目标自主探索推理。

        参数:
            rgb_images   : (B, H, W, 3) uint8 BGR
            depth_images : (B, H, W) 或 (B, H, W, 1) float32

        返回:
            trajectory, all_trajectory, values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 未实现 nogoal_step"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 工具方法（子类可复用）
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap_single_trajectory(
        traj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """将单条轨迹 (B, T, 3) 包装为标准三元组。

        适用于只输出一条轨迹、没有多候选采样的模型。
        values 填充 0.0（pipeline 应将 stop_threshold 设为负数以禁用自动停止）。
        """
        B = traj.shape[0]
        all_traj = traj[:, np.newaxis, :, :]          # (B, 1, T, 3)
        values   = np.zeros((B, 1), dtype=np.float32) # (B, 1)
        return traj, all_traj, values

    @staticmethod
    def _action_to_trajectory(
        action: int,
        step_m: float = 0.25,
        turn_rad: float = 0.2618,  # 15°
        T: int = 8,
    ) -> np.ndarray:
        """将离散动作转换为单步轨迹 (1, T, 3)。

        动作编码 (Habitat 标准):
            0 = STOP     → 零向量
            1 = FORWARD  → x 方向位移
            2 = TURN_LEFT  → 绕 z 轴正向旋转（以 dz 编码角度）
            3 = TURN_RIGHT → 绕 z 轴负向旋转

        轨迹坐标系: x=前, y=左, z=上（z 分量借用存储角度，单位 rad）
        """
        wp = np.zeros((T, 3), dtype=np.float32)
        if action == 1:           # FORWARD
            wp[:, 0] = step_m
        elif action == 2:         # TURN_LEFT
            wp[:, 2] = turn_rad
        elif action == 3:         # TURN_RIGHT
            wp[:, 2] = -turn_rad
        # STOP → 全零
        return wp[np.newaxis]     # (1, T, 3)

    @staticmethod
    def _waypoints_to_trajectory(
        waypoints: np.ndarray,
        T: int = 8,
    ) -> np.ndarray:
        """将 (B, K, 2) 路标点转换为 (B, T, 3) 轨迹。

        路标点坐标系: [x=前, y=左]（与 GNM/ViNT/NoMad 输出一致）
        不足 T 步时用最后一个点填充，超过 T 步时截断。
        """
        B, K, _ = waypoints.shape
        traj = np.zeros((B, T, 3), dtype=np.float32)
        n = min(K, T)
        traj[:, :n, :2] = waypoints[:, :n, :]
        if n < T:
            traj[:, n:, :2] = waypoints[:, -1:, :]  # 用最后一个点填充
        return traj
