"""
NavDP 端侧本地推理客户端

直接在当前进程中加载 NavDP 模型并推理，无需 HTTP 服务器。
接口与 NavDPClient 完全一致，可作为其 drop-in 替换。

适用场景:
  - Jetson Orin NX 等边缘设备直接运行 S1 NavDP
  - 无网络依赖，推理延迟更低（无 HTTP 序列化开销）

用法:
    from navdp_local_client import NavDPLocalClient

    client = NavDPLocalClient(
        checkpoint="/path/to/navdp.ckpt",
        device="cuda:0",
        half=True,        # Jetson 上开启 fp16 加速
    )
    client.reset(camera_intrinsic, batch_size=1, stop_threshold=-3.0)

    traj, all_traj, values = client.pixelgoal_step(pixel_goals, rgb_images, depth_images)

注意:
  - 依赖 NavDPAgent（navdp_agent.py）和外部 NavDP 项目（policy_network.py）
  - NavDP 项目路径通过环境变量 NAVDP_ROOT 或默认与 LegoNav 同级的 NavDP/ 目录指定
"""

import numpy as np

from legonav.clients.base_client import BaseS1Client
from legonav.core.navdp_agent import NavDPAgent


class NavDPLocalClient(BaseS1Client):
    """NavDP 端侧本地推理客户端

    接口与 NavDPClient（HTTP 客户端）完全一致，可无缝替换。

    Args:
        checkpoint    : NavDP 权重文件路径（.ckpt）
        device        : 推理设备，如 "cuda:0"、"cpu"
        half          : 是否使用 fp16 推理（Jetson 推荐开启，可节省约 50% 显存）
        image_size    : 模型输入图像尺寸（默认 224）
        memory_size   : 历史帧记忆队列长度（默认 8）
        predict_size  : 预测轨迹点数（默认 24）
        temporal_depth: Transformer 解码器层数（默认 16）
        heads         : 注意力头数（默认 8）
        token_dim     : Token 维度（默认 384）
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        half: bool = False,
        image_size: int = 224,
        memory_size: int = 8,
        predict_size: int = 24,
        temporal_depth: int = 16,
        heads: int = 8,
        token_dim: int = 384,
    ):
        # camera_intrinsic 在 reset() 时设置，此处用占位内参初始化 agent
        _placeholder_intrinsic = np.eye(4, dtype=np.float64)
        _placeholder_intrinsic[0, 0] = 570.3
        _placeholder_intrinsic[1, 1] = 570.3
        _placeholder_intrinsic[0, 2] = 319.5
        _placeholder_intrinsic[1, 2] = 239.5

        self._agent = NavDPAgent(
            camera_intrinsic=_placeholder_intrinsic,
            checkpoint=checkpoint,
            image_size=image_size,
            memory_size=memory_size,
            predict_size=predict_size,
            temporal_depth=temporal_depth,
            heads=heads,
            token_dim=token_dim,
            device=device,
        )

        if half:
            self._agent.policy = self._agent.policy.half()
            print(f"[NavDPLocalClient] fp16 enabled on {device}", flush=True)

        print(
            f"[NavDPLocalClient] Loaded checkpoint={checkpoint} device={device}",
            flush=True,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # NavDPClient 兼容接口
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        camera_intrinsic: np.ndarray,
        batch_size: int = 1,
        stop_threshold: float = -3.0,
    ) -> str:
        """重置导航器（新 episode 开始时调用）

        Args:
            camera_intrinsic : 相机内参矩阵 (4×4)
            batch_size       : 批次大小（LegoNav 固定为 1）
            stop_threshold   : Critic 停止阈值（值越小越保守）

        Returns:
            "navdp_local"（与 NavDPClient 返回 algo 名称保持一致）
        """
        self._agent.camera_intrinsic = camera_intrinsic
        self._agent.reset(batch_size=batch_size, stop_threshold=stop_threshold)
        return "navdp_local"

    def reset_env(self, env_id: int) -> str:
        """重置指定环境的记忆队列"""
        self._agent.reset_env(env_id)
        return "navdp_local"

    def pixelgoal_step(
        self,
        pixel_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ):
        """像素目标导航推理（与 NavDPClient.pixelgoal_step 接口一致）

        Args:
            pixel_goals  : 目标像素坐标 (B, 2) [x, y]
            rgb_images   : BGR 图像 (B, H, W, 3) uint8
            depth_images : 深度图 (B, H, W) 或 (B, H, W, 1) float32 (米)

        Returns:
            trajectory    : (B, T, 3) 最优轨迹（相对位移，米）
            all_trajectory: (B, N, T, 3) 所有候选轨迹
            all_values    : (B, N) Critic 评分
        """
        if len(depth_images.shape) == 3:
            depth_images = depth_images[:, :, :, np.newaxis]   # (B,H,W) → (B,H,W,1)

        best_traj, all_traj, all_vals, _ = self._agent.step_pixelgoal(
            pixel_goals, rgb_images, depth_images
        )
        # best_traj: (B, T, 3); all_traj: (B, N, T, 3); all_vals: (B, N)
        return best_traj, all_traj, all_vals

    def pointgoal_step(
        self,
        point_goals: np.ndarray,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ):
        """点目标导航推理（与 NavDPClient.pointgoal_step 接口一致）"""
        if len(depth_images.shape) == 3:
            depth_images = depth_images[:, :, :, np.newaxis]

        best_traj, all_traj, all_vals, _ = self._agent.step_pointgoal(
            point_goals, rgb_images, depth_images
        )
        return best_traj, all_traj, all_vals

    def nogoal_step(
        self,
        rgb_images: np.ndarray,
        depth_images: np.ndarray,
    ):
        """无目标探索推理（与 NavDPClient.nogoal_step 接口一致）"""
        if len(depth_images.shape) == 3:
            depth_images = depth_images[:, :, :, np.newaxis]

        best_traj, all_traj, all_vals, _ = self._agent.step_nogoal(
            rgb_images, depth_images
        )
        return best_traj, all_traj, all_vals
