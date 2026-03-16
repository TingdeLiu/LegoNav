"""
LegoNav S1 客户端包

所有客户端均继承 BaseS1Client，接口统一，可在 LegoNavPipeline 中无缝替换。

快速选择指南:
  NavDPClient       — 推荐默认方案，轨迹+Critic评分，支持pixelgoal/pointgoal/imagegoal/nogoal
  NavDPLocalClient  — 同上，端侧本地推理（Jetson，无HTTP开销）
  GNMClient         — 轻量级通用导航，图像目标，适合长距离导航
  ViNTClient        — GNM 升级版（Transformer），效果更强
  NoMaDClient       — 扩散策略，支持探索模式
  DDPPOClient       — RL baseline，输出离散动作
  iPlannerClient    — 局部路径规划，支持点目标
  ViPlannerClient   — 室外语义规划，支持点目标+语义图
"""

from legonav.clients.base_client import BaseS1Client
from legonav.clients.ddppo_client import DDPPOClient
from legonav.clients.gnm_client import GNMClient
from legonav.clients.iplanner_client import iPlannerClient
from legonav.clients.navdp_client import NavDPClient
from legonav.clients.navdp_local_client import NavDPLocalClient
from legonav.clients.nomad_client import NoMaDClient
from legonav.clients.vint_client import ViNTClient
from legonav.clients.viplanner_client import ViPlannerClient

__all__ = [
    "BaseS1Client",
    "NavDPClient",
    "NavDPLocalClient",
    "GNMClient",
    "ViNTClient",
    "NoMaDClient",
    "DDPPOClient",
    "iPlannerClient",
    "ViPlannerClient",
]
