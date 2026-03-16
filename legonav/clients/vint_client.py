"""
ViNT (Vision-and-Language Navigation Transformer) S1 HTTP 客户端

论文: ViNT: A Foundation Model for Visual Navigation (CoRL 2023)
项目: https://github.com/robodhruv/visualnav-transformer

ViNT 在 GNM 基础上使用 Transformer 架构，支持更长历史序列，
并输出 temporal distance（到达目标的预估步数）。

服务端与 GNMClient 使用完全相同的 HTTP 协议（/reset + /navigate），
只需更换 --model 参数启动不同模型，因此 ViNTClient 直接继承 GNMClient。

服务端启动:
  python deployment/src/navigate.py --model vint --port 8047
"""

from legonav.clients.gnm_client import GNMClient


class ViNTClient(GNMClient):
    """ViNT S1 HTTP 客户端（与 GNMClient 协议相同，默认端口 8047）。"""

    algo_name = "vint"

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8047,
        timeout: float = 10.0,
        waypoint_T: int = 8,
        stop_dist: float = 0.3,
    ):
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
            waypoint_T=waypoint_T,
            stop_dist=stop_dist,
        )
        self.algo_name = "vint"
