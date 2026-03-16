#!/usr/bin/env python3
"""
LegoNav Jetson ROS2 客户端

在 Jetson Orin NX 上运行，连接 GPU 服务器的 S2 (Qwen3-VL) 和 S1 (NavDP)，
实现语言指令驱动的自主导航闭环。

线程架构:
  ROS2 spin 线程   — rgb_depth_callback (30Hz) + odom_callback (50Hz)
  规划线程 (0.3s)  — 调用 LegoNavPipeline.step() → 更新 MPC 轨迹 / 旋转目标
  控制线程 (0.1s)  — MPC solve / 角速度控制 → 发布 /cmd_vel

启动:
  python legonav_ros_client.py \
      --instruction "Go to the red chair" \
      --s2_host 192.168.1.100 \
      --s1_host 192.168.1.100

依赖 (Jetson 端):
  pip install numpy requests Pillow opencv-python casadi scipy
  sudo apt install ros-humble-cv-bridge ros-humble-message-filters
"""

import argparse
import copy
import math
import threading
import time
from collections import deque
from enum import Enum, auto

import cv2
import numpy as np

# ROS2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from legonav.core.pipeline import GEMINI_336L_INTRINSIC, LegoNavPipeline
from legonav.robot.controllers import Mpc_controller, PID_controller
from legonav.utils.thread_utils import ReadWriteLock

# ─────────────────────────────────────────────────────────────────────────────
# 配置常量
# ─────────────────────────────────────────────────────────────────────────────
PLAN_PERIOD     = 0.30   # 规划周期 (s)
CONTROL_PERIOD  = 0.10   # 控制周期 (s)
ODOM_QUEUE_LEN  = 50     # 里程计时间队列长度

MAX_LINEAR      = 0.25   # 最大线速度 (m/s)
MAX_ANGULAR     = 0.30   # 最大角速度 (rad/s)
ROTATE_SPEED    = 0.20   # 原地旋转角速度 (rad/s)

COLLISION_DIST  = 0.60   # 碰撞安全距离 (m)
DEPTH_MIN_VALID = 0.10   # 深度有效下限 (m)
DEPTH_MAX_VALID = 5.00   # 深度有效上限 (m)

STOP_DIST_THRESH = 0.30  # 距目标此距离内视为到达 (m), 仅 MPC 完成判断用


# ─────────────────────────────────────────────────────────────────────────────
# 控制模式
# ─────────────────────────────────────────────────────────────────────────────
class Mode(Enum):
    IDLE       = auto()   # 等待第一次规划结果
    TRAJECTORY = auto()   # MPC 轨迹跟踪
    ROTATE     = auto()   # 原地旋转（目标不在视野内）
    STOP       = auto()   # 任务完成 / 手动停止


# ─────────────────────────────────────────────────────────────────────────────
# LegoNavNode
# ─────────────────────────────────────────────────────────────────────────────
class LegoNavNode(Node):
    """LegoNav ROS2 主节点"""

    def __init__(self, cfg: argparse.Namespace):
        super().__init__("legonav_node")
        self.cfg = cfg

        # ── 初始化 S1 客户端（本地模式 or 服务器模式）────────────────────────
        s1_client = None
        if cfg.local_s1:
            from legonav.clients.navdp_local_client import NavDPLocalClient
            s1_client = NavDPLocalClient(
                checkpoint=cfg.s1_checkpoint,
                device=cfg.s1_device,
                half=cfg.s1_half,
            )

        # ── 初始化管线 ────────────────────────────────────────────────────────
        self.pipeline = LegoNavPipeline(
            s2_host=cfg.s2_host, s2_port=cfg.s2_port,
            s1_host=cfg.s1_host, s1_port=cfg.s1_port,
            s1_client=s1_client,
            camera_intrinsic=GEMINI_336L_INTRINSIC,
        )
        self.pipeline.reset(cfg.instruction)

        # ── 控制器 ────────────────────────────────────────────────────────────
        self.mpc: Mpc_controller = None
        self.pid = PID_controller(
            Kp_trans=2.0, Kd_trans=0.0,
            Kp_yaw=1.5,  Kd_yaw=0.0,
            max_v=MAX_LINEAR, max_w=MAX_ANGULAR,
        )

        # ── 共享状态（RGB/Depth） ─────────────────────────────────────────────
        self._rgb_depth_lock = ReadWriteLock()
        self._rgb_bgr: np.ndarray = None
        self._depth_m: np.ndarray = None
        self._rgb_ts: float = 0.0
        self._new_frame: bool = False

        # ── 共享状态（Odom） ──────────────────────────────────────────────────
        self._odom_lock = ReadWriteLock()
        self._odom: list = None            # [x, y, yaw]
        self._odom_queue: deque = deque(maxlen=ODOM_QUEUE_LEN)
        self._homo_odom: np.ndarray = None
        self._vel: list = [0.0, 0.0]       # [linear_x, angular_z]
        self._homo_goal: np.ndarray = None  # PID 目标齐次矩阵

        # ── 共享状态（控制） ──────────────────────────────────────────────────
        self._ctrl_lock = ReadWriteLock()
        self._mode: Mode = Mode.IDLE
        self._world_traj: np.ndarray = None   # (N, 2) MPC 参考轨迹

        # ── 旋转状态（仅规划线程写，控制线程读）─────────────────────────────
        self._rotate_end_time: float = 0.0
        self._rotate_angular_vel: float = 0.0

        # ── ROS2 发布/订阅 ────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)

        _qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        rgb_sub   = Subscriber(self, Image, "/camera/color/image_raw")
        depth_sub = Subscriber(self, Image, "/camera/depth/image_raw")
        self._sync = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=5, slop=0.1)
        self._sync.registerCallback(self._rgb_depth_cb)

        self.create_subscription(Odometry, "/odom", self._odom_cb, _qos)

        # ── 启动后台线程 ──────────────────────────────────────────────────────
        self._running = True
        threading.Thread(target=self._planning_thread, daemon=True, name="plan").start()
        threading.Thread(target=self._control_thread,  daemon=True, name="ctrl").start()

        self.get_logger().info(
            f'LegoNav started | instruction="{cfg.instruction}" '
            f'| S2={cfg.s2_host}:{cfg.s2_port} S1={cfg.s1_host}:{cfg.s1_port}'
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ROS2 回调
    # ──────────────────────────────────────────────────────────────────────────

    def _rgb_depth_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        """RGB + Depth 时间同步回调（≈30 Hz）"""
        # ── RGB：ROS rgb8 → numpy BGR ─────────────────────────────────────────
        rgb_np = self._bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

        # ── Depth：16UC1 (mm) → float32 (m) ──────────────────────────────────
        raw_d = self._bridge.imgmsg_to_cv2(depth_msg, "16UC1").astype(np.float32)
        depth = raw_d / 1000.0
        depth[~np.isfinite(depth)] = 0.0

        ts = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9

        self._rgb_depth_lock.acquire_write()
        self._rgb_bgr   = bgr
        self._depth_m   = depth
        self._rgb_ts    = ts
        self._new_frame = True
        self._rgb_depth_lock.release_write()

    def _odom_cb(self, msg: Odometry) -> None:
        """里程计回调（≈50 Hz）"""
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * qz * qw, 1 - 2 * qz * qz)

        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        homo = np.array([
            [cos_y, -sin_y, 0, px],
            [sin_y,  cos_y, 0, py],
            [0,      0,     1, 0 ],
            [0,      0,     0, 1 ],
        ], dtype=np.float64)

        self._odom_lock.acquire_write()
        self._odom      = [px, py, yaw]
        self._homo_odom = homo
        self._vel       = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        self._odom_queue.append((time.time(), [px, py, yaw]))
        if self._homo_goal is None:        # 初始化 PID 目标为当前位置
            self._homo_goal = homo.copy()
        self._odom_lock.release_write()

    # ──────────────────────────────────────────────────────────────────────────
    # 规划线程
    # ──────────────────────────────────────────────────────────────────────────

    def _planning_thread(self) -> None:
        while self._running:
            t0 = time.time()

            # ── 等待新帧 ──────────────────────────────────────────────────────
            self._rgb_depth_lock.acquire_read()
            has_frame = self._new_frame
            self._rgb_depth_lock.release_read()

            if not has_frame:
                time.sleep(0.02)
                continue

            # ── 读取传感器数据 ─────────────────────────────────────────────────
            self._rgb_depth_lock.acquire_read()
            rgb_bgr = copy.deepcopy(self._rgb_bgr)
            depth_m = copy.deepcopy(self._depth_m)
            self._new_frame = False
            self._rgb_depth_lock.release_read()

            # ── 读取里程计（时间最近的） ────────────────────────────────────────
            self._odom_lock.acquire_read()
            odom = copy.deepcopy(self._odom)
            self._odom_lock.release_read()

            if odom is None:
                time.sleep(0.05)
                continue

            # ── 碰撞预检测 ────────────────────────────────────────────────────
            if _collision_detected(depth_m):
                self.get_logger().warn("[Plan] 障碍物过近，暂停规划")
                self._set_mode(Mode.STOP)
                time.sleep(PLAN_PERIOD)
                continue

            # ── 调用 LegoNav 管线 ─────────────────────────────────────────────
            try:
                result = self.pipeline.step(rgb_bgr, depth_m)
            except Exception as exc:
                self.get_logger().error(f"[Plan] pipeline.step failed: {exc}")
                time.sleep(PLAN_PERIOD)
                continue

            mode = result.get("mode")
            self.get_logger().info(
                f"[Plan] mode={mode} | "
                f"target={result.get('s2', {}).get('target')} | "
                f"nav={result.get('s2', {}).get('navigation')!r}"
            )

            # ── 处理结果 ──────────────────────────────────────────────────────
            if mode == "stop":
                self._set_mode(Mode.STOP)

            elif mode == "rotate":
                rad = result["rotation_rad"]
                dur = abs(rad) / ROTATE_SPEED
                ang_vel = math.copysign(ROTATE_SPEED, rad)
                self._rotate_end_time    = time.time() + dur
                self._rotate_angular_vel = ang_vel
                self._set_mode(Mode.ROTATE)

            elif mode == "trajectory":
                traj = result["trajectory"][0]          # (24, 3)  local frame
                world_pts = _local_to_world(traj, odom) # (N, 2)   world frame
                # 跳过起点附近几个点，避免 MPC 已驶过的点作为参考
                world_pts = world_pts[2:]

                self._ctrl_lock.acquire_write()
                if self.mpc is None:
                    self.mpc = Mpc_controller(
                        world_pts,
                        desired_v=MAX_LINEAR,
                        v_max=MAX_LINEAR,
                        w_max=MAX_ANGULAR,
                    )
                else:
                    self.mpc.update_ref_traj(world_pts)
                self._world_traj = world_pts
                self._ctrl_lock.release_write()

                # 同步更新 PID 目标（取轨迹末端点）
                self._odom_lock.acquire_write()
                goal_xy = world_pts[-1]
                goal_yaw = math.atan2(
                    goal_xy[1] - odom[1],
                    goal_xy[0] - odom[0],
                )
                self._homo_goal = _make_homo(goal_xy[0], goal_xy[1], goal_yaw)
                self._odom_lock.release_write()

                self._set_mode(Mode.TRAJECTORY)

            else:
                self.get_logger().warn(f"[Plan] unknown mode: {mode} | {result.get('message', '')}")

            # ── 维持规划周期 ───────────────────────────────────────────────────
            elapsed = time.time() - t0
            sleep_t = max(0.0, PLAN_PERIOD - elapsed)
            time.sleep(sleep_t)

    # ──────────────────────────────────────────────────────────────────────────
    # 控制线程
    # ──────────────────────────────────────────────────────────────────────────

    def _control_thread(self) -> None:
        while self._running:
            t0 = time.time()

            # ── 读取当前模式 ───────────────────────────────────────────────────
            self._ctrl_lock.acquire_read()
            mode = self._mode
            mpc  = self.mpc
            self._ctrl_lock.release_read()

            vx, wz = 0.0, 0.0

            if mode == Mode.TRAJECTORY and mpc is not None:
                # ── MPC 轨迹跟踪 ────────────────────────────────────────────────
                self._odom_lock.acquire_read()
                odom = copy.deepcopy(self._odom)
                self._odom_lock.release_read()

                if odom is not None:
                    try:
                        u, _ = mpc.solve(np.array(odom))
                        vx = float(np.clip(u[0, 0], 0.0, MAX_LINEAR))
                        wz = float(np.clip(u[0, 1], -MAX_ANGULAR, MAX_ANGULAR))
                    except Exception as exc:
                        self.get_logger().warn(f"[Ctrl] MPC solve failed: {exc}")
                        # 降级为 PID
                        vx, wz = self._pid_fallback(odom)

            elif mode == Mode.ROTATE:
                # ── 原地旋转 ─────────────────────────────────────────────────────
                if time.time() < self._rotate_end_time:
                    wz = float(np.clip(self._rotate_angular_vel, -MAX_ANGULAR, MAX_ANGULAR))
                else:
                    self._set_mode(Mode.IDLE)   # 旋转完成，等待下次规划

            elif mode in (Mode.STOP, Mode.IDLE):
                vx, wz = 0.0, 0.0

            self._publish_cmd(vx, wz)

            elapsed = time.time() - t0
            time.sleep(max(0.0, CONTROL_PERIOD - elapsed))

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode: Mode) -> None:
        self._ctrl_lock.acquire_write()
        self._mode = mode
        self._ctrl_lock.release_write()

    def _pid_fallback(self, odom: list) -> tuple:
        """MPC 失败时用 PID 控制"""
        self._odom_lock.acquire_read()
        homo_odom = copy.deepcopy(self._homo_odom)
        homo_goal = copy.deepcopy(self._homo_goal)
        vel       = copy.deepcopy(self._vel)
        self._odom_lock.release_read()

        if homo_odom is None or homo_goal is None:
            return 0.0, 0.0

        v, w, _, _ = self.pid.solve(homo_odom, homo_goal, np.array(vel))
        v = float(np.clip(v, 0.0, MAX_LINEAR))
        w = float(np.clip(w, -MAX_ANGULAR, MAX_ANGULAR))
        return v, w

    def _publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def stop(self) -> None:
        """关闭：停止机器人，停止线程"""
        self._running = False
        self._publish_cmd(0.0, 0.0)
        self.get_logger().info("[LegoNav] Stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _local_to_world(traj_local: np.ndarray, odom: list) -> np.ndarray:
    """
    将 NavDP 局部帧轨迹（累积位移，x=前, y=左, 米）转为世界帧 (x, y)。

    Args:
        traj_local : (T, 3)  NavDP 输出的累积位移序列
        odom       : [x, y, yaw]  当前里程计状态

    Returns:
        world_pts  : (T, 2)  世界坐标系轨迹点
    """
    x0, y0, yaw = odom
    cy, sy = math.cos(yaw), math.sin(yaw)
    pts = []
    for pt in traj_local:
        dx, dy = float(pt[0]), float(pt[1])
        # 机体坐标(x=前, y=左) → 世界坐标
        wx = x0 + dx * cy - dy * sy
        wy = y0 + dx * sy + dy * cy
        pts.append([wx, wy])
    return np.array(pts, dtype=np.float64)


def _make_homo(x: float, y: float, yaw: float) -> np.ndarray:
    """构造 4×4 齐次变换矩阵（用于 PID 目标）"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy, -sy, 0, x],
        [sy,  cy, 0, y],
        [0,   0,  1, 0],
        [0,   0,  0, 1],
    ], dtype=np.float64)


def _collision_detected(depth_m: np.ndarray) -> bool:
    """
    简单碰撞检测：检查图像前方中心区域最小深度。
    忽略无效值（0 或超出范围）。
    """
    if depth_m is None:
        return False
    h, w = depth_m.shape
    roi = depth_m[h // 3 : 2 * h // 3, w // 4 : 3 * w // 4]
    valid = roi[(roi > DEPTH_MIN_VALID) & (roi < DEPTH_MAX_VALID)]
    if valid.size == 0:
        return False
    return float(valid.min()) < COLLISION_DIST


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LegoNav Jetson ROS2 客户端",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instruction", required=True,
                        help='导航指令，如 "Go to the red chair"')
    parser.add_argument("--s2_host", default="127.0.0.1", help="S2 服务器 IP")
    parser.add_argument("--s2_port", type=int, default=8890,  help="S2 端口")
    parser.add_argument("--s1_host", default="127.0.0.1", help="S1 服务器 IP")
    parser.add_argument("--s1_port", type=int, default=8901,  help="S1 端口")
    parser.add_argument("--max_linear",  type=float, default=MAX_LINEAR,
                        help="最大线速度 (m/s)")
    parser.add_argument("--max_angular", type=float, default=MAX_ANGULAR,
                        help="最大角速度 (rad/s)")

    # ── S1 端侧部署参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--local_s1", action="store_true",
        help="在 Jetson 本地直接运行 NavDP，无需 S1 HTTP 服务器",
    )
    parser.add_argument(
        "--s1_checkpoint", default=None,
        help="NavDP 权重文件路径（--local_s1 时必填）",
    )
    parser.add_argument(
        "--s1_device", default="cuda:0",
        help="NavDP 推理设备（--local_s1 时有效，如 'cuda:0'）",
    )
    parser.add_argument(
        "--s1_half", action="store_true",
        help="NavDP 使用 fp16 推理（--local_s1 时有效，Jetson 推荐开启）",
    )

    return parser.parse_args()


def main():
    cfg = parse_args()

    if cfg.local_s1 and not cfg.s1_checkpoint:
        raise SystemExit("[LegoNav] --s1_checkpoint 是必填项（使用 --local_s1 时）")

    # 允许通过 CLI 参数覆盖模块级速度常量
    global MAX_LINEAR, MAX_ANGULAR
    MAX_LINEAR  = cfg.max_linear
    MAX_ANGULAR = cfg.max_angular

    rclpy.init()
    node = LegoNavNode(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[LegoNav] KeyboardInterrupt, shutting down …")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
