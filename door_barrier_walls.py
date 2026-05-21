"""
门前阻挡墙：门关闭时出现（阻止机器人撞门），门打开时消失。
墙体为薄 Cube（≈0.05m 厚），沿门自身朝向放置（支持斜门）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.objects.primitive_object import PrimitiveObject

# 门名 → 墙体列表 [wall_a, wall_b]
_door_walls: Dict[str, List[PrimitiveObject]] = {}

WALL_THICKNESS_M = 0.05
WALL_OFFSET_M = 0.17  # 距门中心偏移（原 0.25 的 2/3）
WALL_RGBA = (1.0, 1.0, 1.0, 1.0)  # 创建时需要有效颜色，生成后立刻隐藏视觉


# ── 四元数工具 ──────────────────────────────────────────────────

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数 (x,y,z,w) → 3×3 旋转矩阵。"""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _door_pose(door) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """返回 (position(3,), orientation(4,)) 世界系，orientation 为 (x,y,z,w) 四元数。"""
    try:
        pos, ori = door.get_position_orientation()
    except Exception:
        return None
    if th.is_tensor(pos):
        pos = pos.detach().cpu().numpy()
    if th.is_tensor(ori):
        ori = ori.detach().cpu().numpy()
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)
    ori = np.asarray(ori, dtype=np.float64).reshape(-1)
    if pos.size < 3 or ori.size < 4:
        return None
    return pos[:3].copy(), ori[:4].copy()


def _door_aabb_extent(door) -> Optional[np.ndarray]:
    """取门的 AABB extent（世界轴对齐），返回 (3,) 或 None。"""
    try:
        e = door.aabb_extent
    except Exception:
        return None
    if e is None:
        return None
    if th.is_tensor(e):
        e = e.detach().cpu().numpy()
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    return e if e.size >= 3 else None


def spawn_door_barrier_walls(
    scene,
    doors: List,
    door_states: Dict[str, str],
    skip_doors: set | None = None,
) -> None:
    """
    为每扇门沿其自身朝向在两侧各生成一面薄墙（跟随门的旋转）。
    已打开的门：墙隐藏 + 禁碰撞。关闭的门：墙可见 + 有碰撞。
    """
    global _door_walls
    skip = skip_doors or set()

    for idx, door in enumerate(doors, 1):
        if idx in skip:
            continue
        pose = _door_pose(door)
        extent = _door_aabb_extent(door)
        if pose is None or extent is None:
            continue
        door_pos, door_quat = pose  # door_quat: (x,y,z,w)
        R = _quat_to_rotmat(door_quat)

        # 门的局部 X 轴通常是门板平面的法线方向（薄轴）
        # 墙的宽度取 AABB 最大水平 extent（覆盖门洞），厚度固定
        ex, ey, ez = float(extent[0]), float(extent[1]), float(extent[2])
        wall_width = max(ex, ey) * 1.3
        wall_height = ez * 0.95

        # 门局部 X 轴（法线方向）在世界系中的投影
        local_normal = R[:, 0]  # 旋转矩阵第 0 列 = 局部 X 在世界系
        normal_2d = local_normal[:2]
        n_len = float(np.linalg.norm(normal_2d))
        if n_len < 1e-6:
            continue
        normal_2d = normal_2d / n_len

        # 墙中心 = 门 AABB 中心（Z 用 AABB 中心比 door_pos 更准）
        try:
            aabb_c = door.aabb_center
            if th.is_tensor(aabb_c):
                aabb_c = aabb_c.detach().cpu().numpy()
            aabb_c = np.asarray(aabb_c, dtype=np.float64).reshape(-1)[:3]
        except Exception:
            aabb_c = door_pos

        # 墙的 scale：局部 X=厚度, Y=宽度, Z=高度
        wall_scale = np.array([WALL_THICKNESS_M, wall_width, wall_height], dtype=np.float64)

        walls_for_door: List[PrimitiveObject] = []
        for side in (0, 1):
            sign = 1.0 if side == 0 else -1.0
            offset_world = np.zeros(3, dtype=np.float64)
            offset_world[:2] = normal_2d * (WALL_OFFSET_M * sign)
            wall_pos = aabb_c + offset_world

            name = f"door_barrier_wall_{idx}_{side}"
            try:
                wall = PrimitiveObject(
                    relative_prim_path=f"/{name}",
                    primitive_type="Cube",
                    name=name,
                    category="obstacle",
                    scale=wall_scale.copy(),
                    size=1.0,
                    fixed_base=True,
                    visual_only=False,
                    rgba=WALL_RGBA,
                )
                scene.add_object(wall)
                wall.set_position_orientation(position=wall_pos, orientation=door_quat)
                # 隐藏视觉（渲染器里 rgba alpha=0 仍显示黑色），但保留碰撞体积
                for lnk in wall.links.values():
                    try:
                        lnk.visible = False
                    except Exception:
                        pass
                walls_for_door.append(wall)
            except Exception as ex_:
                print(f"  ⚠️ 门#{idx} ({door.name}) side{side} 生成阻挡墙失败: {ex_}", flush=True)
        if walls_for_door:
            _door_walls[door.name] = walls_for_door
            yaw_deg = float(np.degrees(np.arctan2(2*(door_quat[3]*door_quat[2] + door_quat[0]*door_quat[1]),
                                                    1 - 2*(door_quat[1]**2 + door_quat[2]**2))))
            print(
                f"  🧱 门#{idx} ({door.name}) 两侧阻挡墙，"
                f"宽={wall_width:.2f}m 高={wall_height:.2f}m 厚={WALL_THICKNESS_M}m，"
                f"门朝向≈{yaw_deg:.1f}°，偏移±{WALL_OFFSET_M:.2f}m",
                flush=True,
            )

    # 等物理 settle
    for _ in range(10):
        og.sim.step()

    # 根据初始门状态设置碰撞：门开→禁碰撞，门关→启碰撞（视觉始终隐藏）
    for door in doors:
        state = door_states.get(door.name, "close")
        if state == "open":
            hide_barrier_wall(door.name)
        else:
            show_barrier_wall(door.name)


def hide_barrier_wall(door_name: str) -> None:
    """门打开时调用：隐藏墙并禁碰撞。"""
    walls = _door_walls.get(door_name)
    if not walls:
        return
    for wall in walls:
        try:
            for link in wall.links.values():
                link.visible = False
                if getattr(link, "has_collision_meshes", False) and link.collision_meshes:
                    link.disable_collisions()
        except Exception:
            pass


def show_barrier_wall(door_name: str) -> None:
    """门关闭时调用：启碰撞（视觉保持隐藏）。"""
    walls = _door_walls.get(door_name)
    if not walls:
        return
    for wall in walls:
        try:
            for link in wall.links.values():
                if getattr(link, "has_collision_meshes", False) and link.collision_meshes:
                    link.enable_collisions()
        except Exception:
            pass


def remove_all_barrier_walls(scene) -> None:
    """清理所有阻挡墙（env.reset 前调用）。"""
    global _door_walls
    for name, walls in list(_door_walls.items()):
        for wall in walls:
            try:
                scene.remove_object(wall)
            except Exception:
                pass
    _door_walls.clear()


def update_barrier_wall_for_door(door_name: str, action: str) -> None:
    """根据开/关门动作更新墙状态。供 control_door 调用。"""
    if action == "open":
        hide_barrier_wall(door_name)
    else:
        show_barrier_wall(door_name)
