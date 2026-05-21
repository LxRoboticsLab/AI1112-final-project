"""
在已有场景中生成可抓取的小物体（放在桌子等表面上）。
使用 datasets/behavior-1k-assets/objects 中的类别与模型，不修改场景 JSON。
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch as th

import omnigibson as og
from omnigibson.utils.asset_utils import get_all_object_category_models

# 适合抓取的小物体类别（来自 behavior-1k-assets/objects，排除液体、系统等）
SMALL_GRASPABLE_CATEGORIES = [
    "apple",
    "bottle_of_cologne",
    "bowl",
    "remote_control",
    "alarm_clock",
    "pillow",
    "tray",
    "beaker",
    "bottle_of_champagne",
    "bagel",
    "orange",
    "banana",
    "can",
    "box",
    "plate",
    "mug",
    "keyboard",
    "mouse",
    "pen",
    "cell_phone",
]

# 适合作为放置表面的物体类别（用于找“桌子”）
TABLE_LIKE_KEYWORDS = ("table", "countertop", "desk", "cabinet")


def _is_table_like(obj) -> bool:
    """判断物体是否适合放小物体（桌子、台面、书桌等）。"""
    if not hasattr(obj, "category") or not obj.category:
        return False
    cat = obj.category.lower()
    return any(kw in cat for kw in TABLE_LIKE_KEYWORDS)


def _is_soft_seat(obj) -> bool:
    """沙发、扶手椅等软体座面易穿模，需更大生成高度且减小抖动。"""
    if not hasattr(obj, "category") or not obj.category:
        return False
    cat = obj.category.lower()
    return "sofa" in cat or "armchair" in cat or "ottoman" in cat


def _get_surface_position(obj, offset_z: float = 0.02) -> np.ndarray:
    """返回物体上表面中心的世界坐标 [x,y,z]（略高于表面）。若无效则返回 None。"""
    try:
        center = obj.aabb_center
        extent = obj.aabb_extent
        if th.is_tensor(center):
            center = center.cpu().numpy()
        if th.is_tensor(extent):
            extent = extent.cpu().numpy()
        center = np.asarray(center, dtype=np.float64)
        extent = np.asarray(extent, dtype=np.float64)
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(extent)) or np.any(extent < 0):
            return None
        pos = np.array(center, dtype=np.float64)
        pos[2] = center[2] + extent[2] / 2.0 + offset_z
        return pos if np.all(np.isfinite(pos)) else None
    except Exception:
        return None


def _get_available_categories_and_models(dataset_name: str = "behavior-1k-assets") -> List[Tuple[str, str]]:
    """返回 (category, model) 列表，只包含 datasets 里存在的类别和模型。"""
    out = []
    for cat in SMALL_GRASPABLE_CATEGORIES:
        try:
            models = get_all_object_category_models(cat)
        except Exception:
            continue
        if not models:
            continue
        for m in models:
            if m.startswith(".") or m in ("usd", "shape", "material", "misc", "visualizations"):
                continue
            out.append((cat, m))
    return out


def spawn_small_objects(
    scene,
    num_objects: int = 6,
    categories: Optional[List[str]] = None,
    table_names: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    在场景中的桌子/台面上生成可抓取的小物体。

    Args:
        scene: OmniGibson 的 scene（env.scene）
        num_objects: 生成数量
        categories: 限定只从这些类别中选；None 表示用内置的 SMALL_GRASPABLE_CATEGORIES
        table_names: 限定只在这些物体上放置（按 name）；None 表示自动选所有“桌子类”物体
        rng: 随机数生成器；None 则用 random 默认

    Returns:
        生成的物体 name 列表，可用于 pick_place 的 object_name / target_name。
    """
    if rng is None:
        rng = random.Random()

    # 可选类别过滤
    if categories is not None:
        avail = [
            (c, m)
            for c in categories
            for m in (get_all_object_category_models(c) or [])
            if not m.startswith(".") and m not in ("usd", "shape", "material", "misc", "visualizations")
        ]
    else:
        avail = _get_available_categories_and_models()

    if not avail:
        return []

    # 确定“桌子”列表
    if table_names is not None:
        tables = []
        for nm in table_names:
            if nm not in scene.object_registry.object_names:
                continue
            try:
                obj = scene.object_registry("name", nm)
                if obj is not None:
                    tables.append(obj)
            except Exception:
                continue
    else:
        tables = [obj for obj in scene.objects if _is_table_like(obj)]

    if not tables:
        return []

    from omnigibson.objects.dataset_object import DatasetObject

    added_names = []
    for i in range(num_objects):
        category, model = rng.choice(avail)
        table = rng.choice(tables)
        # 唯一名字，避免和场景已有物体冲突
        base_name = f"{category}_{model}_spawn"
        name = base_name
        idx = 0
        while name in scene.object_registry.object_names:
            idx += 1
            name = f"{base_name}_{idx}"

        try:
            obj = DatasetObject(
                name=name,
                category=category,
                model=model,
                dataset_name="behavior-1k-assets",
            )
        except Exception:
            continue

        offset = 0.08 if _is_soft_seat(table) else 0.03
        surface_pos = _get_surface_position(table, offset_z=offset)
        if surface_pos is None or not np.all(np.isfinite(surface_pos)):
            continue
        # 在桌面范围内加一点随机偏移，避免叠在一起；沙发等软座面减小抖动防穿模
        if hasattr(table, "aabb_extent"):
            try:
                ext = table.aabb_extent
                if th.is_tensor(ext):
                    ext = ext.cpu().numpy()
                ext = np.asarray(ext, dtype=np.float64)
                if np.all(np.isfinite(ext)) and np.all(ext >= 0):
                    jitter_scale = 0.12 if _is_soft_seat(table) else 0.3
                    jitter = np.array(
                        [
                            rng.uniform(-max(0, ext[0] * jitter_scale), max(0, ext[0] * jitter_scale)),
                            rng.uniform(-max(0, ext[1] * jitter_scale), max(0, ext[1] * jitter_scale)),
                            0.0,
                        ],
                        dtype=np.float64,
                    )
                    surface_pos = surface_pos + jitter
            except Exception:
                pass
        if not np.all(np.isfinite(surface_pos)):
            continue

        # 使用单位四元数避免退化旋转导致 PhysX/SVD 报错
        orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        try:
            scene.add_object(obj)
            obj.set_position_orientation(position=surface_pos, orientation=orientation)
            added_names.append(name)
        except Exception:
            try:
                scene.remove_object(obj)
            except Exception:
                pass
            continue

    return added_names


def spawn_objects_one_per_category(
    scene,
    categories: List[str],
    surface_names: List[str],
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    每种类别生成一个物体，随机放在 surface_names 中的物体表面。
    用于测试过的「可放置表面」和「可抓取物体」配置。
    """
    if rng is None:
        rng = random.Random()

    surfaces = []
    for nm in surface_names:
        if nm not in scene.object_registry.object_names:
            continue
        try:
            obj = scene.object_registry("name", nm)
            if obj is not None:
                surfaces.append(obj)
        except Exception:
            continue
    if not surfaces:
        return []

    from omnigibson.objects.dataset_object import DatasetObject

    # 打乱表面顺序后按序分配，尽量每个表面只放一个物体
    surfaces_ordered = list(surfaces)
    rng.shuffle(surfaces_ordered)
    added_names = []
    for cat_idx, category in enumerate(categories):
        # 尝试 category 与 category_0（数据集里有的用下划线后缀）
        models = []
        actual_category = category
        for cat in (category, f"{category}_0"):
            try:
                m = get_all_object_category_models(cat)
                if m:
                    models = m
                    actual_category = cat
                    break
            except Exception:
                pass
        valid_models = [
            m for m in models
            if not m.startswith(".") and m not in ("usd", "shape", "material", "misc", "visualizations")
        ]
        if not valid_models:
            continue
        # 随机顺序尝试，直到该类成功生成一个
        rng.shuffle(valid_models)
        spawned_this_category = False
        for model in valid_models:
            if spawned_this_category:
                break
            table = surfaces_ordered[cat_idx % len(surfaces_ordered)]
            base_name = f"{actual_category}_{model}_spawn"
            name = base_name
            idx = 0
            while name in scene.object_registry.object_names:
                idx += 1
                name = f"{base_name}_{idx}"

            try:
                obj = DatasetObject(
                    name=name,
                    category=actual_category,
                    model=model,
                    dataset_name="behavior-1k-assets",
                )
            except Exception:
                continue

            # 沙发/扶手椅等易穿模，用更大高度且中心放置
            offset = 0.08 if _is_soft_seat(table) else 0.03
            surface_pos = _get_surface_position(table, offset_z=offset)
            if surface_pos is None or not np.all(np.isfinite(surface_pos)):
                continue
            if hasattr(table, "aabb_extent"):
                try:
                    ext = table.aabb_extent
                    if th.is_tensor(ext):
                        ext = ext.cpu().numpy()
                    ext = np.asarray(ext, dtype=np.float64)
                    if np.all(np.isfinite(ext)) and np.all(ext >= 0):
                        jitter_scale = 0.12 if _is_soft_seat(table) else 0.3  # 软座面减小抖动，避免落到边缘穿模
                        jitter = np.array(
                            [
                                rng.uniform(-max(0, ext[0] * jitter_scale), max(0, ext[0] * jitter_scale)),
                                rng.uniform(-max(0, ext[1] * jitter_scale), max(0, ext[1] * jitter_scale)),
                                0.0,
                            ],
                            dtype=np.float64,
                        )
                        surface_pos = surface_pos + jitter
                except Exception:
                    pass
            if not np.all(np.isfinite(surface_pos)):
                continue

            orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            try:
                scene.add_object(obj)
                obj.set_position_orientation(position=surface_pos, orientation=orientation)
                added_names.append(name)
                spawned_this_category = True
            except Exception:
                try:
                    scene.remove_object(obj)
                except Exception:
                    pass
                continue

    return added_names
