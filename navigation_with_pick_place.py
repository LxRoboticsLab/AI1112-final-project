"""
Navigation with integrated Pick&Place.

Extends door/navigation with a simplified pick-and-place pipeline.
"""
import os
import random
import time as time_module
import numpy as np
import omnigibson as og
import torch as th
import sys
import select
import termios
import tty
import matplotlib.pyplot as plt
import cv2
import math
from omnigibson.macros import gm
from omnigibson.object_states import Open, AttachedTo
from omnigibson.utils.transform_utils import quat2euler, quat2mat

# Spawn small graspable objects in the scene (0 = none; >0 = count from tests)
SPAWN_SMALL_OBJECTS_COUNT = int(os.environ.get("SPAWN_SMALL_OBJECTS", "11"))

# Surfaces where placement was tested (small objects spawn only on these)
ALLOWED_PLACEMENT_TARGET_NAMES = [
    "armchair_bslhmj_0", "breakfast_table_skczfi_1", "ottoman_miftfy_0",
    "coffee_table_dnsjnv_0", "washer_omeuop_0", "clothes_dryer_zlmnfg_0",
    "desk_rzyfxk_1", "bottom_cabinet_nddvba_0", "bottom_cabinet_immwzb_0",
    "furniture_sink_zexzrc_0", "bookcase_owvfik_1", "bottom_cabinet_qacthv_0",
    "sofa_xhxdqf_0", "coffee_table_qlmqyy_0", "breakfast_table_uhrsex_0",
    "bookcase_owvfik_0", "countertop_tpuwys_3", "burner_pmntxh_0",
    "countertop_tpuwys_0", "breakfast_table_skczfi_0", "oven_wuinhm_0",
    "fridge_dszchb_0", "furniture_sink_czyfhq_0",
]

# Graspable categories to spawn (one each, random on surfaces above)
SPAWN_OBJECT_CATEGORIES = [
    "apple", "pillow", "bottle_of_champagne",
    "bagel", "banana", "can", "mug", "pen", "cell_phone",
]

# Approach these targets from the smaller-y side (avoid wall penetration)
PLACEMENT_FROM_SMALLER_Y_NAMES = {"oven_wuinhm_0", "fridge_dszchb_0", "countertop_tpuwys_0"}
# Place inside the volume, not on top (e.g. fridge interior)
PLACEMENT_INSIDE_TARGET_NAMES = {"fridge_dszchb_0"}

# Reuse navigation helpers from navigation_with_doors
from navigation_with_doors import (
    NonBlockingInput,
    MapVisualizer,
    list_all_doors,
    print_doors,
    get_door_map_region,
    update_traversable_map,
    control_door,
    compute_action_to_waypoint,
    _pure_pursuit_carrot,
    _advance_waypoint_idx_on_path,
    _normalize_path_waypoints_list,
    _initial_path_waypoint_idx,
    _as_xy2,
    _NAV_LOOKAHEAD_M,
    navigate_to_goal,
    set_camera_resolution,
    init_navigation_state,
    navigate_step,
    nav_step,
)

# FlatCache for performance
gm.ENABLE_FLATCACHE = True

# Manually updated attached objects (follows gripper)
_manually_attached_objects = {}

# Per-object grasp attempt count (incremented on successful attach_object_to_robot).
# Reset after successful place: next new task can trigger drop sim again on first grasp.
# Mid-task drop/retry: count >=2, no more drop on regrasp.
_grasp_attach_count_by_object = {}

# After first grasp: Bernoulli drop at t≈1s and t≈2s (first grasp only), probability from GRASP_DROP_SIM_P.
# Disable all drops via ENABLE_GRASP_DROP_SIM=0.
def _grasp_drop_sim_enabled():
    return os.environ.get("ENABLE_GRASP_DROP_SIM", "1").strip().lower() not in ("0", "false", "no")


def _grasp_drop_sim_probability() -> float:
    try:
        return max(0.0, min(1.0, float(os.environ.get("GRASP_DROP_SIM_P", "0.6"))))
    except ValueError:
        return 0.6


# Test: on first successful grasp, force drop on first update_manually_attached_objects (bypasses random window).
def _grasp_first_attach_always_drop():
    return os.environ.get("GRASP_FIRST_ATTACH_ALWAYS_DROP", "0").strip().lower() in ("1", "true", "yes")

# Category keywords for placement surfaces (tables, counters, etc.); listed as “large” in list)
PLACEMENT_SURFACE_KEYWORDS = ("table", "countertop", "desk", "cabinet", "bookcase", "sofa", "armchair", "chair", "ottoman", "piano", "fridge", "oven", "dishwasher", "toilet", "furniture_sink", "cedar_chest", "floor_lamp", "carpet", "rail_fence", "wall_mounted_tv", "burner", "mirror")

# Non-manipulable building-structure categories
NON_MANIPULABLE_CATEGORIES = {
    "door", "sliding_door", "window", "ceiling", "wall", "floor", 
    "walls", "ceilings", "floors", "ground", "ground_plane",
    "room", "room_floor", "room_wall", "room_ceiling"
}


def is_manipulable_object(obj):
    """
    Return True if the object is manipulable (exclude doors, windows, structural building parts).

    Args:
        obj: object handle

    Returns:
        bool
    """
    if not hasattr(obj, 'category'):
        return False
    
    category = obj.category.lower() if obj.category else ""
    
    if category in NON_MANIPULABLE_CATEGORIES:
        return False
    
    non_manipulable_keywords = ["door", "window", "ceiling", "wall", "floor", "ground"]
    for keyword in non_manipulable_keywords:
        if keyword in category:
            return False
    
    if hasattr(obj, 'fixed_base') and obj.fixed_base:
        # Some fixed objects (e.g. tables) are still placement targets; filter by category above
        pass
    
    return True


def _is_placement_surface(obj):
    """True for large placement surfaces (tables, counters, cabinets); distinct from small graspables in list."""
    if not hasattr(obj, "category") or not obj.category:
        return False
    cat = obj.category.lower()
    return any(kw in cat for kw in PLACEMENT_SURFACE_KEYWORDS)


def is_spawned_graspable_object(obj):
    """Only objects added by spawn_small_objects (name contains _spawn) are graspable; scene-native objects are not."""
    if not is_manipulable_object(obj):
        return False
    name = getattr(obj, "name", "") or ""
    return "_spawn" in name


def get_object_by_name(scene, object_name):
    """
    Find object by name; returns only manipulable objects.

    Args:
        scene: scene handle
        object_name (str): object name

    Returns:
        BaseObject or None
    """
    try:
        obj = scene.object_registry("name", object_name)
        if not is_manipulable_object(obj):
            print(f"⚠️  Object {object_name} is non-manipulable building structure (door/window/wall/...)")
            return None
        return obj
    except:
        print(f"❌ Object not found: {object_name}")
        return None


def list_all_objects(scene):
    """
    List objects: only newly spawned small graspables can be picked; placement targets are from allowed surfaces only.
    """
    # Graspable = only spawned (name contains _spawn)
    graspable_objects = [obj for obj in scene.objects if is_spawned_graspable_object(obj)]
    placement_only = []
    for nm in ALLOWED_PLACEMENT_TARGET_NAMES:
        if nm not in scene.object_registry.object_names:
            continue
        try:
            obj = scene.object_registry("name", nm)
            if obj is not None:
                placement_only.append(obj)
        except Exception:
            continue

    def _print_obj(i, obj):
        name = obj.name if hasattr(obj, "name") else "unknown"
        category = obj.category if hasattr(obj, "category") else "unknown"
        pos, _ = obj.get_position_orientation()
        if th.is_tensor(pos):
            pos = pos.cpu().numpy()
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        print(f"  #{i}: {name}")
        print(f"      category: {category}  pos: {pos_str}")

    print("\n" + "="*80)
    print("📦 Object list")
    print("="*80)

    print("\n🫳 Graspable small objects (spawned only; use as object to pick):")
    if not graspable_objects:
        print("  (none; set SPAWN_SMALL_OBJECTS>0 to spawn at startup)")
    else:
        for i, obj in enumerate(graspable_objects, 1):
            _print_obj(i, obj)
        print(f"  total: {len(graspable_objects)}")

    print("\n🪑 Placement targets (scene-native; for placing only, not graspable):")
    if not placement_only:
        print("  (none)")
    else:
        for i, obj in enumerate(placement_only, 1):
            _print_obj(i, obj)
        print(f"  total: {len(placement_only)}")

    print("\n" + "="*80)
    print("Usage: pick <graspable_name> <placement_target_name>  (only spawned small objects are graspable)")
    print("="*80)


def get_objects_list_for_api(scene):
    """
    Same as list_all_objects: graspable names and placement target names for the HTTP API.

    Returns:
        tuple: (graspable_names, placement_names)
    """
    graspable_objects = [obj for obj in scene.objects if is_spawned_graspable_object(obj)]
    graspable_names = [getattr(obj, "name", "") or "" for obj in graspable_objects if getattr(obj, "name", "")]
    placement_only = []
    for nm in ALLOWED_PLACEMENT_TARGET_NAMES:
        if nm not in scene.object_registry.object_names:
            continue
        try:
            obj = scene.object_registry("name", nm)
            if obj is not None:
                placement_only.append(nm)
        except Exception:
            continue
    return graspable_names, placement_only


def get_placement_targets(scene):
    """
    Return placement target objects in the same order as list_all_objects (for tests).
    """
    manipulable = [obj for obj in scene.objects if is_manipulable_object(obj)]
    return [obj for obj in manipulable if not is_spawned_graspable_object(obj)]


def _set_attached_object_gravity(obj, enable):
    """
    Disable gravity while attached; restore after detach. Gravity only; avoids view invalidation issues.
    """
    if not hasattr(obj, "disable_gravity") or not hasattr(obj, "enable_gravity"):
        return
    try:
        if enable:
            obj.enable_gravity()
        else:
            obj.disable_gravity()
    except Exception as e:
        print(f"⚠️  Error setting gravity for {obj.name}: {e}")


def _add_robot_object_collision_filter(obj, robot):
    """
    While attached, filter collisions between object and robot (e.g. large pillows). Match AttachedTo: pause sim to edit.
    """
    if not hasattr(obj, "links") or not hasattr(robot, "links"):
        return
    try:
        was_playing = og.sim.is_playing()
        if was_playing:
            state = og.sim.dump_state()
            og.sim.stop()
        for obj_link in obj.links.values():
            for robot_link in robot.links.values():
                if hasattr(obj_link, "add_filtered_collision_pair") and hasattr(robot_link, "prim_path"):
                    obj_link.add_filtered_collision_pair(robot_link)
        if was_playing:
            og.sim.play()
            og.sim.load_state(state)
    except Exception as e:
        print(f"⚠️  Error adding robot-object collision filter: {e}")


def _remove_robot_object_collision_filter(obj, robot):
    """Restore object-robot collision on detach."""
    if not hasattr(obj, "links") or not hasattr(robot, "links"):
        return
    try:
        was_playing = og.sim.is_playing()
        if was_playing:
            state = og.sim.dump_state()
            og.sim.stop()
        for obj_link in obj.links.values():
            for robot_link in robot.links.values():
                if hasattr(obj_link, "remove_filtered_collision_pair") and hasattr(robot_link, "prim_path"):
                    obj_link.remove_filtered_collision_pair(robot_link)
        if was_playing:
            og.sim.play()
            og.sim.load_state(state)
    except Exception as e:
        print(f"⚠️  Error removing robot-object collision filter: {e}")


def _get_eef_position(robot):
    """World position of the end effector (gripper)."""
    if hasattr(robot, "eef_links") and robot.eef_links:
        arm = robot.default_arm if hasattr(robot, "default_arm") else list(robot.eef_links.keys())[0]
        eef_link = robot.eef_links[arm]
        pos, _ = eef_link.get_position_orientation()
        if th.is_tensor(pos):
            pos = pos.cpu().numpy()
        return np.asarray(pos, dtype=np.float64)
    return None


def attach_object_to_robot(obj, robot):
    """
    Teleport object to gripper, then manual attach (follows gripper each frame).
    """
    global _manually_attached_objects, _grasp_attach_count_by_object

    n = _grasp_attach_count_by_object.get(obj.name, 0) + 1
    _grasp_attach_count_by_object[obj.name] = n
    drop_eligible = _grasp_drop_sim_enabled() and (n == 1)
    guaranteed_first_update_drop = _grasp_first_attach_always_drop() and (n == 1)

    _set_attached_object_gravity(obj, False)
    _add_robot_object_collision_filter(obj, robot)

    eef_pos = _get_eef_position(robot)
    if eef_pos is not None:
        obj.set_position_orientation(position=eef_pos)
        print(f"✅ Object {obj.name} teleported to gripper [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]")

    _manually_attached_objects[obj.name] = {
        'object': obj,
        'robot': robot,
        'obj_orn': None,
        'grasp_attempt_index': n,
        'drop_sim_active': drop_eligible and (not guaranteed_first_update_drop),
        'drop_sim_t0': time_module.monotonic() if drop_eligible else None,
        'drop_sim_seconds_checked': set(),
    }
    print(f"✅ Object {obj.name} attached (no gravity, filtered collision, follows gripper), grasp attempt #{n}")

    if guaranteed_first_update_drop:
        print(f"🧪 [test] {obj.name} forced drop on first attach (GRASP_FIRST_ATTACH_ALWAYS_DROP)", flush=True)
        _do_drop(obj, robot, obj.name)
    return True


def detach_object_from_robot(obj, robot, reason="default"):
    """
    Detach object from robot.

    Args:
        obj: object to detach
        robot: robot
        reason: "place" clears per-object grasp count; failure/drop_sim/default do not.

    Returns:
        bool
    """
    global _manually_attached_objects, _grasp_attach_count_by_object
    
    # In manual attach list
    if obj.name in _manually_attached_objects:
        attachment_info = _manually_attached_objects[obj.name]
        robot = attachment_info["robot"]
        _remove_robot_object_collision_filter(obj, robot)
        _set_attached_object_gravity(obj, True)
        del _manually_attached_objects[obj.name]
        if reason == "place":
            _grasp_attach_count_by_object[obj.name] = 0
        print(f"✅ Object {obj.name} removed from manual attach list; gravity and collision restored")
        return True
    
    # Try AttachedTo state
    if AttachedTo in obj.states:
        try:
            if obj.states[AttachedTo].get_value(robot):
                success = obj.states[AttachedTo].set_value(robot, False)
                if success:
                    print(f"✅ Object {obj.name} detached (AttachedTo state)")
                    return True
        except Exception as e:
            print(f"⚠️  Error detaching object: {e}")
    
    print(f"ℹ️  Object {obj.name} was not attached to the robot")
    return True


def release_abandoned_pick_place(pp_state, robot):
    """
    On preempt/cancel/discard of a pick_place segment: detach if still manually attached,
    restore recorded pick pose (even if object already fell off the bind list).
    """
    if not pp_state or not isinstance(pp_state, dict):
        return
    obj = pp_state.get("obj")
    if obj is None or getattr(obj, "name", None) is None:
        return
    if obj.name in _manually_attached_objects:
        detach_object_from_robot(obj, robot, reason="failure")
    pick_pos = pp_state.get("pick_pos")
    pick_orn = pp_state.get("pick_orn")
    if pick_pos is not None and pick_orn is not None:
        try:
            obj.set_position_orientation(position=pick_pos, orientation=pick_orn)
        except Exception:
            pass


# Fixed offset for attached object in robot frame (m); +X forward, +Z up, avoid dragging on floor
ATTACHED_OBJECT_OFFSET_IN_ROBOT_FRAME = np.array([0.25, 0.0, 0.5], dtype=np.float64)  # 25cm forward, 50cm up


def _maybe_apply_grasp_drop_simulation():
    """
    Random drop sim (only grasp_attempt_index==1 and ENABLE_GRASP_DROP_SIM on):
    at t≈1s and t≈2s after first grasp, each with independent probability GRASP_DROP_SIM_P (default 0.6).
    Later grasps do not trigger.

    GRASP_FIRST_ATTACH_ALWAYS_DROP is handled in attach_object_to_robot, not here.
    """
    for obj_name, attachment_info in list(_manually_attached_objects.items()):
        if not attachment_info.get("drop_sim_active"):
            continue
        if attachment_info.get("grasp_attempt_index", 1) != 1:
            continue
        t0 = attachment_info.get("drop_sim_t0")
        if t0 is None:
            continue
        elapsed = time_module.monotonic() - t0
        obj = attachment_info["object"]
        robot = attachment_info["robot"]

        checked = attachment_info.setdefault("drop_sim_seconds_checked", set())
        p_drop = _grasp_drop_sim_probability()
        for sec in (1, 2):
            if elapsed < float(sec) or sec in checked:
                continue
            checked.add(sec)
            if random.random() < p_drop:
                print(f"🎲 [drop sim] {obj_name} dropped at t={sec}s (first grasp, p={p_drop})", flush=True)
                _do_drop(obj, robot, obj_name)
                break
        if elapsed > 2.0:
            attachment_info["drop_sim_active"] = False


def _do_drop(obj, robot, obj_name: str):
    """Drop: detach and place object under gripper."""
    eef_pos = _get_eef_position(robot)
    detach_object_from_robot(obj, robot, reason="drop_sim")
    if eef_pos is not None:
        drop_pos = np.array(eef_pos, dtype=np.float64)
        drop_pos[2] = max(drop_pos[2] - 0.08, 0.05)
        try:
            obj.set_position_orientation(position=drop_pos)
        except Exception:
            pass


def get_grasp_status_for_api():
    """
    For HTTP polling: whether holding (manual attach), object name, grasp attempt index.
    Client may use with gripper camera to re-issue pick_place.
    """
    if not _manually_attached_objects:
        return {
            "holding": False,
            "object_name": None,
            "grasp_attempt_index": 0,
        }
    obj_name = next(iter(_manually_attached_objects.keys()))
    info = _manually_attached_objects[obj_name]
    return {
        "holding": True,
        "object_name": obj_name,
        "grasp_attempt_index": int(info.get("grasp_attempt_index", 1)),
    }


def update_manually_attached_objects(skip_drop_sim: bool = False):
    """
    Update manually attached object pose to follow gripper (eef).
    skip_drop_sim=True: skip random drop (e.g. nav stuck pause) to avoid wall-clock false drops.
    """
    global _manually_attached_objects

    if not skip_drop_sim:
        _maybe_apply_grasp_drop_simulation()
    
    for obj_name, attachment_info in list(_manually_attached_objects.items()):
        obj = attachment_info['object']
        robot = attachment_info['robot']
        
        try:
            eef_pos = _get_eef_position(robot)
            if eef_pos is None:
                continue

            if attachment_info['obj_orn'] is None:
                _, obj_orn = obj.get_position_orientation()
                if th.is_tensor(obj_orn):
                    obj_orn = obj_orn.cpu().numpy()
                attachment_info['obj_orn'] = np.asarray(obj_orn, dtype=np.float64)
                print(f"📐 Object {obj_name} will follow the gripper")
            obj_orn = attachment_info['obj_orn']

            obj.set_position_orientation(position=eef_pos, orientation=obj_orn)
            if hasattr(obj, "set_linear_velocity") and hasattr(obj, "set_angular_velocity"):
                try:
                    obj.set_linear_velocity(velocity=th.zeros(3))
                    obj.set_angular_velocity(velocity=th.zeros(3))
                except Exception:
                    pass
            
        except Exception as e:
            print(f"⚠️  Error updating object {obj_name} pose: {e}")


def calculate_place_position(target_obj, place_obj=None, surface_clearance=0.005, target_name=None):
    """
    Compute place position: on top of target surface, or inside (e.g. fridge) if in PLACEMENT_INSIDE_TARGET_NAMES.

    Args:
        target_obj: placement target
        place_obj: object being placed (for height)
        surface_clearance: small gap above surface (m)
        target_name: if in PLACEMENT_INSIDE_TARGET_NAMES, place inside volume

    Returns:
        np.array: world position of object center
    """
    target_aabb_center = target_obj.aabb_center
    target_aabb_extent = target_obj.aabb_extent
    if th.is_tensor(target_aabb_center):
        target_aabb_center = target_aabb_center.cpu().numpy()
    if th.is_tensor(target_aabb_extent):
        target_aabb_extent = target_aabb_extent.cpu().numpy()
    
    place_pos = np.array(target_aabb_center, dtype=np.float64)
    inside = target_name is not None and target_name in PLACEMENT_INSIDE_TARGET_NAMES
    
    if inside:
        # Interior: use AABB center (avoid floor of fridge)
        place_pos[0] = target_aabb_center[0]
        place_pos[1] = target_aabb_center[1]
        place_pos[2] = target_aabb_center[2]
    else:
        target_top_z = target_aabb_center[2] + target_aabb_extent[2] / 2.0
        place_pos[2] = target_top_z + surface_clearance
        if place_obj is not None:
            ext = place_obj.aabb_extent
            if th.is_tensor(ext):
                ext = ext.cpu().numpy()
            half_h = float(ext[2]) / 2.0
            place_pos[2] = target_top_z + half_h + surface_clearance
    
    return place_pos


def navigate_to_goal_with_object_update(env, robot, scene, floor, goal_pos, visualizer):
    """
    Navigate to goal and update manually attached object each step (variant of navigate_to_goal).
    """
    print("\nPlanning path with A*...")
    robot_pos_ori = robot.get_position_orientation()
    start_pos = robot_pos_ori[0][:2]
    
    path_waypoints, geodesic_distance = scene.get_shortest_path(
        floor=floor,
        source_world=start_pos,
        target_world=goal_pos[:2],
        entire_path=True,
        robot=None
    )
    
    if path_waypoints is None:
        print("❌ No path found! You may need to open a door...")
        target_pos_np = goal_pos[:2].cpu().numpy() if th.is_tensor(goal_pos) else np.asarray(goal_pos[:2], dtype=np.float64)
        # Diagnose: same eroded map as get_shortest_path
        try:
            eroded = _get_eroded_trav_map(scene, floor, robot=None)
            if eroded is not None:
                tmap = scene.trav_map
                src_val = _cell_value_on_map(eroded, tmap, start_pos)
                tgt_val = _cell_value_on_map(eroded, tmap, target_pos_np)
                src_mp = tmap.world_to_map(start_pos)
                tgt_mp = tmap.world_to_map(target_pos_np)
                if th.is_tensor(src_mp):
                    src_mp = src_mp.cpu().numpy()
                if th.is_tensor(tgt_mp):
                    tgt_mp = tgt_mp.cpu().numpy()
                print(f"   [diag] start [{start_pos[0]:.2f}, {start_pos[1]:.2f}] map px ({int(src_mp[0])}, {int(src_mp[1])}) eroded={src_val}")
                print(f"   [diag] goal [{target_pos_np[0]:.2f}, {target_pos_np[1]:.2f}] map px ({int(tgt_mp[0])}, {int(tgt_mp[1])}) eroded={tgt_val}")
                if (src_val is None or src_val == 0) or (tgt_val is None or tgt_val == 0):
                    print(f"   [diag] start or goal not free after erosion; check obstacle proximity or larger approach radius")
                else:
                    print(f"   [diag] start and goal free but no path; try opening a door, then pick again")
        except Exception as e:
            print(f"   [diag] map check error: {e}")
        if visualizer is not None:
            visualizer.add_debug_goal(target_pos_np)
            robot_pos_2d = robot.get_position_orientation()[0][:2]
            if th.is_tensor(robot_pos_2d):
                robot_pos_2d = robot_pos_2d.cpu().numpy()
            visualizer.update(
                robot_pos=robot_pos_2d,
                goal_pos=target_pos_np,
                path=None,
                current_waypoint=None,
            )
            n = len(visualizer.debug_goal_positions)
            print(f"   Marked unreachable goal on map: [{target_pos_np[0]:.2f}, {target_pos_np[1]:.2f}] ({n} total, purple star)")
        return False
    
    print(f"✅ Path found! length={geodesic_distance:.2f}m, waypoints={len(path_waypoints)}")
    
    print("\nNavigating along polyline toward goal...")
    max_steps = 5000
    step_count = 0
    
    pl = _normalize_path_waypoints_list(path_waypoints)
    path_for_viz = [np.asarray(_as_xy2(w), dtype=np.float64).copy() for w in pl] if pl else []
    n_wp = len(pl)
    
    target_pos_np = goal_pos[:2].cpu().numpy() if th.is_tensor(goal_pos) else goal_pos[:2]
    arrival_threshold = 0.1  # m
    
    rp0 = robot.get_position_orientation()[0][:2]
    rp0_np = rp0.cpu().numpy() if th.is_tensor(rp0) else np.asarray(rp0, dtype=np.float64)
    start_xy = np.asarray(rp0_np, dtype=np.float64).reshape(-1)[:2]
    current_waypoint_idx = _initial_path_waypoint_idx(start_xy, path_waypoints)
    
    while current_waypoint_idx < n_wp and step_count < max_steps:
        current_waypoint = path_waypoints[current_waypoint_idx]
        current_waypoint_np = (
            current_waypoint.cpu().numpy() if th.is_tensor(current_waypoint) else np.asarray(current_waypoint)
        )
        rp0 = robot.get_position_orientation()[0][:2]
        rp0_np = rp0.cpu().numpy() if th.is_tensor(rp0) else np.asarray(rp0)
        carrot = _pure_pursuit_carrot(rp0_np, path_waypoints, current_waypoint_idx, _NAV_LOOKAHEAD_M)
        track_xy = carrot if carrot is not None else _as_xy2(current_waypoint_np)
        base_action = compute_action_to_waypoint(robot, track_xy)
        nav_step(env, robot, base_action, update_manually_attached_objects)
        robot_pos_after = robot.get_position_orientation()[0][:2]
        robot_pos_after_np = robot_pos_after.cpu().numpy() if th.is_tensor(robot_pos_after) else np.asarray(robot_pos_after)
        current_waypoint_idx = _advance_waypoint_idx_on_path(
            robot_pos_after_np, path_waypoints, current_waypoint_idx, 0.3
        )
        step_count += 1
        if visualizer is not None and step_count % 5 == 0:
            visualizer.update(
                robot_pos=robot_pos_after_np,
                goal_pos=target_pos_np,
                path=path_for_viz,
                current_waypoint=current_waypoint_np,
            )
        if step_count % 100 == 0:
            cur = robot.get_position_orientation()[0][:2]
            cur = cur.cpu().numpy() if th.is_tensor(cur) else np.asarray(cur)
            dg = np.linalg.norm(cur - target_pos_np)
            print(f"step: {step_count}, dist_to_goal: {dg:.2f}m, wp: {current_waypoint_idx}/{n_wp}")
    
    if current_waypoint_idx >= n_wp and step_count < max_steps:
        print("✅ Past all waypoints, homing to final goal...")
        while step_count < max_steps:
            robot_pos = robot.get_position_orientation()[0][:2]
            robot_pos_np = robot_pos.cpu().numpy() if th.is_tensor(robot_pos) else np.asarray(robot_pos)
            distance_to_goal = np.linalg.norm(robot_pos_np - target_pos_np)
            if distance_to_goal < arrival_threshold:
                print(f"✅ Reached goal! dist={distance_to_goal:.2f}m, steps={step_count}")
                break
            base_action = compute_action_to_waypoint(robot, target_pos_np)
            nav_step(env, robot, base_action, update_manually_attached_objects)
            robot_pos_after = robot.get_position_orientation()[0][:2]
            robot_pos_after_np = robot_pos_after.cpu().numpy() if th.is_tensor(robot_pos_after) else np.asarray(robot_pos_after)
            step_count += 1
            if visualizer is not None and step_count % 5 == 0:
                visualizer.update(
                    robot_pos=robot_pos_after_np,
                    goal_pos=target_pos_np,
                    path=path_for_viz,
                    current_waypoint=None,
                )
            if step_count % 50 == 0:
                dgc = float(np.linalg.norm(robot_pos_after_np - target_pos_np))
                print(f"step: {step_count}, dist_to_final: {dgc:.2f}m")
    
    stop_action = np.array([0.0, 0.0])
    for _ in range(10):
        nav_step(env, robot, stop_action, update_manually_attached_objects)
    
    final_pos = robot.get_position_orientation()[0][:2]
    if th.is_tensor(final_pos):
        final_pos = final_pos.cpu().numpy()
    final_distance = np.linalg.norm(final_pos - target_pos_np)
    
    if final_distance < arrival_threshold:
        print(f"✅ Reached goal! final_dist={final_distance:.2f}m, steps={step_count}")
        return True
    else:
        print(f"❌ Did not reach goal. final_dist={final_distance:.2f}m, steps={step_count}")
        return False


def _get_eroded_trav_map(scene, floor, robot=None):
    """
    Eroded trav map consistent with get_shortest_path (reachability + diagnostics).
    If robot is None, uses trav_map default_erosion_radius.
    """
    tmap = scene.trav_map
    if floor >= len(tmap.floor_map):
        return None
    import math
    trav = th.clone(tmap.floor_map[floor])
    if robot is not None:
        ext = robot.reset_joint_pos_aabb_extent[:2]
        if th.is_tensor(ext):
            radius = th.norm(ext).item() / 2.0 + 0.2
        else:
            radius = float(np.linalg.norm(ext)) / 2.0 + 0.2
    else:
        radius = getattr(tmap, "default_erosion_radius", 0.0)
    radius_pixel = max(0, int(math.ceil(radius / tmap.map_resolution)))
    if radius_pixel > 0:
        kern = np.ones((radius_pixel, radius_pixel), dtype=np.uint8)
        trav = th.tensor(cv2.erode(trav.cpu().numpy(), kern))
    return trav


def _cell_value_on_map(trav_map, tmap, world_xy):
    """Pixel value of world_xy on trav_map."""
    mp = tmap.world_to_map(world_xy)
    if th.is_tensor(mp):
        r, c = int(mp[0].item()), int(mp[1].item())
    else:
        r, c = int(mp[0]), int(mp[1])
    h, w = trav_map.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return None
    cell = trav_map[r, c]
    return cell.item() if th.is_tensor(cell) else cell


def _is_traversable(scene, floor, world_xy, use_eroded=True):
    """
    True if (x,y) is on traversable map. use_eroded=True uses same erosion as get_shortest_path.
    """
    tmap = scene.trav_map
    if floor >= len(tmap.floor_map):
        return False
    trav = _get_eroded_trav_map(scene, floor, robot=None) if use_eroded else tmap.floor_map[floor]
    if trav is None:
        return False
    v = _cell_value_on_map(trav, tmap, world_xy)
    return v is not None and v != 0


def calculate_reachable_position_near_object(obj, robot_pos, scene, floor, operation_radius=0.8, target_name=None):
    """
    A navigable point near the object (avoids goals on table/obstacles that break A*).
    If target_name in PLACEMENT_FROM_SMALLER_Y_NAMES, prefer approaching from smaller y (ovens/fridges).
    """
    obj_aabb_center = obj.aabb_center
    obj_aabb_extent = obj.aabb_extent
    if th.is_tensor(obj_aabb_center):
        obj_aabb_center = obj_aabb_center.cpu().numpy()
    if th.is_tensor(obj_aabb_extent):
        obj_aabb_extent = obj_aabb_extent.cpu().numpy()

    obj_center_2d = np.array(obj_aabb_center[:2], dtype=np.float64)
    obj_size_xy = max(obj_aabb_extent[0], obj_aabb_extent[1])
    base_distance = obj_size_xy / 2.0 + operation_radius
    floor_height = scene.get_floor_height(floor)
    require_smaller_y = target_name is not None and target_name in PLACEMENT_FROM_SMALLER_Y_NAMES

    def try_candidate(candidate):
        if _is_traversable(scene, floor, candidate):
            return np.array([candidate[0], candidate[1], floor_height])
        return None

    if require_smaller_y:
        for radius_scale in [1.0, 1.2, 1.5]:
            d = base_distance * radius_scale
            c = obj_center_2d + np.array([0.0, -d])
            out = try_candidate(c)
            if out is not None:
                return out
            for i in range(12):
                angle = -np.pi * (i + 1) / 13
                dx, dy = d * np.cos(angle), d * np.sin(angle)
                if dy >= 0:
                    continue
                c = obj_center_2d + np.array([dx, dy])
                out = try_candidate(c)
                if out is not None:
                    return out
        for radius_scale in [1.0, 1.2, 1.5]:
            d = base_distance * radius_scale
            for i in range(24):
                angle = 2 * np.pi * i / 24
                c = obj_center_2d + d * np.array([np.cos(angle), np.sin(angle)])
                out = try_candidate(c)
                if out is not None:
                    return out
        fallback = obj_center_2d + np.array([0.0, -base_distance])
        return np.array([fallback[0], fallback[1], floor_height])

    robot_pos_2d = np.array(robot_pos[:2], dtype=np.float64)
    direction_2d = obj_center_2d - robot_pos_2d
    distance_to_center = np.linalg.norm(direction_2d)
    if distance_to_center > 1e-6:
        direction_normalized = direction_2d / distance_to_center
        candidate = obj_center_2d - direction_normalized * base_distance
        out = try_candidate(candidate)
        if out is not None:
            return out

    for radius_scale in [1.0, 1.2, 1.5]:
        d = base_distance * radius_scale
        for i in range(24):
            angle = 2 * np.pi * i / 24
            dx, dy = d * np.cos(angle), d * np.sin(angle)
            candidate = obj_center_2d + np.array([dx, dy])
            out = try_candidate(candidate)
            if out is not None:
                return out

    if distance_to_center > 1e-6:
        target_pos_2d = obj_center_2d - direction_normalized * base_distance
    else:
        target_pos_2d = obj_center_2d + np.array([base_distance, 0])
    return np.array([target_pos_2d[0], target_pos_2d[1], floor_height])


def iter_reachable_candidates_near_object(obj, robot_pos, scene, floor, operation_radius=0.8, target_name=None, max_candidates=16):
    """
    Yields reachable approach points near the object in priority order.
    If the first target fails, try the next. target_name: placement constraints (or None for grasp only).
    """
    obj_aabb_center = obj.aabb_center
    obj_aabb_extent = obj.aabb_extent
    if th.is_tensor(obj_aabb_center):
        obj_aabb_center = obj_aabb_center.cpu().numpy()
    if th.is_tensor(obj_aabb_extent):
        obj_aabb_extent = obj_aabb_extent.cpu().numpy()

    obj_center_2d = np.array(obj_aabb_center[:2], dtype=np.float64)
    obj_size_xy = max(obj_aabb_extent[0], obj_aabb_extent[1])
    base_distance = obj_size_xy / 2.0 + operation_radius
    floor_height = scene.get_floor_height(floor)
    require_smaller_y = target_name is not None and target_name in PLACEMENT_FROM_SMALLER_Y_NAMES
    seen = set()  # dedupe (x,y) rounded

    def add_candidate(candidate_2d):
        key = (round(float(candidate_2d[0]), 3), round(float(candidate_2d[1]), 3))
        if key in seen:
            return None
        if not _is_traversable(scene, floor, candidate_2d):
            return None
        seen.add(key)
        return np.array([candidate_2d[0], candidate_2d[1], floor_height])

    count = 0
    if require_smaller_y:
        for radius_scale in [1.0, 1.2, 1.5]:
            d = base_distance * radius_scale
            c = obj_center_2d + np.array([0.0, -d])
            out = add_candidate(c)
            if out is not None:
                count += 1
                yield out
                if count >= max_candidates:
                    return
            for i in range(12):
                angle = -np.pi * (i + 1) / 13
                dx, dy = d * np.cos(angle), d * np.sin(angle)
                if dy >= 0:
                    continue
                c = obj_center_2d + np.array([dx, dy])
                out = add_candidate(c)
                if out is not None:
                    count += 1
                    yield out
                    if count >= max_candidates:
                        return
        for radius_scale in [1.0, 1.2, 1.5]:
            d = base_distance * radius_scale
            for i in range(24):
                angle = 2 * np.pi * i / 24
                c = obj_center_2d + d * np.array([np.cos(angle), np.sin(angle)])
                out = add_candidate(c)
                if out is not None:
                    count += 1
                    yield out
                    if count >= max_candidates:
                        return
        return

    robot_pos_2d = np.array(robot_pos[:2], dtype=np.float64)
    direction_2d = obj_center_2d - robot_pos_2d
    distance_to_center = np.linalg.norm(direction_2d)
    if distance_to_center > 1e-6:
        direction_normalized = direction_2d / distance_to_center
        c = obj_center_2d - direction_normalized * base_distance
        out = add_candidate(c)
        if out is not None:
            count += 1
            yield out
            if count >= max_candidates:
                return

    for radius_scale in [1.0, 1.2, 1.5]:
        d = base_distance * radius_scale
        for i in range(24):
            angle = 2 * np.pi * i / 24
            c = obj_center_2d + np.array([d * np.cos(angle), d * np.sin(angle)])
            out = add_candidate(c)
            if out is not None:
                count += 1
                yield out
                if count >= max_candidates:
                    return


# Max nav substeps per pick_place frame so the main loop can also run door control, etc.
PICK_PLACE_NAV_STEPS_PER_CALL = 30


def _stall_until_transport_visual_ready(s: dict, max_steps_per_call: int) -> bool:
    """
    At place distance in nav_to_target: if transport-visual gating is on and hold streak is short,
    step sim only; do not place. Client should POST transport_visual_ack (hold|lost) each frame.
    """
    try:
        import behavior_env_api as be
    except ImportError:
        return False
    if not be.pick_place_transport_visual_gating_enabled():
        return False
    need = be.get_pick_place_min_transport_visual_frames()
    if need <= 0:
        return False
    streak = be.get_transport_visual_hold_streak()
    if streak >= need:
        return False
    if not s.get("printed_place_visual_gate"):
        print(
            f"\n🤖 Pick&Place: at place range, waiting for transport-vision **consecutive HOLD** {streak}/{need} "
            f"(LOST resets streak; client must POST holds)...",
            flush=True,
        )
        s["printed_place_visual_gate"] = True
    n = max(1, min(int(max_steps_per_call), 15))
    for _ in range(n):
        og.sim.step()
        update_manually_attached_objects()
    return True


def _update_pick_place_map_markers(s: dict, visualizer) -> None:
    """Draw pick/place approach points and object/target centers on the map (vs A* goal)."""
    if visualizer is None or not hasattr(visualizer, "set_pick_place_nav_markers"):
        return
    try:
        grasp = np.asarray(s["candidates"][s["candidate_idx"]][:2], dtype=np.float64)
        objp, _ = s["obj"].get_position_orientation()
        if th.is_tensor(objp):
            objp = objp.cpu().numpy()
        obj_xy = np.asarray(objp[:2], dtype=np.float64)
        tv, _ = s["target_obj"].get_position_orientation()
        if th.is_tensor(tv):
            tv = tv.cpu().numpy()
        tgt_xy = np.asarray(tv[:2], dtype=np.float64)
        place_xy = None
        ns = s.get("nav_state")
        if ns is not None and s.get("phase") in ("nav_to_target", "place"):
            tp = ns.get("target_pos_np")
            if tp is not None:
                place_xy = np.asarray(tp, dtype=np.float64)
        visualizer.set_pick_place_nav_markers(
            grasp_nav_xy=grasp,
            place_nav_xy=place_xy,
            object_xy=obj_xy,
            target_surface_xy=tgt_xy,
        )
    except Exception:
        pass


def init_pick_place_state(
    env,
    robot,
    scene,
    floor,
    object_name,
    target_name,
    visualizer=None,
    operation_radius_pick=0.8,
    operation_radius_place=0.5,
    doors=None,
    door_positions=None,
    door_states=None,
    door_map_regions=None,
    door_joint_limits=None,
    skip_doors=None,
    original_map_dict=None,
):
    """
    Init incremental Pick&Place state for pick_place_step, or None on failure.
    """
    obj = get_object_by_name(scene, object_name)
    if obj is None or not is_spawned_graspable_object(obj):
        return None
    # After preempt/cancel, object may still be on gripper; detach or reachability samples cluster on robot
    if obj.name in _manually_attached_objects:
        detach_object_from_robot(obj, robot, reason="failure")
    target_obj = get_object_by_name(scene, target_name)
    if target_obj is None or not is_manipulable_object(target_obj):
        return None
    robot_pos, _ = robot.get_position_orientation()
    if th.is_tensor(robot_pos):
        robot_pos = robot_pos.cpu().numpy()
    robot_pos_2d = robot_pos[:2]
    candidates = list(
        iter_reachable_candidates_near_object(
            obj, robot_pos, scene, floor, operation_radius_pick, target_name=None, max_candidates=16
        )
    )
    if not candidates:
        return None
    if visualizer is not None:
        visualizer.clear_debug_goals()
    return {
        "phase": "nav_to_obj",
        "object_name": object_name,
        "target_name": target_name,
        "obj": obj,
        "target_obj": target_obj,
        "nav_state": None,
        "candidates": candidates,
        "candidate_idx": 0,
        "approach_pos": None,
        "pick_pos": None,
        "pick_orn": None,
        "doors": doors,
        "door_positions": door_positions,
        "door_states": door_states,
        "door_map_regions": door_map_regions,
        "door_joint_limits": door_joint_limits,
        "skip_doors": skip_doors,
        "original_map_dict": original_map_dict,
        "operation_radius_pick": operation_radius_pick,
        "operation_radius_place": operation_radius_place,
        "floor": floor,
        "printed_nav_obj": False,
        "printed_attach": False,
        "printed_nav_target": False,
    }


def pick_place_step(
    env,
    robot,
    scene,
    floor,
    state,
    visualizer,
    max_steps_per_call=30,
):
    """
    One step of incremental Pick&Place. Returns (new_state, done, success, error_msg).
    If done is True, success/error_msg are defined; new_state may be ignored.
    """
    s = dict(state)
    door_args = {
        "doors": s.get("doors"),
        "door_positions": s.get("door_positions"),
        "door_states": s.get("door_states"),
        "door_map_regions": s.get("door_map_regions"),
        "door_joint_limits": s.get("door_joint_limits"),
        "skip_doors": s.get("skip_doors"),
        "original_map_dict": s.get("original_map_dict"),
    }

    _update_pick_place_map_markers(s, visualizer)

    # ---------- phase: nav_to_obj ----------
    if s["phase"] == "nav_to_obj":
        robot_pos_2d = robot.get_position_orientation()[0][:2]
        if th.is_tensor(robot_pos_2d):
            robot_pos_2d = robot_pos_2d.cpu().numpy()
        obj_pos_full = s["obj"].get_position_orientation()[0]
        obj_pos_2d = obj_pos_full[:2]
        if th.is_tensor(obj_pos_2d):
            obj_pos_2d = obj_pos_2d.cpu().numpy()
        dist_robot_to_obj = float(np.linalg.norm(robot_pos_2d - obj_pos_2d))
        if dist_robot_to_obj < s.get("operation_radius_pick", 0.8):
            rp3 = robot.get_position_orientation()[0][:3]
            if th.is_tensor(rp3):
                rp3 = rp3.cpu().numpy()
            s["phase"] = "attach"
            s["approach_pos"] = np.asarray(rp3, dtype=np.float64)
            return pick_place_step(env, robot, scene, floor, s, visualizer, max_steps_per_call)
        approach_pos = s["candidates"][s["candidate_idx"]]
        approach_pos_2d = approach_pos[:2]
        if np.linalg.norm(robot_pos_2d - approach_pos_2d) < 0.1:
            s["phase"] = "attach"
            s["approach_pos"] = approach_pos
            return pick_place_step(env, robot, scene, floor, s, visualizer, max_steps_per_call)
        if s["nav_state"] is None:
            if not s.get("printed_nav_obj"):
                print("\n🤖 Pick&Place step: navigating to object...", flush=True)
                s["printed_nav_obj"] = True
            nav_state = init_navigation_state(
                env, robot, scene, floor, approach_pos, goal_yaw_rad=None,
            )
            if nav_state is None:
                s["candidate_idx"] += 1
                if s["candidate_idx"] >= len(s["candidates"]):
                    return s, True, False, "cannot reach approach near object"
                s["nav_state"] = None
                return s, False, None, None
            s["nav_state"] = nav_state
        status, new_nav = navigate_step(
            env, robot, scene, floor, s["nav_state"], visualizer, max_steps_per_call=max_steps_per_call,
            after_step_callback=update_manually_attached_objects,
        )
        s["nav_state"] = new_nav
        update_manually_attached_objects()
        if status == "reached":
            s["phase"] = "attach"
            s["approach_pos"] = s["candidates"][s["candidate_idx"]]
        elif status == "failed":
            s["nav_state"] = None
            s["candidate_idx"] += 1
            if s["candidate_idx"] >= len(s["candidates"]):
                return s, True, False, "cannot reach approach near object"
        return s, False, None, None

    # ---------- phase: attach ----------
    if s["phase"] == "attach":
        if not s.get("printed_attach"):
            print("\n🤖 Pick&Place step: attaching object...", flush=True)
            s["printed_attach"] = True
        obj = s["obj"]
        pick_pos, pick_orn = obj.get_position_orientation()
        s["pick_pos"] = np.array(pick_pos.cpu().numpy() if th.is_tensor(pick_pos) else pick_pos, dtype=np.float64).copy()
        s["pick_orn"] = np.array(pick_orn.cpu().numpy() if th.is_tensor(pick_orn) else pick_orn, dtype=np.float64).copy()
        for _ in range(10):
            og.sim.step()
            update_manually_attached_objects()
        if not attach_object_to_robot(obj, robot):
            return s, True, False, "attach failed"
        for _ in range(10):
            og.sim.step()
            update_manually_attached_objects()
        s["phase"] = "nav_to_target"
        s["printed_place_visual_gate"] = False
        robot_pos, _ = robot.get_position_orientation()
        if th.is_tensor(robot_pos):
            robot_pos = robot_pos.cpu().numpy()
        place_approach = calculate_reachable_position_near_object(
            s["target_obj"], robot_pos, scene, floor, s["operation_radius_place"], target_name=s["target_name"],
        )
        s["nav_state"] = init_navigation_state(
            env, robot, scene, floor, place_approach, goal_yaw_rad=None,
        )
        if s["nav_state"] is None:
            detach_object_from_robot(obj, robot, reason="failure")
            obj.set_position_orientation(position=s["pick_pos"], orientation=s["pick_orn"])
            return s, True, False, "cannot reach approach near placement target"
        return s, False, None, None

    # ---------- phase: nav_to_target ----------
    if s["phase"] == "nav_to_target":
        robot_pos_2d = robot.get_position_orientation()[0][:2]
        if th.is_tensor(robot_pos_2d):
            robot_pos_2d = robot_pos_2d.cpu().numpy()
        goal_2d = s["nav_state"]["target_pos_np"]
        if np.linalg.norm(robot_pos_2d - goal_2d) < 0.1:
            if _stall_until_transport_visual_ready(s, max_steps_per_call):
                return s, False, None, None
            s["phase"] = "place"
            return pick_place_step(env, robot, scene, floor, s, visualizer, max_steps_per_call)
        if not s.get("printed_nav_target"):
            print("\n🤖 Pick&Place step: navigating to placement target...", flush=True)
            s["printed_nav_target"] = True
        status, new_nav = navigate_step(
            env, robot, scene, floor, s["nav_state"], visualizer, max_steps_per_call=max_steps_per_call,
            after_step_callback=update_manually_attached_objects,
        )
        s["nav_state"] = new_nav
        update_manually_attached_objects()
        if status == "reached":
            if _stall_until_transport_visual_ready(s, max_steps_per_call):
                return s, False, None, None
            s["phase"] = "place"
        elif status == "failed":
            detach_object_from_robot(s["obj"], robot, reason="failure")
            s["obj"].set_position_orientation(position=s["pick_pos"], orientation=s["pick_orn"])
            return s, True, False, "cannot reach approach near placement target"
        return s, False, None, None

    # ---------- phase: place ----------
    if s["phase"] == "place":
        if s["obj"].name not in _manually_attached_objects:
            print(f"⚠️  At place: object {s['obj'].name} not in gripper", flush=True)
            return s, True, False, "place failed: object not in gripper (dropped?)"
        place_pos = calculate_place_position(
            s["target_obj"], place_obj=s["obj"], surface_clearance=0.005, target_name=s["target_name"],
        )
        _, obj_orn = s["obj"].get_position_orientation()
        if th.is_tensor(obj_orn):
            obj_orn = obj_orn.cpu().numpy()
        obj_orn = np.array(obj_orn, dtype=np.float64)
        s["obj"].set_position_orientation(position=place_pos, orientation=obj_orn)
        detach_object_from_robot(s["obj"], robot, reason="place")
        for _ in range(25):
            og.sim.step()
        for _ in range(10):
            og.sim.step()
        return s, True, True, None

    return s, False, None, None


def pick_and_place(
    env,
    robot,
    scene,
    floor,
    object_name,
    target_name,
    visualizer=None,
    operation_radius_pick=0.8,
    operation_radius_place=0.5,
    doors=None,
    door_positions=None,
    door_states=None,
    door_map_regions=None,
    door_joint_limits=None,
    skip_doors=None,
    original_map_dict=None,
):
    """
    One-shot pick&place. doors/original_map_dict are for interactive door use; no auto-open on nav failure.

    Returns:
        bool: success
    """
    print("\n" + "="*80)
    print(f"🤖 Pick&Place start")
    print("="*80)
    print(f"object: {object_name}")
    print(f"target: {target_name}")
    print("="*80)
    if visualizer is not None:
        visualizer.clear_debug_goals()
    
    print(f"\n📦 Step 1: resolve objects...")
    obj = get_object_by_name(scene, object_name)
    if obj is None:
        print(f"❌ Object not found or not manipulable: {object_name}")
        return False
    if not is_spawned_graspable_object(obj):
        print(f"❌ Only spawned graspables (name contains _spawn). Use list_objects for the list.")
        return False

    target_obj = get_object_by_name(scene, target_name)
    if target_obj is None:
        print(f"❌ Target not found or not manipulable: {target_name}")
        return False
    
    if not is_manipulable_object(target_obj):
        print(f"❌ Target {target_name} is structural / not a valid placement parent")
        return False
    
    print(f"✅ object: {obj.name} (category: {obj.category})")
    print(f"✅ target: {target_obj.name} (category: {target_obj.category})")
    
    print(f"\n🚶 Step 2: navigate near object (try candidates)...")
    robot_pos, robot_orn = robot.get_position_orientation()
    if th.is_tensor(robot_pos):
        robot_pos = robot_pos.cpu().numpy()
    
    robot_pos_2d = robot_pos[:2]
    distance_threshold = 0.1
    candidates = list(iter_reachable_candidates_near_object(
        obj, robot_pos, scene, floor, operation_radius_pick, target_name=None, max_candidates=16
    ))
    if not candidates:
        print(f"❌ No reachable approach points near object")
        print("="*80)
        return False

    success = False
    approach_pos = None
    for attempt, candidate_pos in enumerate(candidates):
        approach_pos = candidate_pos
        approach_pos_2d = approach_pos[:2]
        distance_to_approach = np.linalg.norm(robot_pos_2d - approach_pos_2d)
        if distance_to_approach <= distance_threshold:
            success = True
            print(f"✅ Already near object")
            break
        print(f"   try approach {attempt+1}/{len(candidates)}: [{approach_pos[0]:.2f}, {approach_pos[1]:.2f}, {approach_pos[2]:.2f}]")
        success = navigate_to_goal_with_object_update(env, robot, scene, floor, approach_pos, visualizer)
        if success:
            break

    if not success:
        print(f"❌ Tried {len(candidates)} approaches; cannot reach near object (see map markers)")
        print("="*80)
        return False
    
    for _ in range(10):
        og.sim.step()
        update_manually_attached_objects()
    
    # Save pre-grasp pose to restore on cancel
    pick_pos, pick_orn = obj.get_position_orientation()
    pick_pos = np.array(pick_pos.cpu().numpy() if th.is_tensor(pick_pos) else pick_pos, dtype=np.float64).copy()
    pick_orn = np.array(pick_orn.cpu().numpy() if th.is_tensor(pick_orn) else pick_orn, dtype=np.float64).copy()
    
    print(f"\n🔗 Step 3: attach to robot...")
    if not attach_object_to_robot(obj, robot):
        print(f"❌ Attach failed")
        return False
    
    for _ in range(10):
        og.sim.step()
        update_manually_attached_objects()
    
    print(f"✅ Object attached to robot")
    
    print(f"\n🚶 Step 4: navigate near target...")
    robot_pos, _ = robot.get_position_orientation()
    if th.is_tensor(robot_pos):
        robot_pos = robot_pos.cpu().numpy()
    
    approach_pos = calculate_reachable_position_near_object(
        target_obj, robot_pos, scene, floor, operation_radius_place, target_name=target_name
    )
    
    robot_pos_2d = robot_pos[:2]
    approach_pos_2d = approach_pos[:2]
    distance_to_approach = np.linalg.norm(robot_pos_2d - approach_pos_2d)
    
    nav_to_target_ok = True
    if distance_to_approach > 0.1:
        print(f"   approach: [{approach_pos[0]:.2f}, {approach_pos[1]:.2f}, {approach_pos[2]:.2f}]")
        print(f"   place radius: {operation_radius_place}m")
        
        nav_to_target_ok = navigate_to_goal_with_object_update(env, robot, scene, floor, approach_pos, visualizer)
        if not nav_to_target_ok:
            print(f"❌ Failed to reach near target {target_name}, aborting place")
            detach_object_from_robot(obj, robot, reason="failure")
            obj.set_position_orientation(position=pick_pos, orientation=pick_orn)
            print(f"   object restored to pre-grasp pose")
            print("="*80)
            return False
    else:
        print(f"✅ Already near target")
    
    for _ in range(10):
        og.sim.step()
        update_manually_attached_objects()
    
    print(f"\n📦 Step 5: compute place pose and teleport...")
    place_pos = calculate_place_position(
        target_obj, place_obj=obj, surface_clearance=0.005, target_name=target_name
    )
    print(f"   place: [{place_pos[0]:.2f}, {place_pos[1]:.2f}, {place_pos[2]:.2f}]")
    
    obj_pos, obj_orn = obj.get_position_orientation()
    if th.is_tensor(obj_orn):
        obj_orn = obj_orn.cpu().numpy()
    obj_orn = np.array(obj_orn, dtype=np.float64)
    
    inside = target_name in PLACEMENT_INSIDE_TARGET_NAMES
    obj.set_position_orientation(position=place_pos, orientation=obj_orn)
    print(f"✅ Placed on target ({'inside' if inside else 'on surface'})")
    
    detach_object_from_robot(obj, robot, reason="place")
    for _ in range(25):
        og.sim.step()
    
    print(f"\n🔓 Step 6: final detach (if still linked)...")
    detach_object_from_robot(obj, robot, reason="default")
    
    for _ in range(10):
        og.sim.step()
    
    print(f"\n✅ Pick&Place done.")
    print("="*80)
    
    return True


def main():
    """Interactive CLI: nav, doors, pick&place."""
    CAMERA_WIDTH = 512
    CAMERA_HEIGHT = 512
    
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Beechwood_0_int",
            "load_object_categories": None,  # all categories (includes doors)
            "trav_map_resolution": 0.05,
            "default_erosion_radius": 0.3,
        },
        "robots": [
            {
                "type": "Turtlebot",
                "obs_modalities": ["scan", "rgb"],
                "action_type": "continuous",
                "action_normalize": True,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_width": CAMERA_WIDTH,
                            "image_height": CAMERA_HEIGHT,
                        }
                    }
                }
            }
        ],
    }
    
    print("Creating environment...")
    env = og.Environment(configs=config)
    robot = env.robots[0]
    scene = env.scene
    
    og.sim.enable_viewer_camera_teleoperation()
    
    env.reset()
    print(f"Robot initial pos: {robot.get_position_orientation()[0]}")

    if SPAWN_SMALL_OBJECTS_COUNT > 0:
        try:
            from spawn_small_objects import spawn_objects_one_per_category
            spawned = spawn_objects_one_per_category(
                scene,
                categories=SPAWN_OBJECT_CATEGORIES,
                surface_names=ALLOWED_PLACEMENT_TARGET_NAMES,
            )
            if spawned:
                print(f"✅ Spawned {len(spawned)} graspable objects: {spawned}")
                print("   pick <name_above> <placement_target_name>")
                for _ in range(30):
                    og.sim.step()
            else:
                print("⚠️  No small objects spawned (no table or category?)")
        except Exception as e:
            print(f"⚠️  spawn_small_objects failed: {e}")

    floor = 0
    
    original_map_dict = {}
    original_map_dict[floor] = scene.trav_map.floor_map[floor].clone()
    
    erosion_radius_meters = 0.4
    radius_pixel = int(math.ceil(erosion_radius_meters / scene.trav_map.map_resolution))
    kernel = np.ones((radius_pixel, radius_pixel), dtype=np.uint8)
    trav_map_eroded = cv2.erode(original_map_dict[floor].cpu().numpy(), kernel)
    scene.trav_map.floor_map[floor] = th.tensor(trav_map_eroded)
    
    print(f"✅ Map erosion: {erosion_radius_meters}m = {radius_pixel} px")
    
    doors = list_all_doors(scene)
    print_doors(doors)
    
    door_joint_limits = {}
    door_states = {}
    door_map_regions = {}
    door_shrink_factors = {
        3: 1.2,
        4: 1.2,
        8: 0.5,
    }
    
    print("\n📍 Initializing door map regions...")
    for idx, door in enumerate(doors, 1):
        shrink_factor = door_shrink_factors.get(idx, 1.0)
        door_region = get_door_map_region(door, scene.trav_map, floor, shrink_factor=shrink_factor)
        if door_region is not None:
            door_map_regions[door.name] = door_region
            shrink_info = f" (shrink {shrink_factor})" if shrink_factor < 1.0 else ""
            print(f"  ✅ door #{idx} ({door.name}): region [{door_region[0]}:{door_region[1]}, {door_region[2]}:{door_region[3]}]{shrink_info}")
        else:
            print(f"  ⚠️  door #{idx} ({door.name}): no map region")
    
    skip_doors = {7}
    
    door_positions = {
        1: {"open": -45, "close": 0},
        2: {"open": 100, "close": 0},
        3: {"open": 90, "close": 0},
        4: {"open": 70, "close": 0},
        5: {"open": 90, "close": 0},
        6: {"open": 135, "close": 0},
        8: {"open": 175, "close": 0},
        9: {"open": 175, "close": 0},
    }
    
    print(f"\n📐 Door joint positions (deg):")
    for door_idx, positions in door_positions.items():
        print(f"  door #{door_idx}: close={positions['close']}° open={positions['open']}°")
    print(f"  other doors: default")
    
    if skip_doors:
        print(f"\n⚠️  Skipped doors: {', '.join(f'#{d}' for d in skip_doors)}")
    
    print("\nCreating map visualizer...")
    visualizer = MapVisualizer(
        trav_map_obj=scene.trav_map,
        floor=floor,
        doors=doors,
        door_states=door_states
    )
    print("✅ Map viz (red=closed door, green=open door)")
    
    print("\n" + "="*80)
    print("🚀 Nav + doors + Pick&Place")
    print("="*80)
    print("Commands:")
    print("  'nav' — click map for goal")
    print("  'door <id> <open/close>' — one door")
    print("  'all open' / 'all close' — all doors")
    print("  'list' — doors")
    print("  'list_objects' — objects")
    print("  'pick <object> <target>' — pick&place")
    print("  'quit' / 'exit'")
    print("  WASD — move camera")
    print("="*80)
    
    print("\n⌨️  Type a command, Enter...")
    command_buffer = ""
    
    try:
        with NonBlockingInput() as nbi:
            while True:
                og.sim.step()
                # After physics step, update attached objects (avoid gravity pulling them)
                update_manually_attached_objects()
                
                char = nbi.get_data()
                
                if char:
                    if char == '\n' or char == '\r':
                        if command_buffer:
                            user_input = command_buffer.strip().lower()
                            command_buffer = ""
                            print()
                            
                            if user_input in ['quit', 'exit', 'q']:
                                print("👋 Exiting...")
                                break
                            
                            if user_input == 'list':
                                print_doors(doors)
                                print("\n⌨️  Type a command, Enter...")
                                continue
                            
                            if user_input == 'list_objects':
                                list_all_objects(scene)
                                print("\n⌨️  Type a command, Enter...")
                                continue
                            
                            if user_input == 'nav':
                                clicked_world_pos = visualizer.wait_for_click()
                                if clicked_world_pos is not None:
                                    floor_height = scene.get_floor_height(floor)
                                    target_pos = np.array([clicked_world_pos[0], clicked_world_pos[1], floor_height])
                                    print(f"✅ Goal: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
                                    navigate_to_goal_with_object_update(env, robot, scene, floor, target_pos, visualizer)
                                print("\n⌨️  Type a command, Enter...")
                                continue
                            
                            if user_input.startswith('pick'):
                                parts = user_input.split()
                                if len(parts) != 3:
                                    print("❌ Invalid. Use: pick <object_name> <target_name>")
                                    print("\n⌨️  Type a command, Enter...")
                                    continue
                                
                                object_name = parts[1]
                                target_name = parts[2]
                                
                                pick_and_place(env, robot, scene, floor, object_name, target_name, visualizer)
                                print("\n⌨️  Type a command, Enter...")
                                continue
                            
                            if user_input.startswith('all'):
                                parts = user_input.split()
                                if len(parts) != 2:
                                    print("❌ Invalid. Use: all <open/close>")
                                    print("\n⌨️  Type a command, Enter...")
                                    continue
                                
                                action = parts[1]
                                if action not in ['open', 'close']:
                                    print("❌ Action must be 'open' or 'close'")
                                    print("\n⌨️  Type a command, Enter...")
                                    continue
                                
                                action_str = "Opening" if action == "open" else "Closing"
                                print(f"\n🚪 {action_str} all doors...")
                                success_count = 0
                                skip_count = 0
                                
                                for idx, door in enumerate(doors, 1):
                                    if idx in skip_doors:
                                        print(f"  ⏭️  skip door #{idx} ({door.name})")
                                        skip_count += 1
                                        continue
                                    
                                    print(f"\nDoor #{idx} ({door.name})...")
                                    if control_door(door, idx, action, scene, floor, visualizer, original_map_dict, door_joint_limits, door_positions, door_states, doors=doors, door_map_regions=door_map_regions):
                                        success_count += 1
                                
                                verb = "opened" if action == "open" else "closed"
                                print(f"\n✅ Done: {verb} {success_count}/{len(doors)-skip_count} doors")
                                if skip_count > 0:
                                    print(f"⏭️  skipped {skip_count} door(s)")
                                
                                print("\n⌨️  Type a command, Enter...")
                                continue
                            
                            if user_input.startswith('door'):
                                parts = user_input.split()
                                if len(parts) != 3:
                                    print("❌ Invalid. Use: door <index> <open/close>")
                                    print("\n⌨️  Type a command, Enter...")
                                    continue
                                
                                try:
                                    door_idx = int(parts[1]) - 1
                                    action = parts[2]
                                    
                                    if door_idx < 0 or door_idx >= len(doors):
                                        print(f"❌ Door index must be 1..{len(doors)}")
                                    elif action not in ['open', 'close']:
                                        print("❌ Action must be 'open' or 'close'")
                                    elif (door_idx + 1) in skip_doors:
                                        print(f"⚠️  door #{door_idx + 1} ({doors[door_idx].name}) skipped (known issue)")
                                    else:
                                        control_door(doors[door_idx], door_idx + 1, action, scene, floor, visualizer, original_map_dict, door_joint_limits, door_positions, door_states, doors=doors, door_map_regions=door_map_regions)
                                except ValueError:
                                    print("❌ Door index must be a number")
                                
                                print("\n⌨️  Type a command, Enter...")
                                continue
                            
                            print("❌ Unknown command. Use 'nav', 'door', 'all', 'list', 'list_objects', 'pick', or 'quit'")
                            print("\n⌨️  Type a command, Enter...")
                        else:
                            print()
                    
                    elif char == '\x7f' or char == '\x08':
                        if command_buffer:
                            command_buffer = command_buffer[:-1]
                            sys.stdout.write('\b \b')
                            sys.stdout.flush()
                    
                    elif char == '\x03':
                        raise KeyboardInterrupt
                    
                    else:
                        command_buffer += char
                        sys.stdout.write(char)
                        sys.stdout.flush()
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted, exiting...")
    
    print("\nClosing map visualizer...")
    visualizer.close()
    
    print("Shutting down simulator...")
    og.shutdown()
    print("✅ Done.")


if __name__ == "__main__":
    main()

