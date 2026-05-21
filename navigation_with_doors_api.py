"""
Navigation + HTTP API (all doors open at startup).

Same logic as navigation_with_doors, plus an HTTP API for remote nav, camera, and Pick&Place. Doors are opened once at startup; no runtime door control or nav obstacle spawning.
This file runs standalone; navigation_with_doors.py is not modified.
"""
import bootstrap_paths  # noqa: F401  # must run before omnigibson

import base64
import os
import sys
import math
import time
import numpy as np
import cv2
import torch as th
import omnigibson as og

from omnigibson.object_states import Open

from navigation_with_doors import (
    list_all_doors,
    print_doors,
    get_door_map_region,
    control_door,
    ensure_all_listed_doors_visible_and_collidable,
    navigate_to_goal,
    init_navigation_state,
    navigate_step,
    nav_step,
    install_global_anti_topple,
    MapVisualizer,
    NonBlockingInput,
    _normalize_path_waypoints_list,
    _as_xy2,
    update_traversable_map,
    apply_doors_push_resistance,
    hide_scene_ceilings,
)
from navigation_with_pick_place import (
    update_manually_attached_objects,
    pick_and_place,
    init_pick_place_state,
    pick_place_step,
    release_abandoned_pick_place,
    get_objects_list_for_api,
    get_grasp_status_for_api,
    list_all_objects,
    SPAWN_SMALL_OBJECTS_COUNT,
    ALLOWED_PLACEMENT_TARGET_NAMES,
    SPAWN_OBJECT_CATEGORIES,
    PICK_PLACE_NAV_STEPS_PER_CALL,
)
import behavior_env_api as env_api

# Plan hooks (provided by env_api for the main loop)
_plan_set_and_start = getattr(env_api, "plan_set_and_start", None)
_plan_get_next_task = getattr(env_api, "plan_get_next_task", None)
_plan_after_task = getattr(env_api, "plan_after_task", None)
_plan_apply_update = getattr(env_api, "plan_apply_update", None)
_plan_cancel = getattr(env_api, "plan_cancel", None)
_plan_get_state = getattr(env_api, "plan_get_state", None)
_set_single_nav_navigating = getattr(env_api, "set_single_nav_navigating", None)
_set_single_nav_result = getattr(env_api, "set_single_nav_result", None)
_set_single_pick_place_running = getattr(env_api, "set_single_pick_place_running", None)
_set_single_pick_place_result = getattr(env_api, "set_single_pick_place_result", None)
_report_assistant_message = getattr(env_api, "report_assistant_message", None)

# After interrupting nav/Pick&Place, send several zero-velocity steps to reduce residual base motion ("drift" on next segment)
NAV_BRAKE_STEPS_ON_INTERRUPT = int(os.environ.get("NAV_BRAKE_STEPS_ON_INTERRUPT", "12"))


def _robot_xy_2d(robot):
    p = robot.get_position_orientation()[0]
    if th.is_tensor(p):
        p = p.cpu().numpy()
    p = np.asarray(p, dtype=np.float64)
    return p[:2]


def _door_xy_2d(door):
    pos, _ = door.get_position_orientation()
    if th.is_tensor(pos):
        pos = pos.cpu().numpy()
    pos = np.asarray(pos, dtype=np.float64)
    return pos[:2]


def _distance_robot_to_door(robot, door):
    return float(np.linalg.norm(_robot_xy_2d(robot) - _door_xy_2d(door)))


def _object_xy_2d(ob):
    """Object world XY; None on failure."""
    try:
        pos, _ = ob.get_position_orientation()
        if th.is_tensor(pos):
            pos = pos.cpu().numpy()
        pos = np.asarray(pos, dtype=np.float64)
        return pos[:2].copy()
    except Exception:
        return None


def _door_world_xy_list(doors):
    """World XY per door; id matches list_doors (1-based)."""
    out = []
    for i, door in enumerate(doors, 1):
        xy = _door_xy_2d(door)
        out.append({"id": i, "x": float(xy[0]), "y": float(xy[1])})
    return out


def _sync_door_states_from_scene(doors, door_states: dict) -> None:
    """Fill door_states from sim Open state (for rebuild of nav map after reset)."""
    door_states.clear()
    for door in doors:
        nm = getattr(door, "name", None) or "unknown"
        if hasattr(door, "states") and Open in door.states:
            door_states[nm] = "open" if door.states[Open].get_value() else "close"
        else:
            door_states[nm] = "close"


def _clear_navigation_overlay(robot, visualizer):
    """Clear path/goal overlay after nav step to avoid mixing with later Pick&Place."""
    if visualizer is None:
        return
    rp = robot.get_position_orientation()[0][:2]
    if th.is_tensor(rp):
        rp = rp.cpu().numpy()
    rp = np.asarray(rp, dtype=np.float64)
    visualizer.update(robot_pos=rp, goal_pos=None, path=None, current_waypoint=None)


def _stabilize_robot_after_nav_interrupt(env, robot, visualizer, after_step_callback=None):
    """Brake after interrupt so previous velocity does not carry into the next path."""
    n = max(0, NAV_BRAKE_STEPS_ON_INTERRUPT)
    for _ in range(n):
        nav_step(env, robot, [0.0, 0.0], after_step_callback)
    _clear_navigation_overlay(robot, visualizer)


# Max long edge of camera image for the model (avoid huge context, ~128k token risk)
CAMERA_IMAGE_MAX_SIZE = 384
CAMERA_JPEG_QUALITY = 78


def setup_stretch_for_navigation(robot):
    """
    Configure Stretch for navigation. Base uses kinematic control (set pose), not wheel physics,
    so no wheel slip. Re-enable gravity for a normal look (no floating), and disable finger collision
    to reduce tipping when passing through doorways.
    """
    if robot.__class__.__name__ != "Stretch":
        return
    for link_name, link in robot.links.items():
        if hasattr(link, "enable_gravity"):
            link.enable_gravity()

    _FINGER_LINK_NAMES = {"link_gripper_finger_left", "link_gripper_finger_right"}
    disabled_n = 0
    for link_name, link in robot.links.items():
        if link_name in _FINGER_LINK_NAMES:
            try:
                link.disable_collisions()
                try:
                    link.visible = False
                except Exception:
                    pass
                disabled_n += 1
            except Exception as e:
                print(f"⚠️  disable collision failed for {link_name}: {e}")
    if disabled_n:
        print(f"✅ Disabled/hidden {disabled_n} gripper finger link colliders (fewer door-scrape tips)")
    print("✅ Stretch nav: kinematic base (no slip), gravity re-enabled")


def _head_camera_path_score(name_lower: str) -> int:
    """Higher = more like head RGB (OG sensor path names); wrist/arm/misc negative."""
    s = 0
    if "link_head" in name_lower or "head_tilt" in name_lower or "head_pan" in name_lower:
        s += 200
    elif ":head" in name_lower or "joint_head" in name_lower:
        s += 120
    elif "head" in name_lower:
        s += 60
    if any(x in name_lower for x in ("wrist", "gripper", "eef", "finger", "link_gripper")):
        s -= 320
    # Lift/arm cams often point at sky; avoid using them as head if no link_head
    if any(x in name_lower for x in ("link_lift", "link_arm_l0", "link_arm_l1", "link_arm_l2", "link_arm_l3")):
        s -= 100
    if any(x in name_lower for x in ("external", "viewer", "occupancy", "trav", "orthographic")):
        s -= 500
    return s


def _gripper_camera_path_score(name_lower: str) -> int:
    s = 0
    if any(x in name_lower for x in ("wrist", "gripper", "eef", "finger", "link_gripper")):
        s += 200
    if "head" in name_lower and "link_head" not in name_lower:
        s -= 50
    return s


def _select_rgb_tensor_for_view(candidates, view: str):
    """
    Pick one RGB stream for head vs gripper. Taking the first dict's rgb can pick lift-arm cams
    and yield sky views for nav-stuck diagnosis; score by sensor path name instead.
    """
    view = (view or "head").strip().lower()
    if view not in ("head", "gripper"):
        view = "head"

    def headish(low: str) -> bool:
        return "head" in low

    def gripperish(low: str) -> bool:
        return any(x in low for x in ("gripper", "wrist", "eef", "finger"))

    lows = [n.lower() for n, _ in candidates]

    if view == "head":
        scored = sorted(
            zip(candidates, lows),
            key=lambda pair: _head_camera_path_score(pair[1]),
            reverse=True,
        )
        best_s = _head_camera_path_score(scored[0][1]) if scored else -9999
        if best_s >= 60:
            name, rgb = scored[0][0]
            return name, rgb
        for (name, rgb), low in zip(candidates, lows):
            if headish(low) and _head_camera_path_score(low) >= 0:
                return name, rgb
        for (name, rgb), low in zip(candidates, lows):
            if headish(low):
                return name, rgb
        for (name, rgb), low in zip(candidates, lows):
            if not gripperish(low) and _head_camera_path_score(low) >= -50:
                return name, rgb
        for (name, rgb), low in zip(candidates, lows):
            if not gripperish(low):
                return name, rgb
        return candidates[0]

    scored_g = sorted(
        zip(candidates, lows),
        key=lambda pair: _gripper_camera_path_score(pair[1]),
        reverse=True,
    )
    if scored_g and _gripper_camera_path_score(scored_g[0][1]) >= 100:
        return scored_g[0][0]
    for (name, rgb), low in zip(candidates, lows):
        if gripperish(low) and not headish(low):
            return name, rgb
    for (name, rgb), low in zip(candidates, lows):
        if gripperish(low):
            return name, rgb
    return candidates[0]


def _trav_map_world_center_xy(scene, floor: int, robot) -> np.ndarray:
    """Centroid XY of free grid (meters); if empty, use robot XY."""
    fm = scene.trav_map.floor_map[floor]
    if th.is_tensor(fm):
        fm = fm.cpu().numpy()
    rows, cols = np.where(fm > 0)
    if len(rows) == 0:
        p = robot.get_position_orientation()[0]
        if th.is_tensor(p):
            p = p.cpu().numpy()
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        return np.array([float(p[0]), float(p[1])], dtype=np.float64)
    mr = float(rows.mean())
    mc = float(cols.mean())
    w = scene.trav_map.map_to_world(th.tensor([mr, mc], dtype=th.float32))
    if th.is_tensor(w):
        w = w.cpu().numpy()
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return w[:2].copy()


def setup_navigation_view_layout(robot, scene, floor: int) -> None:
    """
    Default: hide ceilings; main viewport top-down on scene center.
    Viewport 1/2 come from VisionSensor (head + wrist). NAV_DISABLE_VIEW_LAYOUT=1 skips all;
    NAV_SHOW_CEILINGS=1 keeps ceilings visible.
    """
    from omnigibson.macros import gm

    if gm.HEADLESS:
        return
    if os.environ.get("NAV_DISABLE_VIEW_LAYOUT", "").strip().lower() in ("1", "true", "yes"):
        return

    nc = hide_scene_ceilings(scene)
    if nc:
        print(f"✅ Hidden {nc} ceiling-related prims (NAV_SHOW_CEILINGS=1 to keep ceilings)")

    center = _trav_map_world_center_xy(scene, floor, robot)
    z0 = float(scene.get_floor_height(floor))
    dz = float(os.environ.get("NAV_TOPDOWN_CAM_HEIGHT_M", "14.0"))
    pos = th.tensor([center[0], center[1], z0 + dz], dtype=th.float32)
    quat_topdown = th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)

    try:
        vc = og.sim.viewer_camera
        try:
            vc.active_camera_path = vc.prim_path
        except Exception:
            pass
        try:
            vp = vc._viewport
            win_w, win_h = vp.viewport_api.get_texture_resolution()
            if win_h > 0 and win_w > 0:
                vp.viewport_api.set_texture_resolution((win_w, win_w))
                for _ in range(2):
                    og.sim.render()
        except Exception:
            pass
        vc.set_position_orientation(position=pos, orientation=quat_topdown)
        for _ in range(4):
            og.sim.render()
        print(
            f"✅ Main viewport: top-down (h={dz}m); left Viewport 1/2 = head/wrist"
            " (NAV_DISABLE_VIEW_LAYOUT=1 off; NAV_SHOW_CEILINGS=1 keeps ceilings)"
        )
    except Exception as ex:
        print(f"⚠️  main viewport top-down failed: {ex}")


def refresh_physics_handles_after_hard_reset() -> None:
    """
    After hard reset (scene.restore) PhysX may still reference removed colliders (Illegal BroadPhaseUpdateData);
    main viewport may be broken. Refresh handles and render a few frames.
    """
    try:
        og.sim.update_handles()
    except Exception as ex:
        print(f"⚠️  update_handles after reset: {ex}", flush=True)
    for _ in range(4):
        try:
            og.sim.render()
        except Exception:
            break


def get_robot_camera_base64(robot, view: str = "head"):
    """RGB for head|gripper, resized, JPEG base64.

    Returns (success, data, meta): on success data is base64, meta has sensor_name;
    on failure data is an error string, meta is {}.
    """
    try:
        obs_dict, _ = robot.get_obs()
        candidates = []
        for name, sensor_obs in obs_dict.items():
            if name == "proprio":
                continue
            if not isinstance(sensor_obs, dict):
                continue
            rgb = sensor_obs.get("rgb")
            if rgb is None:
                continue
            candidates.append((str(name), rgb))
        if not candidates:
            return False, "No robot camera RGB in observations", {}

        picked_name, rgb = _select_rgb_tensor_for_view(candidates, view)
        if th.is_tensor(rgb):
            rgb = rgb.cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return False, "Invalid RGB tensor shape", {}
        rgb = rgb[:, :, :3].astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = rgb.shape[:2]
        if max(h, w) > CAMERA_IMAGE_MAX_SIZE:
            scale = CAMERA_IMAGE_MAX_SIZE / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
        b64 = base64.b64encode(buf).decode("ascii")
        if not b64 or not b64.strip():
            return False, "Empty JPEG after encode", {}
        return True, b64, {"sensor_name": picked_name}
    except Exception as e:
        return False, str(e), {}


def main():
    """Entry: same as non-API main, plus HTTP API and remote commands in the loop."""
    CAMERA_WIDTH = 512
    CAMERA_HEIGHT = 512

    # Default Stretch pose: wrist_yaw < 90° so gripper biases toward body midline (fewer door hits)
    # Order: base(2), camera(2), arm(8)=[lift, arm_l3..l0, wrist_yaw, wrist_pitch, wrist_roll], gripper(2)
    STRETCH_RESET_JOINT_POS = [
        0.0, 0.0,           # base
        0.5, 0.0,           # camera: head_pan, head_tilt
        0.0, 0.0, 0.0, 0.0, 0.0,  # lift, arm_l3..l0
        math.pi / 2.0 + 0.50, 0.0, 0.0,  # wrist_yaw ~29° inward
        math.pi / 8.0, math.pi / 8.0,  # gripper open
    ]

    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Beechwood_0_int",
            "load_object_categories": None,
            "trav_map_resolution": 0.05,
            "default_erosion_radius": 0.3,
        },
        "robots": [
            {
                "type": "Stretch",
                "obs_modalities": ["scan", "rgb"],
                "action_type": "continuous",
                "action_normalize": True,
                "reset_joint_pos": STRETCH_RESET_JOINT_POS,
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_width": CAMERA_WIDTH,
                            "image_height": CAMERA_HEIGHT,
                        }
                    }
                },
            }
        ],
    }

    print("Creating environment...")
    env = og.Environment(configs=config)
    robot = env.robots[0]
    scene = env.scene
    og.sim.enable_viewer_camera_teleoperation()
    env.reset()
    print(f"Robot initial position: {robot.get_position_orientation()[0]}")
    setup_stretch_for_navigation(robot)

    install_global_anti_topple(robot)

    # Spawn small graspable objects on placement surfaces (same as navigation_with_pick_place)
    if SPAWN_SMALL_OBJECTS_COUNT > 0:
        try:
            from spawn_small_objects import spawn_objects_one_per_category
            spawned = spawn_objects_one_per_category(
                scene,
                categories=SPAWN_OBJECT_CATEGORIES,
                surface_names=ALLOWED_PLACEMENT_TARGET_NAMES,
            )
            if spawned:
                print(f"✅ Spawned {len(spawned)} small graspable objects: {spawned}")
                for _ in range(30):
                    og.sim.step()
            else:
                print("⚠️  No small objects spawned (no table or category?)")
        except Exception as e:
            print(f"⚠️  small object spawn failed: {e}")

    floor = 0
    original_map_dict = {}
    original_map_dict[floor] = scene.trav_map.floor_map[floor].clone()

    erosion_radius_meters = 0.4
    radius_pixel = int(math.ceil(erosion_radius_meters / scene.trav_map.map_resolution))
    kernel = np.ones((radius_pixel, radius_pixel), dtype=np.uint8)
    trav_map_eroded = cv2.erode(original_map_dict[floor].cpu().numpy(), kernel)
    scene.trav_map.floor_map[floor] = th.tensor(trav_map_eroded)
    print(f"✅ Map erosion done: {erosion_radius_meters}m = {radius_pixel} px")

    doors = list_all_doors(scene)
    print_doors(doors)

    door_joint_limits = {}
    door_states = {}
    door_map_regions = {}
    door_shrink_factors = {3: 1.2, 4: 1.2, 8: 0.5}

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

    print(f"\n📐 Door open/close joint angles (deg):")
    for door_idx, positions in door_positions.items():
        print(f"  door #{door_idx}: close={positions['close']}°, open={positions['open']}°")
    if skip_doors:
        print(f"\n⚠️  Skipped doors (known issues): {', '.join(f'#{d}' for d in skip_doors)}")

    _sync_door_states_from_scene(doors, door_states)
    apply_doors_push_resistance(doors)
    update_traversable_map(
        scene,
        floor,
        original_map_dict[floor],
        doors=doors,
        door_states=door_states,
        door_map_regions=door_map_regions,
    )

    print("\nCreating map visualizer...")
    visualizer = MapVisualizer(
        trav_map_obj=scene.trav_map,
        floor=floor,
        doors=doors,
        door_states=door_states
    )
    print("✅ Map visualizer (red/green=doors)")

    print("\n🚪 Opening all controllable doors at startup (no runtime door control)...")
    doors_opened = 0
    for idx, door in enumerate(doors, 1):
        if idx in skip_doors:
            continue
        try:
            if control_door(
                door,
                idx,
                "open",
                scene,
                floor,
                visualizer,
                original_map_dict,
                door_joint_limits,
                door_positions,
                door_states,
                doors=doors,
                door_map_regions=door_map_regions,
            ):
                doors_opened += 1
        except Exception as _open_ex:
            print(f"  ⚠️  open door #{idx} failed: {_open_ex}", flush=True)
    print(f"✅ Opened {doors_opened} door(s) at startup (remain open; door control API disabled)")

    # HTTP API for MCP door-robot-server
    api_port = int(os.getenv("BEHAVIOR_ENV_PORT", "5001"))
    graspable_names, placement_names = get_objects_list_for_api(scene)
    env_api.update_state(
        doors=[d.name for d in doors],
        door_states=door_states,
        objects_graspable=graspable_names,
        objects_placement=placement_names,
        door_world_xy=_door_world_xy_list(doors),
        skip_door_ids=sorted(skip_doors),
    )

    try:
        _api_ready_timeout = float(os.getenv("BEHAVIOR_API_READY_TIMEOUT_S", "30"))
    except ValueError:
        _api_ready_timeout = 30.0
    _api_ready_timeout = max(5.0, min(_api_ready_timeout, 300.0))

    api_ready = False
    try:
        env_api.run_server(host="0.0.0.0", port=api_port)
        print(f"✅ Behavior API starting: http://0.0.0.0:{api_port} ...")
        if env_api.wait_until_ready(api_port, timeout=_api_ready_timeout):
            print(f"✅ Behavior API ready: http://127.0.0.1:{api_port}")
            api_ready = True
        else:
            print(
                f"⚠️  port {api_port} not accepting connections within {_api_ready_timeout:.0f}s "
                f"(in use? check stderr). Set BEHAVIOR_API_READY_TIMEOUT_S if cold start is slow."
            )
    except Exception as e:
        print(f"⚠️  Behavior API start failed: {e}")

    if not api_ready:
        print(
            "\n[FATAL] HTTP API did not become ready — exiting so you are not misled by a later "
            "[BEHAVIOR_ENV_READY] line.\n"
            f"  Port: {api_port} (override with BEHAVIOR_ENV_PORT)\n"
            f"  Check: `ss -lntp | grep {api_port}` — if another PID listens, kill it or pick another port.\n"
            "  Stderr: look for `[Behavior Env API] server thread failed` (often need `pip install flask`).\n",
            flush=True,
        )
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🚀 Navigation + HTTP API (all doors open at startup)")
    print("=" * 80)
    print("📖 CLI:")
    print("  - Type 'nav' then click on the map to set a goal")
    print("  - All controllable doors are opened at startup; runtime door control is disabled")
    print("  - Trav map can reflect door apertures (NAV_MAP_REFLECT_DOOR_STATE=1)")
    print(
        "  - 3D view: default hide ceiling, main top-down; left Viewport 1/2 = head/wrist"
        " (NAV_DISABLE_VIEW_LAYOUT=1 off; NAV_SHOW_CEILINGS=1 keeps ceilings)"
    )
    print("  - 'list' — all doors")
    print("  - 'list_objects' — graspable and placement names")
    print("  - 'pick <object> <target>' — Pick&Place")
    print("  - 'quit' / 'exit'")
    print("  - MCP/API: robot nav, camera, Pick&Place (GET /api/doors is read-only)")
    print("  - API: GET /api/objects, POST /api/robot/pick_place")
    print("  - Plans: POST /api/plan/submit, GET /api/plan/status, POST /api/plan/update")
    print("=" * 80)

    print("\n⌨️  Type a command and Enter...")
    command_buffer = ""

    NAV_STEPS_PER_FRAME = 30  # max sim steps of nav per frame so we can handle new commands
    active_nav = None
    active_pick_place = None  # stepped pick_place state; door/nav can interleave

    _clear_navigation_overlay(robot, visualizer)
    setup_navigation_view_layout(robot, scene, floor)

    def execute_one_command(cmd_type, payload, plan_si=None, plan_ti=None):
        """One API primitive (camera only; doors are fixed open at startup)."""
        try:
            if cmd_type == "get_camera":
                v = str((payload or {}).get("view", "head") or "head").strip().lower()
                if v not in ("head", "gripper"):
                    v = "head"
                ok, data, cam_meta = get_robot_camera_base64(robot, view=v)
                if ok:
                    out = {"image_base64": data}
                    out.update(cam_meta or {})
                    return True, None, out
                return False, data, {}
            return False, "unknown command type", {}
        except Exception as e:
            return False, str(e), {}

    def start_nav(payload, nav_type, plan_si=None, plan_ti=None):
        """Init stepped nav and set active_nav."""
        nonlocal active_nav
        x, y = payload.get("x"), payload.get("y")
        fl = payload.get("floor", 0)
        goal_yaw_deg = payload.get("goal_yaw_deg")
        if x is None or y is None:
            if nav_type == "single" and _set_single_nav_result:
                _set_single_nav_result(False, "x, y required")
            elif nav_type == "plan" and plan_si is not None and plan_ti is not None:
                _plan_after_task(plan_si, plan_ti, False, error="x, y required")
            return False
        floor_height = scene.get_floor_height(fl)
        target_pos = np.array([float(x), float(y), floor_height], dtype=np.float64)
        goal_yaw_rad = math.radians(float(goal_yaw_deg)) if goal_yaw_deg is not None else None
        nav_state = init_navigation_state(
            env, robot, scene, floor, target_pos, goal_yaw_rad,
        )
        if nav_state is None:
            if nav_type == "single" and _set_single_nav_result:
                _set_single_nav_result(False, "no path found")
            elif nav_type == "plan" and plan_si is not None and plan_ti is not None:
                _plan_after_task(plan_si, plan_ti, False, error="no path found")
            return False
        if nav_type == "single" and _set_single_nav_navigating:
            _set_single_nav_navigating()
        active_nav = {
            "type": nav_type,
            "payload": payload,
            "nav_state": nav_state,
            "si": plan_si,
            "ti": plan_ti,
        }
        return True

    # Ready signal for external runners (e.g. run_eval_experiments.py): scene,
    # robot, traversable map, HTTP API and (optionally) all doors are now set up
    # and the main loop is about to start.
    print(
        f"[BEHAVIOR_ENV_READY] api=http://127.0.0.1:{api_port} doors={len(doors)}",
        flush=True,
    )

    try:
        with NonBlockingInput() as nbi:
            while True:
                if active_pick_place is not None:
                    plan_si_saved = active_pick_place.get("plan_si")
                    plan_ti_saved = active_pick_place.get("plan_ti")
                    new_state, done, success, err_msg = pick_place_step(
                        env, robot, scene, floor, active_pick_place, visualizer,
                        max_steps_per_call=PICK_PLACE_NAV_STEPS_PER_CALL,
                    )
                    active_pick_place = dict(new_state)
                    active_pick_place["plan_si"] = plan_si_saved
                    active_pick_place["plan_ti"] = plan_ti_saved
                    update_manually_attached_objects()
                    if not done and plan_si_saved is None:
                        env_api.set_single_pick_place_phase(active_pick_place.get("phase", ""))
                    if done:
                        plan_si = plan_si_saved
                        plan_ti = plan_ti_saved
                        if plan_si is not None and plan_ti is not None and _plan_after_task:
                            _plan_after_task(plan_si, plan_ti, success, error=err_msg)
                        else:
                            if _set_single_pick_place_result:
                                _set_single_pick_place_result(success, err_msg)
                        if _report_assistant_message:
                            if success:
                                _report_assistant_message("Pick&Place finished (success)")
                            else:
                                _report_assistant_message(f"Pick&Place finished (failed): {err_msg or 'unknown error'}")
                        active_pick_place = None
                elif active_nav is not None:
                    if active_nav["type"] == "plan" and _plan_get_state and _plan_get_state().get("status") != "running":
                        _clear_navigation_overlay(robot, visualizer)
                        active_nav = None
                    else:
                        status, new_state = navigate_step(
                            env, robot, scene, floor,
                            active_nav["nav_state"], visualizer,
                            max_steps_per_call=NAV_STEPS_PER_FRAME,
                            after_step_callback=update_manually_attached_objects,
                        )
                        active_nav["nav_state"] = new_state
                        update_manually_attached_objects()
                        if status in ("reached", "failed"):
                            nav_type = active_nav["type"]
                            si, ti = active_nav.get("si"), active_nav.get("ti")
                            ok = status == "reached"
                            err = None if ok else "Navigation failed or timed out"
                            if nav_type == "single" and _set_single_nav_result:
                                _set_single_nav_result(ok, err)
                            elif nav_type == "plan" and si is not None and ti is not None and _plan_after_task:
                                _plan_after_task(si, ti, ok, error=err)
                            if ok and _report_assistant_message:
                                _report_assistant_message("Robot has reached the goal")
                            _clear_navigation_overlay(robot, visualizer)
                            active_nav = None
                else:
                    og.sim.step()
                    update_manually_attached_objects()

                # API state
                robot_pos = robot.get_position_orientation()[0]
                if th.is_tensor(robot_pos):
                    robot_pos = robot_pos.cpu().numpy()
                else:
                    robot_pos = np.array(robot_pos)
                env_api.update_state(
                    robot_position=robot_pos[:2].tolist(),
                    door_states=door_states,
                    door_world_xy=_door_world_xy_list(doors),
                    skip_door_ids=sorted(skip_doors),
                )

                # Command handling: (1) cancel/camera always (2) plan meta (3) next task when idle

                def _cancel_all_active(reason="Task cancelled"):
                    nonlocal active_pick_place, active_nav
                    had_motion = active_pick_place is not None or active_nav is not None
                    stopped = False
                    if active_pick_place is not None:
                        release_abandoned_pick_place(active_pick_place, robot)
                        active_pick_place = None
                        stopped = True
                        if _set_single_pick_place_result:
                            _set_single_pick_place_result(False, reason)
                    if active_nav is not None:
                        active_nav = None
                        stopped = True
                        if _set_single_nav_result:
                            _set_single_nav_result(False, reason)
                    if had_motion:
                        _stabilize_robot_after_nav_interrupt(
                            env, robot, visualizer, update_manually_attached_objects
                        )
                    else:
                        _clear_navigation_overlay(robot, visualizer)
                    if _plan_cancel:
                        _plan_cancel()
                    return stopped

                def _start_pick_place(obj_name, tgt_name, plan_si=None, plan_ti=None):
                    nonlocal active_pick_place
                    obj_name = (obj_name or "").strip()
                    tgt_name = (tgt_name or "").strip()
                    if not obj_name or not tgt_name:
                        if plan_si is not None and _plan_after_task:
                            _plan_after_task(plan_si, plan_ti, False, error="object_name and target_name required")
                        elif _set_single_pick_place_result:
                            _set_single_pick_place_result(False, "object_name and target_name required")
                        return
                    pp_state = init_pick_place_state(
                        env, robot, scene, floor, obj_name, tgt_name, visualizer,
                        doors=doors, door_positions=door_positions, door_states=door_states,
                        door_map_regions=door_map_regions, door_joint_limits=door_joint_limits,
                        skip_doors=skip_doors, original_map_dict=original_map_dict,
                    )
                    if pp_state is None:
                        if plan_si is not None and _plan_after_task:
                            _plan_after_task(plan_si, plan_ti, False, error="Object not found or not manipulable")
                        elif _set_single_pick_place_result:
                            _set_single_pick_place_result(False, "Object not found or not manipulable")
                        return
                    pp_state["plan_si"] = plan_si
                    pp_state["plan_ti"] = plan_ti
                    if plan_si is None and _set_single_pick_place_running:
                        _set_single_pick_place_running(obj_name, tgt_name)
                    active_pick_place = pp_state

                # ①a Preempt: pick_place can take over (e.g. LOST retry)
                preempt_pp = env_api.pop_preempt_pick_place()
                if preempt_pp:
                    had_motion = active_pick_place is not None or active_nav is not None
                    old_pp = active_pick_place
                    if active_pick_place is not None:
                        active_pick_place = None
                    if old_pp is not None:
                        release_abandoned_pick_place(old_pp, robot)
                    if active_nav is not None:
                        active_nav = None
                        if _set_single_nav_result:
                            _set_single_nav_result(False, "Interrupted by new pick_place task")
                    if had_motion:
                        _stabilize_robot_after_nav_interrupt(
                            env, robot, visualizer, update_manually_attached_objects
                        )
                    else:
                        _clear_navigation_overlay(robot, visualizer)
                    if _plan_cancel:
                        _plan_cancel()
                    _start_pick_place(preempt_pp.get("object_name"), preempt_pp.get("target_name"))

                # ①b Always: cancel / camera / plan (even when tasks running)
                cmd = env_api.pop_command_if_type(
                    "cancel_current_task", "cancel_pick_place", "get_camera",
                    "plan_submit", "plan_update",
                )
                if cmd:
                    ctype, payload = cmd
                    if ctype == "cancel_current_task":
                        stopped = _cancel_all_active()
                        env_api.put_result(True, None, message=("Stopped current task" if stopped else "No task running"))
                    elif ctype == "cancel_pick_place":
                        if active_pick_place is not None:
                            release_abandoned_pick_place(active_pick_place, robot)
                            active_pick_place = None
                            if _set_single_pick_place_result:
                                _set_single_pick_place_result(False, "Pick&Place cancelled")
                            _stabilize_robot_after_nav_interrupt(
                                env, robot, visualizer, update_manually_attached_objects
                            )
                            env_api.put_result(True, None, message="Pick&Place cancelled")
                        else:
                            env_api.put_result(True, None, message="No pick_place in progress")
                    elif ctype == "get_camera":
                        ok, err, extra = execute_one_command(ctype, payload)
                        env_api.put_result(ok, err, **(extra or {}))
                    elif ctype == "plan_submit" and _plan_set_and_start:
                        _plan_set_and_start(payload)
                    elif ctype == "plan_update" and _plan_apply_update:
                        _plan_apply_update(payload)

                # ② When idle: next task from queue or plan
                if active_nav is None and active_pick_place is None:
                    cmd = env_api.get_next_command(timeout=0)
                    started_from_queue = False
                    if cmd:
                        ctype, payload = cmd
                        started_from_queue = True
                        if ctype == "navigate":
                            start_nav(payload, "single")
                        elif ctype == "pick_place":
                            _start_pick_place(payload.get("object_name"), payload.get("target_name"))
                        elif ctype == "get_camera":
                            ok, err, extra = execute_one_command(ctype, payload)
                            env_api.put_result(ok, err, **(extra or {}))
                        elif ctype == "cancel_current_task":
                            stopped = _cancel_all_active()
                            env_api.put_result(True, None, message=("Stopped current task" if stopped else "No task running"))
                        elif ctype == "cancel_pick_place":
                            env_api.put_result(True, None, message="No pick_place in progress")
                        elif ctype == "plan_submit" and _plan_set_and_start:
                            _plan_set_and_start(payload)
                        elif ctype == "plan_update" and _plan_apply_update:
                            _plan_apply_update(payload)

                    if not started_from_queue or (active_nav is None and active_pick_place is None):
                        if _plan_get_next_task and _plan_after_task and _plan_get_state:
                            st = _plan_get_state()
                            if st.get("status") == "running" and active_nav is None and active_pick_place is None:
                                task = _plan_get_next_task()
                                if task:
                                    si, ti, ctype, payload = task
                                    if ctype == "navigate":
                                        start_nav(payload, "plan", plan_si=si, plan_ti=ti)
                                    elif ctype == "pick_place":
                                        _start_pick_place(
                                            payload.get("object_name"), payload.get("target_name"),
                                            plan_si=si, plan_ti=ti,
                                        )
                                    elif ctype == "get_camera":
                                        ok, err, extra = execute_one_command(ctype, payload, plan_si=si, plan_ti=ti)
                                        if not (extra and extra.get("deferred")):
                                            ex = {k: v for k, v in (extra or {}).items() if k != "deferred"}
                                            _plan_after_task(si, ti, ok, error=err, **ex)
                                    elif ctype == "cancel_current_task":
                                        _cancel_all_active()
                                        _plan_after_task(si, ti, True, message="Current task stopped")

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
                                print("\n⌨️  Type a command and press Enter...")
                                continue

                            if user_input == 'list_objects':
                                list_all_objects(scene)
                                print("\n⌨️  Type a command and press Enter...")
                                continue

                            if user_input.startswith('pick'):
                                parts = user_input.split(maxsplit=2)
                                if len(parts) != 3:
                                    print("❌ Bad format. Use: pick <object> <target>")
                                else:
                                    pick_and_place(
                                        env, robot, scene, floor, parts[1], parts[2], visualizer,
                                        doors=doors, door_positions=door_positions, door_states=door_states,
                                        door_map_regions=door_map_regions, door_joint_limits=door_joint_limits,
                                        skip_doors=skip_doors, original_map_dict=original_map_dict,
                                    )
                                print("\n⌨️  Type a command and press Enter...")
                                continue

                            if user_input == 'nav':
                                result = visualizer.wait_for_goal_with_orientation()
                                if result is not None:
                                    clicked_world_pos, goal_yaw_rad = result
                                    floor_height = scene.get_floor_height(floor)
                                    target_pos = np.array([clicked_world_pos[0], clicked_world_pos[1], floor_height])
                                    print(f"✅ Goal: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]")
                                    navigate_to_goal(env, robot, scene, floor, target_pos, visualizer, goal_yaw_rad=goal_yaw_rad)
                                print("\n⌨️  Type a command and press Enter...")
                                continue

                            print(
                                "❌ Unknown command. Try: nav, list, list_objects, pick, quit"
                            )
                            print("\n⌨️  Type a command and press Enter...")
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

    print("\nClosing visualizer...")
    visualizer.close()
    print("Shutting down simulator...")
    og.shutdown()
    print("✅ Done")


if __name__ == "__main__":
    main()
