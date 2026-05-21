"""
Navigation with doors: A* pathing, door control, and traversable-map updates after door state changes.
"""
import bootstrap_paths  # noqa: F401  # must run before omnigibson

import os
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
from omnigibson.object_states import Open
from omnigibson.utils.transform_utils import quat2euler

# FlatCache for performance
gm.ENABLE_FLATCACHE = True

# Path tracking: pure pursuit toward the current vertex can cut corners through walls; lookahead + caps + heading decay stay closer to the polyline
_NAV_MAX_LIN_VEL = float(os.environ.get("NAV_MAX_LIN_VEL", "0.38"))
_NAV_MAX_ANG_VEL = float(os.environ.get("NAV_MAX_ANG_VEL", "1.4"))
_NAV_LOOKAHEAD_M = float(os.environ.get("NAV_LOOKAHEAD_M", "0.36"))
_NAV_TURN_IN_PLACE_RAD = float(os.environ.get("NAV_TURN_IN_PLACE_RAD", "0.45"))

# Anti-topple: after each nav substep, if the base roll/pitch exceeds a small threshold (e.g. after bumping a door),
# rewrite the orientation to yaw-only and zero root-link velocities so the robot cannot tip over.
_NAV_ANTI_TOPPLE = os.environ.get("NAV_ANTI_TOPPLE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
try:
    _NAV_ANTI_TOPPLE_TILT_RAD = float(os.environ.get("NAV_ANTI_TOPPLE_TILT_RAD", "0.18"))
except ValueError:
    _NAV_ANTI_TOPPLE_TILT_RAD = 0.18
_NAV_ANTI_TOPPLE_TILT_RAD = max(0.05, min(0.8, _NAV_ANTI_TOPPLE_TILT_RAD))
try:
    _NAV_IMPACT_BRAKE_STEPS = int(os.environ.get("NAV_IMPACT_BRAKE_STEPS", "2"))
except ValueError:
    _NAV_IMPACT_BRAKE_STEPS = 2
_NAV_IMPACT_BRAKE_STEPS = max(0, min(20, _NAV_IMPACT_BRAKE_STEPS))

try:
    from omnigibson.utils.motion_planning_utils import (
        detect_robot_collision_in_sim as _og_detect_robot_collision_in_sim,
    )
except Exception:
    _og_detect_robot_collision_in_sim = None


def _nav_robot_in_non_ground_contact(robot) -> bool:
    """ContactBodies (non-zero impulse, ignoring ground/self/in-hand): is the base hitting something solid?"""
    if _og_detect_robot_collision_in_sim is None:
        return False
    try:
        return bool(_og_detect_robot_collision_in_sim(robot))
    except Exception:
        return False


# Default True: on open, hide door visuals and disable collisions to avoid hinge squeeze; on close, restore then drive joints closed.
# NAV_DOOR_OPEN_HIDE=0/false/no uses joint-only motion again.
_USE_DOOR_OPEN_HIDE = os.environ.get("NAV_DOOR_OPEN_HIDE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# door.name -> per-link visibility and collision mesh collision_enabled (for restore on close)
_door_hide_snapshots: dict = {}


def clear_door_hide_snapshots() -> None:
    """Clear snapshot dict only (does not restore USD). On reset use restore_all_doors_from_hide_snapshots."""
    _door_hide_snapshots.clear()


def restore_all_doors_from_hide_snapshots(doors) -> None:
    """Call before env.reset(): restore visibility/collision from snapshots and clear _door_hide_snapshots.
    Clearing snapshots without restore leaves doors invisible."""
    global _door_hide_snapshots
    if not _door_hide_snapshots:
        return
    by_name = {}
    for d in doors or []:
        nm = getattr(d, "name", None)
        if nm:
            by_name[str(nm)] = d
    for nm, snap in list(_door_hide_snapshots.items()):
        d = by_name.get(nm)
        if d is not None:
            try:
                _door_restore_link_passage_state(d, snap)
            except Exception as ex:
                print(f"⚠️ Failed to restore door snapshot before reset {nm}: {ex}")
    _door_hide_snapshots.clear()
    try:
        for _ in range(5):
            og.sim.step()
    except Exception:
        pass


def ensure_all_listed_doors_visible_and_collidable(doors) -> None:
    """After env.reset() and re-listing doors: restore visibility/collision for any hide-open leftovers."""
    for door in doors or []:
        if not hasattr(door, "links"):
            continue
        for link in door.links.values():
            try:
                link.visible = True
            except Exception:
                pass
            try:
                if getattr(link, "has_collision_meshes", False) and link.collision_meshes:
                    link.enable_collisions()
            except Exception:
                pass


def _door_capture_link_passage_state(door) -> dict:
    snap = {}
    for link_name, link in door.links.items():
        vis = True
        try:
            vis = bool(link.visible)
        except Exception:
            pass
        col = {}
        try:
            if getattr(link, "has_collision_meshes", False) and link.collision_meshes:
                for mesh_name, mesh in link.collision_meshes.items():
                    try:
                        col[mesh_name] = bool(mesh.collision_enabled)
                    except Exception:
                        col[mesh_name] = True
        except Exception:
            pass
        snap[link_name] = {"visible": vis, "collision_meshes": col}
    return snap


def _door_apply_hide_for_passage(door) -> None:
    for link in door.links.values():
        try:
            link.visible = False
        except Exception:
            pass
        try:
            if getattr(link, "has_collision_meshes", False) and link.collision_meshes:
                link.disable_collisions()
        except Exception:
            pass


def _door_restore_link_passage_state(door, snap: dict) -> None:
    for link_name, link in door.links.items():
        st = snap.get(link_name)
        if st is None:
            continue
        try:
            link.visible = st["visible"]
        except Exception:
            pass
        for mesh_name, enabled in st.get("collision_meshes", {}).items():
            try:
                if mesh_name in link.collision_meshes:
                    link.collision_meshes[mesh_name].collision_enabled = bool(enabled)
            except Exception:
                pass


def nav_map_reflects_door_state() -> bool:
    """
    If True, traversable grid paints closed-door regions as blocked; default False keeps doorways walkable
    for planning (independent of hinge angle) so A* does not lose paths when doors read as closed.
    Env: NAV_MAP_REFLECT_DOOR_STATE=1/true to enable.
    """
    return os.environ.get("NAV_MAP_REFLECT_DOOR_STATE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def nav_close_all_doors_after_plan_enabled() -> bool:
    """Deprecated: kept so old imports do not break; always False."""
    return False


def apply_doors_push_resistance(doors) -> None:
    """
    Raise PhysX jointFriction on door hinges so the base/body does not push passive doors open.
    DOOR_RESIST_ROBOT_PUSH=0/false/no disables; DOOR_JOINT_FRICTION overrides default friction.
    """
    if os.getenv("DOOR_RESIST_ROBOT_PUSH", "1").strip().lower() in ("0", "false", "no"):
        return
    try:
        friction = float(os.getenv("DOOR_JOINT_FRICTION", "80.0"))
    except ValueError:
        friction = 80.0
    n = 0
    for door in doors or []:
        if not hasattr(door, "joints") or not door.joints:
            continue
        for jname, joint in door.joints.items():
            try:
                if not getattr(joint, "is_single_dof", False):
                    continue
                joint.friction = float(friction)
                n += 1
            except Exception as e:
                print(f"⚠️ Door {getattr(door, 'name', '?')} joint {jname}: failed to set friction: {e}")
    if n:
        print(
            f"✅ Raised jointFriction={friction} on {n} door hinge(s) (reduces passive doors being pushed open; "
            "set DOOR_RESIST_ROBOT_PUSH=0 to disable)"
        )


def close_all_scene_doors(
    doors,
    scene,
    floor,
    visualizer,
    original_map_dict,
    door_joint_limits,
    door_positions,
    door_states,
    skip_doors=None,
    door_map_regions=None,
):
    """
    Close all doors in the scene and refresh the traversable map (debug/script; main flow no longer uses this after plan).
    """
    if not doors:
        return
    skip = skip_doors if skip_doors is not None else set()
    for idx, door in enumerate(doors, 1):
        if idx in skip:
            continue
        if door_states.get(door.name) == "close":
            continue
        control_door(
            door,
            idx,
            "close",
            scene,
            floor,
            visualizer,
            original_map_dict,
            door_joint_limits,
            door_positions,
            door_states,
            doors=doors,
            door_map_regions=door_map_regions,
        )


class NonBlockingInput:
    """Non-blocking keyboard reader. Falls back to a no-op when stdin is not a TTY
    (e.g. running headless under a parent process that pipes stdin)."""
    def __init__(self):
        self.old_settings = None
        self._enabled = False

    def __enter__(self):
        try:
            if not sys.stdin.isatty():
                self._enabled = False
                return self
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._enabled = True
        except (termios.error, OSError, ValueError):
            self._enabled = False
            self.old_settings = None
        return self

    def __exit__(self, type, value, traceback):
        if self._enabled and self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except (termios.error, OSError, ValueError):
                pass

    def get_data(self):
        """Return one input char without blocking, or None when stdin is not a TTY / no key pressed."""
        if not self._enabled:
            return None
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
        except (OSError, ValueError):
            return None
        return None


class MapVisualizer:
    """Real-time traversable-map visualization."""
    def __init__(self, trav_map_obj, floor=0, border_pixels=10, doors=None, door_states=None):
        self.trav_map_obj = trav_map_obj
        self.floor = floor
        self.clicked_goal = None
        self.waiting_for_click = False
        self._goal_input_state = "idle"  # "idle" | "position" | "orientation"
        self._goal_with_orientation_result = None  # (world_pos, goal_yaw_rad) when done
        self._orientation_line = None
        self._goal_cropped_col = None
        self._goal_cropped_row = None
        self._goal_world = None
        self._cid_motion = None
        self._cid_release = None
        self.doors = doors if doors is not None else []
        self.door_states = door_states if door_states is not None else {}
        self.door_markers = []
        
        # Load map tensor
        trav_map_full = trav_map_obj.floor_map[floor]
        if th.is_tensor(trav_map_full):
            trav_map_full = trav_map_full.cpu().numpy()
        
        # Bounding box of traversable cells
        rows, cols = np.where(trav_map_full > 0)
        if len(rows) == 0:
            self.crop_min_row, self.crop_max_row = 0, trav_map_full.shape[0]
            self.crop_min_col, self.crop_max_col = 0, trav_map_full.shape[1]
        else:
            self.crop_min_row = max(0, rows.min() - border_pixels)
            self.crop_max_row = min(trav_map_full.shape[0], rows.max() + border_pixels + 1)
            self.crop_min_col = max(0, cols.min() - border_pixels)
            self.crop_max_col = min(trav_map_full.shape[1], cols.max() + border_pixels + 1)
        
        # Crop map to ROI
        self.trav_map = trav_map_full[
            self.crop_min_row:self.crop_max_row,
            self.crop_min_col:self.crop_max_col
        ]
        
        print(f"📏 Map crop: {trav_map_full.shape} -> {self.trav_map.shape}")
        
        # Matplotlib interactive
        plt.ion()
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.manager.set_window_title('Navigation with Doors')
        
        # Show map image
        self.map_image = self.ax.imshow(
            self.trav_map, 
            cmap='gray',
            origin='lower',
            extent=[0, self.trav_map.shape[1], 0, self.trav_map.shape[0]],
            alpha=0.7
        )
        
        # Markers
        self.robot_marker, = self.ax.plot([], [], 'go', markersize=12, label='Robot')
        self.goal_marker, = self.ax.plot([], [], 'r*', markersize=20, label='Goal')
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label='Path')
        self.current_waypoint_marker, = self.ax.plot([], [], 'co', markersize=8, label='Current Waypoint')
        # Unreachable goals (path failures) for debug overlay
        self.debug_goal_positions = []  # list of (2,) world pos
        self.debug_goal_marker, = self.ax.plot([], [], 'm*', markersize=14, label='Unreachable')
        # Pick&Place: world XY hints vs current A* goal (red star)
        self._pp_grasp_nav_xy = None
        self._pp_place_nav_xy = None
        self._pp_object_xy = None
        self._pp_target_surface_xy = None
        self.pp_grasp_marker, = self.ax.plot(
            [], [], 's', color='gold', markersize=10, markeredgecolor='black', markeredgewidth=0.6,
            label='Pick approach',
        )
        self.pp_place_marker, = self.ax.plot(
            [], [], '^', color='cyan', markersize=10, markeredgecolor='black', markeredgewidth=0.6,
            label='Place approach',
        )
        self.pp_obj_marker, = self.ax.plot(
            [], [], 'o', color='darkorange', markersize=6, alpha=0.9, markeredgecolor='black', markeredgewidth=0.4,
            label='Held object',
        )
        self.pp_tgt_marker, = self.ax.plot(
            [], [], 'D', color='magenta', markersize=7, alpha=0.85, markeredgecolor='black', markeredgewidth=0.4,
            label='Place target',
        )
        
        # Door markers (filled in update_door_markers)
        self.door_markers = []
        # nav_obs_box_* footprints as translucent rects, refreshed in update()
        self.nav_obstacle_patches = []
        self._nav_obstacle_scene = None
        self._nav_obstacle_names_ref = None
        self._nav_obstacle_half_m = float(os.environ.get("NAV_MAP_OBSTACLE_HALF_M", "0.21"))
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            self.robot_marker,
            self.goal_marker,
            self.pp_grasp_marker,
            self.pp_place_marker,
            self.pp_obj_marker,
            self.pp_tgt_marker,
            self.debug_goal_marker,
            self.path_line,
            self.current_waypoint_marker,
            Patch(facecolor='#8B4513', alpha=0.5, label='Nav obstacle box'),
            Patch(facecolor='red', label='Door (closed)'),
            Patch(facecolor='green', label='Door (open)')
        ]
        
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        self.ax.set_title('Navigation with Doors - Real-time Map')
        self.ax.legend(handles=legend_elements, loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        
        # Mouse click for goal picking
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Initial door markers
        self.update_door_markers()
    
    def update_map_display(self):
        """Refresh map image after door toggles."""
        trav_map_full = self.trav_map_obj.floor_map[self.floor]
        if th.is_tensor(trav_map_full):
            trav_map_full = trav_map_full.cpu().numpy()
        
        # Crop map to ROI
        self.trav_map = trav_map_full[
            self.crop_min_row:self.crop_max_row,
            self.crop_min_col:self.crop_max_col
        ]
        
        # Push new raster to imshow
        self.map_image.set_data(self.trav_map)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def map_to_world(self, map_pos):
        """Map pixel coords -> world XY."""
        map_pos_tensor = th.tensor(map_pos, dtype=th.float32)
        world_pos = self.trav_map_obj.map_to_world(map_pos_tensor)
        if th.is_tensor(world_pos):
            world_pos = world_pos.cpu().numpy()
        return world_pos
    
    def on_click(self, event):
        """First click sets goal position; then drag for heading."""
        if not self.waiting_for_click or self._goal_input_state != "position":
            return
        if event.inaxes != self.ax:
            return
        click_x = event.xdata
        click_y = event.ydata
        if click_x is None or click_y is None:
            return
        click_col = int(round(click_x))
        click_row = int(round(click_y))
        if not (0 <= click_row < self.trav_map.shape[0] and 0 <= click_col < self.trav_map.shape[1]):
            print("❌ Click outside map bounds")
            return
        if self.trav_map[click_row, click_col] == 0:
            print("❌ Clicked non-traversable cell; pick a free (white) cell")
            return
        full_map_row = click_row + self.crop_min_row
        full_map_col = click_col + self.crop_min_col
        world_pos = self.map_to_world(np.array([full_map_row, full_map_col]))
        print(f"✅ Goal: pixels ({click_col}, {click_row}) -> world ({world_pos[0]:.2f}, {world_pos[1]:.2f})")
        print("   Drag for heading, release to finish (no drag = no heading)")
        self.clicked_goal = world_pos
        self._goal_world = world_pos.copy() if hasattr(world_pos, 'copy') else np.array(world_pos)
        self._goal_cropped_col = click_col
        self._goal_cropped_row = click_row
        self._goal_input_state = "orientation"
        self.goal_marker.set_data([click_col], [click_row])
        if self._orientation_line is not None:
            self._orientation_line.remove()
        self._orientation_line, = self.ax.plot(
            [click_col, click_col], [click_row, click_row], "b-", linewidth=2, alpha=0.8
        )
        self.fig.canvas.draw_idle()
        self._cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_orientation_motion)
        self._cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_orientation_release)

    def _on_orientation_motion(self, event):
        """Update heading guide while dragging."""
        if self._goal_input_state != "orientation" or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._orientation_line.set_data(
            [self._goal_cropped_col, event.xdata],
            [self._goal_cropped_row, event.ydata],
        )
        self.fig.canvas.draw_idle()

    def _on_orientation_release(self, event):
        """Mouse release: finalize goal yaw from drag direction."""
        if self._goal_input_state != "orientation":
            return
        if self._cid_motion is not None:
            self.fig.canvas.mpl_disconnect(self._cid_motion)
            self._cid_motion = None
        if self._cid_release is not None:
            self.fig.canvas.mpl_disconnect(self._cid_release)
            self._cid_release = None
        self._goal_input_state = "idle"
        release_x = event.xdata
        release_y = event.ydata
        if release_x is None or release_y is None or event.inaxes != self.ax:
            self._goal_with_orientation_result = (self._goal_world, None)
            self.waiting_for_click = False
            self._cleanup_orientation_ui()
            return
        release_col = int(round(release_x))
        release_row = int(round(release_y))
        full_map_row = release_row + self.crop_min_row
        full_map_col = release_col + self.crop_min_col
        release_world = self.map_to_world(np.array([full_map_row, full_map_col]))
        dx = release_world[0] - self._goal_world[0]
        dy = release_world[1] - self._goal_world[1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 0.05:
            goal_yaw_rad = None
        else:
            goal_yaw_rad = float(np.arctan2(dy, dx))
            print(f"   Heading: {math.degrees(goal_yaw_rad):.1f}°")
        self._goal_with_orientation_result = (self._goal_world, goal_yaw_rad)
        self.waiting_for_click = False
        self._cleanup_orientation_ui()

    def _cleanup_orientation_ui(self):
        """Remove heading line and reset title."""
        if self._orientation_line is not None:
            self._orientation_line.remove()
            self._orientation_line = None
        self.ax.set_title("Navigation with Doors - Real-time Map")
        self.fig.canvas.draw_idle()

    def wait_for_click(self):
        """Legacy: wait for click only; returns (world_pos, None)."""
        r = self.wait_for_goal_with_orientation()
        return r[0] if r else None

    def wait_for_goal_with_orientation(self):
        """Click goal + drag heading (RViz-like). Returns (world_pos, goal_yaw_rad or None)."""
        self.clicked_goal = None
        self._goal_with_orientation_result = None
        self._goal_input_state = "position"
        self.waiting_for_click = True
        self.ax.set_title("Click goal, drag heading (release to finish)")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        print("🖱️  Click map for goal, drag direction for heading, release to finish...")
        import time
        while self.waiting_for_click:
            self.fig.canvas.flush_events()
            time.sleep(0.05)
        self._goal_input_state = "idle"
        return self._goal_with_orientation_result
    
    def world_to_map(self, world_pos):
        """World XY -> map pixel coords."""
        if th.is_tensor(world_pos):
            world_pos = world_pos.cpu().numpy()
        
        map_pos = self.trav_map_obj.world_to_map(th.tensor(world_pos, dtype=th.float32))
        if th.is_tensor(map_pos):
            map_pos = map_pos.cpu().numpy()
        
        return map_pos
    
    def add_debug_goal(self, world_pos):
        """Record an unreachable goal for purple-star debug overlay."""
        pos = np.asarray(world_pos, dtype=np.float64)
        if pos.size >= 2:
            self.debug_goal_positions.append(pos[:2].copy())
    
    def clear_debug_goals(self):
        """Clear unreachable-goal markers (e.g. new pick cycle)."""
        self.debug_goal_positions.clear()
        self.clear_pick_place_nav_markers()

    def clear_pick_place_nav_markers(self):
        """Clear Pick&Place overlay markers on the map."""
        self._pp_grasp_nav_xy = None
        self._pp_place_nav_xy = None
        self._pp_object_xy = None
        self._pp_target_surface_xy = None

    def set_pick_place_nav_markers(
        self,
        grasp_nav_xy=None,
        place_nav_xy=None,
        object_xy=None,
        target_surface_xy=None,
    ):
        """
        World XY hints for the current Pick&Place on the map.
        - grasp_nav_xy: approach to grasp (current candidate)
        - place_nav_xy: approach to place (valid once nav_to_target runs)
        - object_xy: small object center
        - target_surface_xy: placement receptacle center
        """
        def _to2d(xy):
            if xy is None:
                return None
            a = np.asarray(xy, dtype=np.float64).reshape(-1)
            if a.size < 2:
                return None
            return a[:2].copy()

        self._pp_grasp_nav_xy = _to2d(grasp_nav_xy)
        self._pp_place_nav_xy = _to2d(place_nav_xy)
        self._pp_object_xy = _to2d(object_xy)
        self._pp_target_surface_xy = _to2d(target_surface_xy)

    def _plot_world_xy_on_map(self, xy, marker_line):
        if xy is None:
            marker_line.set_data([], [])
            return
        m = self.world_to_map(xy)
        mc = m - np.array([self.crop_min_row, self.crop_min_col], dtype=np.float64)
        marker_line.set_data([mc[1]], [mc[0]])
    
    def update(self, robot_pos, goal_pos=None, path=None, current_waypoint=None):
        """Redraw overlays from latest sim state."""
        robot_map_pos = self.world_to_map(robot_pos)
        robot_map_pos_cropped = robot_map_pos - np.array([self.crop_min_row, self.crop_min_col])
        self.robot_marker.set_data([robot_map_pos_cropped[1]], [robot_map_pos_cropped[0]])
        
        if goal_pos is not None:
            goal_map_pos = self.world_to_map(goal_pos)
            goal_map_pos_cropped = goal_map_pos - np.array([self.crop_min_row, self.crop_min_col])
            self.goal_marker.set_data([goal_map_pos_cropped[1]], [goal_map_pos_cropped[0]])
        else:
            self.goal_marker.set_data([], [])
        
        # Unreachable goals (magenta stars)
        if self.debug_goal_positions:
            debug_map = np.array([self.world_to_map(p) for p in self.debug_goal_positions])
            debug_cropped = debug_map - np.array([self.crop_min_row, self.crop_min_col])
            self.debug_goal_marker.set_data(debug_cropped[:, 1], debug_cropped[:, 0])
        else:
            self.debug_goal_marker.set_data([], [])
        
        if path is not None and len(path) > 0:
            path_map_coords = np.array([self.world_to_map(p) for p in path])
            path_map_coords_cropped = path_map_coords - np.array([self.crop_min_row, self.crop_min_col])
            self.path_line.set_data(path_map_coords_cropped[:, 1], path_map_coords_cropped[:, 0])
        else:
            self.path_line.set_data([], [])
        
        if current_waypoint is not None:
            waypoint_map_pos = self.world_to_map(current_waypoint)
            waypoint_map_pos_cropped = waypoint_map_pos - np.array([self.crop_min_row, self.crop_min_col])
            self.current_waypoint_marker.set_data([waypoint_map_pos_cropped[1]], [waypoint_map_pos_cropped[0]])
        else:
            self.current_waypoint_marker.set_data([], [])

        self._plot_world_xy_on_map(self._pp_grasp_nav_xy, self.pp_grasp_marker)
        self._plot_world_xy_on_map(self._pp_place_nav_xy, self.pp_place_marker)
        self._plot_world_xy_on_map(self._pp_object_xy, self.pp_obj_marker)
        self._plot_world_xy_on_map(self._pp_target_surface_xy, self.pp_tgt_marker)

        self._refresh_nav_obstacle_patches()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_nav_obstacle_tracking(self, scene, names_list, half_extent_m=None):
        """
        Track nav_obs_box_* footprints on the map (half-extent matches spawn_nav_obstacle_boxes default).
        Pass the live names list by reference so spawn/reset can extend/clear in place.
        """
        self._nav_obstacle_scene = scene
        self._nav_obstacle_names_ref = names_list
        if half_extent_m is not None:
            self._nav_obstacle_half_m = float(half_extent_m)

    def _refresh_nav_obstacle_patches(self):
        from matplotlib.patches import Rectangle

        for p in self.nav_obstacle_patches:
            try:
                p.remove()
            except Exception:
                pass
        self.nav_obstacle_patches = []

        scene = self._nav_obstacle_scene
        names_ref = self._nav_obstacle_names_ref
        if scene is None or not names_ref:
            return

        half = float(self._nav_obstacle_half_m)
        crop = np.array([self.crop_min_row, self.crop_min_col], dtype=np.float64)

        for nm in names_ref:
            if not nm:
                continue
            try:
                obj = scene.object_registry("name", nm)
            except Exception:
                obj = None
            if obj is None:
                continue
            try:
                pos, _ = obj.get_position_orientation()
            except Exception:
                continue
            if th.is_tensor(pos):
                pos = pos.cpu().numpy()
            pos = np.asarray(pos, dtype=np.float64).reshape(-1)
            if pos.size < 2:
                continue
            wx, wy = float(pos[0]), float(pos[1])
            corners = np.array(
                [
                    [wx - half, wy - half],
                    [wx + half, wy - half],
                    [wx + half, wy + half],
                    [wx - half, wy + half],
                ],
                dtype=np.float64,
            )
            mc = np.array([self.world_to_map(c) for c in corners], dtype=np.float64)
            cc = mc - crop
            xs = cc[:, 1]
            ys = cc[:, 0]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            if x1 <= x0:
                x1 = x0 + 1.0
            if y1 <= y0:
                y1 = y0 + 1.0
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=1.2,
                edgecolor="#5c3d1e",
                facecolor="#8B4513",
                alpha=0.42,
            )
            self.ax.add_patch(rect)
            self.nav_obstacle_patches.append(rect)
    
    def update_door_markers(self):
        """Refresh door markers from door world positions on the cropped map."""
        # Remove old markers
        for marker in self.door_markers:
            marker.remove()
        self.door_markers = []
        
        if not self.doors:
            return
        
        # Full-resolution map for world->map
        trav_map_full = self.trav_map_obj.floor_map[self.floor]
        if th.is_tensor(trav_map_full):
            trav_map_full = trav_map_full.cpu().numpy()
        
        # One marker per door
        for door in self.doors:
            # Door center in world XY
            door_pos_3d, _ = door.get_position_orientation()
            door_pos_2d = door_pos_3d[:2]
            
            if th.is_tensor(door_pos_2d):
                door_pos_2d = door_pos_2d.cpu().numpy()
            
            door_map_pos = self.world_to_map(door_pos_2d)
            door_map_pos_cropped = door_map_pos - np.array([self.crop_min_row, self.crop_min_col])
            
            # Color by open/closed
            door_state = self.door_states.get(door.name, None)
            
            if door_state == "open":
                color = 'green'  # open
                marker_style = 'o'  # circle
                markersize = 20
            else:
                color = 'red'  # closed
                marker_style = 's'  # square
                markersize = 20
            
            # Plot center marker
            marker, = self.ax.plot(
                door_map_pos_cropped[1], 
                door_map_pos_cropped[0], 
                marker_style, 
                color=color, 
                markersize=markersize, 
                alpha=0.7,
                markeredgecolor='black',
                markeredgewidth=2
            )
            self.door_markers.append(marker)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the matplotlib figure."""
        for p in getattr(self, "nav_obstacle_patches", []):
            try:
                p.remove()
            except Exception:
                pass
        self.nav_obstacle_patches = []
        plt.close(self.fig)


def hide_scene_ceilings(scene) -> int:
    """
    Hide ceiling-category objects/links for top-down views.
    Set NAV_SHOW_CEILINGS=1/true to keep ceilings visible.
    """
    if os.environ.get("NAV_SHOW_CEILINGS", "0").strip().lower() in ("1", "true", "yes", "on"):
        return 0
    n = 0
    for obj in list(getattr(scene, "objects", []) or []):
        cat = (getattr(obj, "category", None) or "").lower()
        if cat not in ("ceilings", "ceiling") and "ceiling" not in cat:
            continue
        try:
            obj.visible = False
            n += 1
        except Exception:
            pass
        links = getattr(obj, "links", None)
        if isinstance(links, dict):
            for link in links.values():
                try:
                    link.visible = False
                    n += 1
                except Exception:
                    pass
    return n


def list_all_doors(scene):
    """Return all door-like objects in the scene."""
    doors = []
    for obj in scene.objects:
        category = obj.category if hasattr(obj, 'category') else "unknown"
        if "door" in category.lower():
            doors.append(obj)
    return doors


def print_doors(doors):
    """Pretty-print door list to stdout."""
    print("\n" + "="*80)
    print("🚪 All interactive doors in the scene:")
    print("="*80)
    
    for i, door in enumerate(doors, 1):
        name = door.name if hasattr(door, 'name') else "unknown"
        pos, _ = door.get_position_orientation()
        pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        
        is_open = "unknown"
        if hasattr(door, 'states') and Open in door.states:
            is_open = "open" if door.states[Open].get_value() else "closed"
        
        print(f"\nDoor #{i}: {name}")
        print(f"  Position: {pos_str}")
        print(f"  State: {is_open}")
    
    print("\n" + "="*80)
    print(f"Total doors: {len(doors)}")
    print("="*80)


def get_door_map_region(door, trav_map_obj, floor=0, shrink_factor=1.0):
    """
    Axis-aligned door footprint in map pixel indices.

    Args:
        door: door object
        trav_map_obj: TraversableMap
        floor: floor index
        shrink_factor: 0..1 shrink of AABB extent (avoid over-erasing pixels)

    Returns:
        (min_row, max_row, min_col, max_col) or None
    """
    try:
        # Door AABB in world
        bbox_center = door.aabb_center
        bbox_extent = door.aabb_extent
        
        if bbox_center is None or bbox_extent is None:
            return None
        
        # To numpy
        if th.is_tensor(bbox_center):
            bbox_center = bbox_center.cpu().numpy()
        if th.is_tensor(bbox_extent):
            bbox_extent = bbox_extent.cpu().numpy()
        
        # Shrink extent from center
        shrunk_extent = bbox_extent * shrink_factor
        
        # AABB corners
        lower_corner = bbox_center - shrunk_extent / 2.0
        upper_corner = bbox_center + shrunk_extent / 2.0
        
        # XY only
        min_xy_world = lower_corner[:2]
        max_xy_world = upper_corner[:2]
        
        # World -> map indices
        min_xy_map = trav_map_obj.world_to_map(th.tensor(min_xy_world, dtype=th.float32))
        max_xy_map = trav_map_obj.world_to_map(th.tensor(max_xy_world, dtype=th.float32))
        
        if th.is_tensor(min_xy_map):
            min_xy_map = min_xy_map.cpu().numpy()
        if th.is_tensor(max_xy_map):
            max_xy_map = max_xy_map.cpu().numpy()
        
        # Map size
        map_h, map_w = trav_map_obj.floor_map[floor].shape
        
        # Clamp to map bounds
        min_r = max(0, int(min_xy_map[0]))
        max_r = min(map_h, int(max_xy_map[0]) + 1)  # +1: slice end is exclusive
        min_c = max(0, int(min_xy_map[1]))
        max_c = min(map_w, int(max_xy_map[1]) + 1)
        
        # Non-empty slice
        if max_r <= min_r:
            max_r = min_r + 1
        if max_c <= min_c:
            max_c = min_c + 1
        
        return (min_r, max_r, min_c, max_c)
    except Exception as e:
        print(f"⚠️ Failed to get map region for door {door.name}: {e}")
        return None


def update_traversable_map(scene, floor=0, original_map=None, doors=None, door_states=None, door_map_regions=None):
    """
    Refresh traversable map after door open/close.

    By default (NAV_MAP_REFLECT_DOOR_STATE off) doorways stay traversable for A*;
    when enabled, closed-door regions become obstacles on the grid.

    Original map uses 0 obstacle / 255 free; door cells are 0 when closed and become 255 when open.
    Each update starts from the stored original map and applies all door states.

    Args:
        scene: scene handle
        floor: floor index
        original_map: uncorroded map with all doors closed (kept immutable)
        doors: door object list
        door_states: {door_name: "open" | "close"}
        door_map_regions: {door_name: (min_r, max_r, min_c, max_c)}

    Returns:
        None; writes scene.trav_map.floor_map[floor]
    """
    print("🔄 Updating traversable map...")
    
    # Let door state settle in the sim
    for _ in range(30):
        og.sim.step()
    
    # Start from the stored original (all doors closed snapshot)
    if original_map is None:
        original_map = scene.trav_map.floor_map[floor].clone()
    
    # Clone to working map; keep dtype/device consistent
    if th.is_tensor(original_map):
        updated_map = original_map.clone()
    else:
        updated_map = th.tensor(original_map).clone()
    
    # Optional: paint closed doors as obstacles when NAV_MAP_REFLECT_DOOR_STATE is on
    if doors is not None and door_states is not None and door_map_regions is not None:
        trav_map_obj = scene.trav_map
        reflect_doors_on_map = nav_map_reflects_door_state()

        for door in doors:
            door_name = door.name
            door_state = door_states.get(door_name, None)
            if not reflect_doors_on_map:
                door_state = "open"
            elif door_state is None:
                door_state = "close"
            door_region = door_map_regions.get(door_name, None)
            
            if door_region is None:
                # Lazily cache door footprint on first use
                door_region = get_door_map_region(door, trav_map_obj, floor)
                if door_region is not None:
                    door_map_regions[door_name] = door_region
                else:
                    continue
            
            # Door mesh may move when open; we still edit the closed-time footprint stored at init.
            
            min_r, max_r, min_c, max_c = door_region
            
            # Apply open/close to that footprint: 0 obstacle / 255 free in TraversableMap convention
            region_original = original_map[min_r:max_r, min_c:max_c]
            
            # Work in numpy for boolean indexing
            if th.is_tensor(region_original):
                region_original_np = region_original.cpu().numpy()
            else:
                region_original_np = region_original
            
            if door_state == "open":
                # Open: obstacle pixels in original door mask -> free (255)
                region_mask = (region_original_np == 0)  # door footprint in original map
                if region_mask.any():
                    current_region = updated_map[min_r:max_r, min_c:max_c]
                    if th.is_tensor(current_region):
                        current_region_np = current_region.cpu().numpy()
                    else:
                        current_region_np = current_region
                    
                    current_region_np[region_mask] = 255
                    # Write back as tensor if needed
                    if th.is_tensor(updated_map):
                        updated_map[min_r:max_r, min_c:max_c] = th.tensor(current_region_np, dtype=updated_map.dtype, device=updated_map.device)
                    else:
                        updated_map[min_r:max_r, min_c:max_c] = current_region_np
                    
                    num_pixels = int(np.sum(region_mask))
                    print(
                        f"  ✅ Door {door_name} open: region [{min_r}:{max_r}, {min_c}:{max_c}] "
                        f"marked {num_pixels} obstacle pixel(s) (0) as free (255)"
                    )
                else:
                    print(
                        f"  ℹ️  Door {door_name} open: region [{min_r}:{max_r}, {min_c}:{max_c}] "
                        f"already free; no map change"
                    )
                    
            elif door_state == "close":
                # Close: restore obstacle pixels from original mask
                region_mask = (region_original_np == 0)
                if region_mask.any():
                    current_region = updated_map[min_r:max_r, min_c:max_c]
                    if th.is_tensor(current_region):
                        current_region_np = current_region.cpu().numpy()
                    else:
                        current_region_np = current_region
                    
                    current_region_np[region_mask] = 0
                    # Write back as tensor if needed
                    if th.is_tensor(updated_map):
                        updated_map[min_r:max_r, min_c:max_c] = th.tensor(current_region_np, dtype=updated_map.dtype, device=updated_map.device)
                    else:
                        updated_map[min_r:max_r, min_c:max_c] = current_region_np
                    
                    num_pixels = int(np.sum(region_mask))
                    print(
                        f"  ✅ Door {door_name} closed: region [{min_r}:{max_r}, {min_c}:{max_c}] "
                        f"restored {num_pixels} pixel(s) to obstacle (0)"
                    )
                else:
                    print(
                        f"  ℹ️  Door {door_name} closed: region [{min_r}:{max_r}, {min_c}:{max_c}] "
                        f"already obstacle; no map change"
                    )
    
    # Erode traversable map for robot radius margin
    erosion_radius_meters = 0.4
    map_resolution = scene.trav_map.map_resolution
    radius_pixel = int(math.ceil(erosion_radius_meters / map_resolution))
    
    if th.is_tensor(updated_map):
        updated_map_np = updated_map.cpu().numpy()
    else:
        updated_map_np = updated_map
    
    if radius_pixel > 0:
        kernel = np.ones((radius_pixel, radius_pixel), dtype=np.uint8)
        trav_map_eroded = cv2.erode(updated_map_np.astype(np.uint8), kernel)
        trav_map_eroded[trav_map_eroded < 255] = 0  # binarize to {0,255}
        scene.trav_map.floor_map[floor] = th.tensor(trav_map_eroded)
    else:
        updated_map_np[updated_map_np < 255] = 0
        if th.is_tensor(updated_map):
            scene.trav_map.floor_map[floor] = th.tensor(updated_map_np)
        else:
            scene.trav_map.floor_map[floor] = updated_map_np
    
    print("✅ Traversable map updated")
    # original_map stays immutable; scene.trav_map.floor_map[floor] holds the updated grid


def control_door(door, door_index, action, scene, floor, visualizer, original_map_dict, door_joint_limits, door_positions, door_states, doors=None, door_map_regions=None, fully=True):
    """
    Open/close a door. With NAV_DOOR_OPEN_HIDE (default): hide meshes and disable collisions on open (no hinge motion);
    on close, set joints while still hidden then restore visibility/collisions to avoid pushing the robot.
    If NAV_DOOR_OPEN_HIDE=0, use legacy joint targets (e.g. slide into wall) for open.

    Args:
        door: door object
        door_index: 1-based index
        action: "open" or "close"
        scene: scene handle
        floor: floor index
        visualizer: map UI or None
        original_map_dict: floor -> original traversable tensor
        door_joint_limits: saved limits {door_name: {joint_name: (lower, upper)}}
        door_positions: per-index angles {door_index: {"open": deg, "close": deg}}
        door_states: current states {door_name: "open"|"close"}
        doors: all doors for map refresh
        door_map_regions: cached door regions on grid
        fully: unused legacy flag
    """
    try:
        # All joints on this door
        if not hasattr(door, 'joints') or len(door.joints) == 0:
            print(f"❌ Door {door.name} has no controllable joints")
            return False
        
        # Skip if already in requested state
        if door.name in door_states and door_states[door.name] == action:
            current_state_str = "open" if action == "open" else "closed"
            print(f"ℹ️  Door {door.name} is already {current_state_str}; skipping")
            return True
        
        # First use: snapshot original joint limits
        if door.name not in door_joint_limits:
            door_joint_limits[door.name] = {}
            for joint_name, joint in door.joints.items():
                lower_val = joint.lower_limit
                upper_val = joint.upper_limit
                
                if th.is_tensor(lower_val):
                    lower_val = lower_val.item()
                if th.is_tensor(upper_val):
                    upper_val = upper_val.item()
                
                door_joint_limits[door.name][joint_name] = (lower_val, upper_val)
                print(f"💾 Saved door {door.name} joint {joint_name} original limits: [{lower_val:.3f}, {upper_val:.3f}]")

        restore_snap = None
        if _USE_DOOR_OPEN_HIDE and action == "close":
            restore_snap = _door_hide_snapshots.pop(door.name, None)

        if _USE_DOOR_OPEN_HIDE and action == "open":
            if door.name not in _door_hide_snapshots:
                _door_hide_snapshots[door.name] = _door_capture_link_passage_state(door)
            _door_apply_hide_for_passage(door)
            for _ in range(5):
                og.sim.step()
            door_states[door.name] = "open"
            try:
                from door_barrier_walls import update_barrier_wall_for_door
                update_barrier_wall_for_door(door.name, "open")
            except Exception:
                pass
            print(f"🚪 Door {door.name}: hidden, collisions disabled (open passage, no hinge motion)")
            original_map = original_map_dict.get(floor)
            update_traversable_map(
                scene,
                floor,
                original_map,
                doors=doors,
                door_states=door_states,
                door_map_regions=door_map_regions,
            )
            if visualizer is not None:
                visualizer.update_map_display()
                visualizer.update_door_markers()
                print("✅ Map display updated")
            return True
        
        # Log joint state before motion
        print(f"\n🔧 Door {door.name} joint info:")
        for joint_name, joint in door.joints.items():
            current_pos = joint.get_state()[0]
            lower_limit = joint.lower_limit
            upper_limit = joint.upper_limit
            joint_type = joint.joint_type
            
            # Tensor -> Python scalars for printing
            if th.is_tensor(current_pos):
                current_pos = current_pos.item()
            if th.is_tensor(lower_limit):
                lower_limit = lower_limit.item()
            if th.is_tensor(upper_limit):
                upper_limit = upper_limit.item()
            
            print(f"  Joint: {joint_name}")
            print(f"    Type: {joint_type}")
            print(f"    Position: {current_pos:.3f}")
            print(f"    Range: [{lower_limit:.3f}, {upper_limit:.3f}]")
        
        # Drive each joint
        for joint_name, joint in door.joints.items():
            original_lower, original_upper = door_joint_limits[door.name][joint_name]
            
            if door_index in door_positions:
                target_angle_deg = door_positions[door_index][action]
                target_angle_rad = math.radians(target_angle_deg)
                
                # Widen limits if configured angle is outside USD limits
                if target_angle_rad < original_lower:
                    new_lower_limit = target_angle_rad - 0.1
                    new_upper_limit = original_upper
                elif target_angle_rad > original_upper:
                    new_lower_limit = original_lower
                    new_upper_limit = target_angle_rad + 0.1
                else:
                    new_lower_limit = original_lower
                    new_upper_limit = original_upper
                
                joint.lower_limit = new_lower_limit
                joint.upper_limit = new_upper_limit
                target_pos = target_angle_rad
                
                print(f"  ✅ Joint {joint_name}:")
                print(f"     Door index: {door_index}, action: {action}")
                print(f"     Config angle: {target_angle_deg}° ({target_angle_rad:.3f} rad)")
                print(f"     Original limits: [{original_lower:.3f}, {original_upper:.3f}]")
                if new_lower_limit != original_lower or new_upper_limit != original_upper:
                    print(f"     Expanded limits: [{new_lower_limit:.3f}, {new_upper_limit:.3f}]")
                print(f"     Set position: {target_pos:.3f} rad ({target_angle_deg}°)")
                
                joint.set_pos(target_pos, drive=False)
                og.sim.step()
                
            else:
                if action == "open":
                    # Default: swing past upper limit into wall cavity
                    extra_offset = 1.5
                    new_upper_limit = original_upper + extra_offset
                    joint.lower_limit = original_lower
                    joint.upper_limit = new_upper_limit
                    target_pos = new_upper_limit
                    
                    print(f"  ✅ Joint {joint_name}:")
                    print(f"     Default: expand upper limit into wall")
                    print(f"     New limits: [{original_lower:.3f}, {new_upper_limit:.3f}]")
                    print(f"     Set position: {target_pos:.3f}")
                    
                    joint.set_pos(target_pos, drive=False)
                    og.sim.step()
                    joint.set_pos(target_pos, drive=False)
                    og.sim.step()
                    
                elif action == "close":
                    # Default: snap to lower limit
                    joint.lower_limit = original_lower
                    joint.upper_limit = original_upper
                    target_pos = original_lower
                    
                    print(f"  ✅ Joint {joint_name}: restored to lower limit {target_pos:.3f}")
                    
                    for _ in range(5):
                        joint.set_pos(target_pos, drive=False)
                        og.sim.step()
        
        for _ in range(10):
            og.sim.step()
            
        # Verify final joint values
        print(f"\n📍 Verifying final door joint positions:")
        has_nan = False
        for joint_name, joint in door.joints.items():
            final_pos = joint.get_state()[0]
            if th.is_tensor(final_pos):
                final_pos = final_pos.item()
            
            if math.isnan(final_pos):
                print(f"  Joint {joint_name}: ⚠️  NaN (invalid)")
                has_nan = True
            else:
                print(f"  Joint {joint_name}: {final_pos:.3f} rad ({math.degrees(final_pos):.1f}°)")
        
        if has_nan:
            print(f"\n❌ Error: door {door.name} joint position is NaN")
            print(f"   This can break physics. Consider adding door #{door_index} to the skip list.")
            if restore_snap is not None:
                _door_restore_link_passage_state(door, restore_snap)
                for _ in range(3):
                    og.sim.step()
            return False

        if restore_snap is not None:
            _door_restore_link_passage_state(door, restore_snap)
            for _ in range(5):
                og.sim.step()
            if action == "close":
                print(f"🚪 Door {door.name}: visibility and collisions restored (closed pose)")
        
        door_states[door.name] = action
        print(f"🔄 Door state updated: {door.name} -> {action}")

        # Optional barrier prims: hide when open, show when closed
        try:
            from door_barrier_walls import update_barrier_wall_for_door
            update_barrier_wall_for_door(door.name, action)
        except Exception:
            pass
        
        print(f"\n✅ Door {door.name} is now {'open' if action == 'open' else 'closed'}")
        
        # Rebuild traversable grid from immutable original_map snapshot
        original_map = original_map_dict.get(floor)
        update_traversable_map(
            scene, floor, original_map, 
            doors=doors, 
            door_states=door_states, 
            door_map_regions=door_map_regions
        )
        if visualizer is not None:
            visualizer.update_map_display()
            visualizer.update_door_markers()
            print("✅ Map display updated")
        
        return True
    except Exception as e:
        import traceback
        print(f"❌ Error controlling door: {e}")
        print(traceback.format_exc())
        return False


def _as_xy2(p):
    a = p.cpu().numpy() if th.is_tensor(p) else np.asarray(p)
    out = np.asarray(a, dtype=np.float64).reshape(-1)
    return out[:2].copy()


def _normalize_path_waypoints_list(path_waypoints_list):
    """
    get_shortest_path may return a list of waypoints or a single (N,3) Tensor.
    Do not use `if not x` on a Tensor (ambiguous bool). Normalize to a per-point list.
    """
    if path_waypoints_list is None:
        return None
    if th.is_tensor(path_waypoints_list):
        t = path_waypoints_list.detach().cpu()
        if t.numel() == 0:
            return []
        arr = np.asarray(t.numpy(), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return [arr[i] for i in range(arr.shape[0])]
    if isinstance(path_waypoints_list, np.ndarray):
        arr = np.asarray(path_waypoints_list, dtype=np.float64)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return [arr[i] for i in range(arr.shape[0])]
    try:
        return list(path_waypoints_list)
    except TypeError:
        return None


def _pure_pursuit_carrot(robot_xy, path_waypoints_list, current_wp_idx, lookahead_m):
    """
    Pure-pursuit carrot on the A* polyline: project robot to nearest segment, then advance
    along-path by lookahead_m arc length. Reduces corner-cutting vs staring at the current vertex.
    """
    try:
        lm = float(lookahead_m)
    except (TypeError, ValueError):
        return None
    if lm <= 0:
        return None
    path_waypoints_list = _normalize_path_waypoints_list(path_waypoints_list)
    if not path_waypoints_list:
        return None
    pts = np.stack([_as_xy2(w) for w in path_waypoints_list], axis=0)
    n = pts.shape[0]
    r = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
    if n == 1:
        return pts[0].copy()
    i0 = max(0, int(current_wp_idx) - 1)
    best_d = 1e18
    best_proj = None
    best_seg = 0
    for i in range(i0, n - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        lab2 = float(np.dot(ab, ab)) + 1e-12
        t = float(np.dot(r - a, ab) / lab2)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        d = float(np.linalg.norm(r - proj))
        if d < best_d:
            best_d = d
            best_proj = proj
            best_seg = i
    if best_proj is None:
        return _as_xy2(path_waypoints_list[min(int(current_wp_idx), n - 1)])

    remain = float(lm)
    cur = best_proj.copy()
    seg = int(best_seg)
    while seg < n - 1 and remain > 1e-6:
        b = pts[seg + 1]
        toward = b - cur
        dtb = float(np.linalg.norm(toward))
        if dtb < 1e-9:
            seg += 1
            if seg < n:
                cur = pts[seg].copy()
            continue
        if dtb >= remain:
            return cur + (toward / dtb) * remain
        remain -= dtb
        cur = b.copy()
        seg += 1
    return pts[-1].copy()


def _advance_waypoint_idx_on_path(robot_xy, path_waypoints, current_idx, wp_reach):
    """
    Advance waypoint index when within wp_reach of the vertex, or when projection t on a segment
    is high (near segment end). Mitigates pure-pursuit inner-corner cases where idx never advances.
    """
    path_list = _normalize_path_waypoints_list(path_waypoints)
    if not path_list:
        return int(current_idx)
    pts = np.stack([_as_xy2(w) for w in path_list], axis=0)
    n = int(pts.shape[0])
    if n == 0:
        return int(current_idx)
    i = int(np.clip(int(current_idx), 0, n - 1))
    r = np.asarray(robot_xy, dtype=np.float64).reshape(-1)[:2]
    wr = float(wp_reach)
    lateral_tol = max(wr * 1.65, 0.36)

    while i < n and float(np.linalg.norm(r - pts[i])) < wr:
        i += 1

    if n >= 2 and i < n:
        seg_lo = max(0, i - 1)
        for seg in range(seg_lo, n - 1):
            a = pts[seg]
            b = pts[seg + 1]
            ab = b - a
            lab2 = float(np.dot(ab, ab)) + 1e-12
            t = float(np.dot(r - a, ab) / lab2)
            tc = max(0.0, min(1.0, t))
            proj = a + ab * tc
            d_lat = float(np.linalg.norm(r - proj))
            if d_lat > lateral_tol:
                continue
            if tc > 0.9 and (seg + 2) <= n:
                i = max(i, seg + 2)

    return min(i, n)


def _initial_path_waypoint_idx(robot_xy, path_waypoints, max_snap_dist=1.05):
    """
    After replanning, starting at vertex 0 can make pure pursuit look backward past doorways.
    Pick a start index from projection on the polyline; return 0 if too far from the path.
    """
    path_list = _normalize_path_waypoints_list(path_waypoints)
    n = len(path_list)
    if n <= 1:
        return 0
    r = np.asarray(_as_xy2(robot_xy), dtype=np.float64).reshape(-1)[:2]
    pts = np.stack([_as_xy2(w) for w in path_list], axis=0)
    best_d = 1e18
    best_seg = 0
    best_t = 0.0
    for seg in range(n - 1):
        a, b = pts[seg], pts[seg + 1]
        ab = b - a
        lab2 = float(np.dot(ab, ab)) + 1e-12
        t = float(np.dot(r - a, ab) / lab2)
        t = max(0.0, min(1.0, t))
        proj = a + ab * t
        d = float(np.linalg.norm(r - proj))
        if d < best_d:
            best_d = d
            best_seg = seg
            best_t = t
    if best_d > float(max_snap_dist):
        return 0
    # Projection near segment end → start tracking from a forward vertex
    if best_t > 0.72:
        return min(best_seg + 2, n - 1)
    if best_t > 0.38:
        return min(best_seg + 1, n - 1)
    return min(best_seg, n - 1)


def compute_action_to_waypoint(
    robot,
    waypoint,
    max_lin_vel=None,
    max_ang_vel=None,
):
    """
    Compute [linear_vel, angular_vel] (m/s, rad/s) toward a tracking point.
    Defaults: NAV_MAX_LIN_VEL / NAV_MAX_ANG_VEL. Large heading error reduces linear speed.
    """
    if max_lin_vel is None:
        max_lin_vel = _NAV_MAX_LIN_VEL
    if max_ang_vel is None:
        max_ang_vel = _NAV_MAX_ANG_VEL
    robot_pos_ori = robot.get_position_orientation()
    robot_pos = robot_pos_ori[0][:2]
    robot_ori = robot_pos_ori[1]
    robot_yaw = _quat_xyzw_yaw_from_ori(robot_ori)

    if th.is_tensor(robot_pos):
        robot_pos = robot_pos.cpu().numpy()
    if th.is_tensor(waypoint):
        waypoint = waypoint.cpu().numpy()

    robot_pos = np.asarray(robot_pos, dtype=np.float64).reshape(-1)[:2]
    waypoint = np.asarray(waypoint, dtype=np.float64).reshape(-1)[:2]

    diff = waypoint - robot_pos
    distance = float(np.linalg.norm(diff))

    if distance < 0.08:
        return np.array([0.0, 0.0], dtype=np.float64)

    target_yaw = float(np.arctan2(diff[1], diff[0]))
    angle_diff = target_yaw - robot_yaw
    angle_diff = float(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))

    ad = abs(angle_diff)
    if ad > _NAV_TURN_IN_PLACE_RAD:
        linear_vel = 0.0
        angular_vel = float(np.clip(angle_diff * 1.6, -max_ang_vel, max_ang_vel))
    else:
        speed_cap = min(max_lin_vel, 0.18 + 0.75 * distance)
        linear_vel = float(np.clip(distance * 0.9, 0.0, speed_cap))
        align = max(0.08, math.cos(angle_diff))
        linear_vel *= align * align * align
        angular_vel = float(np.clip(angle_diff * 1.2, -max_ang_vel, max_ang_vel))

    return np.array([linear_vel, angular_vel], dtype=np.float64)


def compute_rotate_to_yaw_action(robot, target_yaw_rad, max_ang_vel=None):
    """In-place rotate toward target yaw: angular velocity only, linear velocity zero."""
    if max_ang_vel is None:
        max_ang_vel = _NAV_MAX_ANG_VEL
    robot_ori = robot.get_position_orientation()[1]
    robot_yaw = _quat_xyzw_yaw_from_ori(robot_ori)
    angle_diff = float(target_yaw_rad) - float(robot_yaw)
    angle_diff = float(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))
    if abs(angle_diff) < 0.05:
        return np.array([0.0, 0.0])
    angular_vel = np.clip(angle_diff * 1.4, -max_ang_vel, max_ang_vel)
    return np.array([0.0, angular_vel])


def make_full_nav_action(robot, base_action_2d):
    """
    Pad base [lin, ang] to full env.step action (zeros elsewhere).
    Turtlebot (dim=2) unchanged; mobile manipulators use base in first two dims.
    """
    base = np.asarray(base_action_2d, dtype=np.float32)
    full_dim = robot.action_dim
    if full_dim == 2:
        return base
    action = np.zeros(full_dim, dtype=np.float32)
    action[0:2] = base
    return action


# ──── Kinematic base (optional) ────
# Bypass wheel physics: integrate [lin_vel, ang_vel] and set_pose each substep.
# Mobile manipulators are hard to tune for slip; kinematic base is stable for sim nav.
_USE_KINEMATIC_BASE = True   # False: use wheel physics via env.step
_KINEMATIC_DT = 1.0 / 30.0  # Match default action_frequency


def _quat_xyzw_yaw_from_ori(ori) -> float:
    """
    Yaw from quaternion [qx,qy,qz,qw]; supports Tensor or numpy.
    For numpy ori, avoid torch quat2euler (@torch_compile) to dodge dynamo issues.
    """
    if th.is_tensor(ori):
        t = ori.detach().cpu().float().reshape(4)
        e = quat2euler(t)
        y = e[2]
        return float(y.detach().cpu().reshape(-1)[0].item())
    v = np.asarray(ori, dtype=np.float64).reshape(-1)
    if v.size != 4:
        return 0.0
    qx, qy, qz, qw = float(v[0]), float(v[1]), float(v[2]), float(v[3])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    yaw = math.atan2(siny_cosp, cosy_cosp)
    yaw = yaw % (2.0 * math.pi)
    if yaw > math.pi:
        yaw -= 2.0 * math.pi
    return float(yaw)


def _euler2quat_np(rpy):
    """numpy euler (r,p,y) -> quat (x,y,z,w); no torch overhead."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)


def _nav_base_roll_pitch(ori) -> tuple:
    """Roll/pitch (rad) from base quaternion xyzw; (0,0) on any parse failure."""
    if th.is_tensor(ori):
        ori = ori.detach().cpu().numpy()
    v = np.asarray(ori, dtype=np.float64).reshape(-1)
    if v.size != 4:
        return 0.0, 0.0
    qx, qy, qz, qw = float(v[0]), float(v[1]), float(v[2]), float(v[3])
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    return float(roll), float(pitch)


_anti_topple_warned: bool = False


def _nav_force_robot_upright(robot) -> bool:
    """If base roll/pitch exceeds NAV_ANTI_TOPPLE_TILT_RAD, rewrite ori to yaw-only and zero root velocities.

    Returns True if a correction was applied. The robot's z position is preserved so
    we do not teleport vertically (kinematic_base_step likewise keeps z).
    """
    global _anti_topple_warned
    if not _NAV_ANTI_TOPPLE:
        return False
    try:
        pos, ori = robot.get_position_orientation()
    except Exception:
        return False
    roll, pitch = _nav_base_roll_pitch(ori)
    if max(abs(roll), abs(pitch)) < _NAV_ANTI_TOPPLE_TILT_RAD:
        return False
    yaw = _quat_xyzw_yaw_from_ori(ori)
    new_ori = _euler2quat_np(np.array([0.0, 0.0, yaw]))
    pos_np = pos.detach().cpu().numpy() if th.is_tensor(pos) else np.asarray(pos)
    pos_np = np.asarray(pos_np, dtype=np.float64).reshape(-1)
    if pos_np.size < 3:
        return False
    try:
        robot.set_position_orientation(
            position=th.tensor(pos_np[:3].astype(np.float32), dtype=th.float32),
            orientation=th.tensor(new_ori.astype(np.float32), dtype=th.float32),
        )
    except Exception:
        return False
    try:
        root = getattr(robot, "root_link", None)
        if root is not None:
            zero3 = th.zeros(3, dtype=th.float32)
            for setter in ("set_linear_velocity", "set_angular_velocity"):
                fn = getattr(root, setter, None)
                if callable(fn):
                    try:
                        fn(zero3)
                    except Exception:
                        pass
    except Exception:
        pass
    if not _anti_topple_warned:
        _anti_topple_warned = True
        print(
            f"⚠️  [Anti-topple] base tilted roll {math.degrees(roll):+.1f}° pitch {math.degrees(pitch):+.1f}° "
            f"(> {math.degrees(_NAV_ANTI_TOPPLE_TILT_RAD):.0f}°) → forced upright; further events silent",
            flush=True,
        )
    return True


_ANTI_TOPPLE_INSTALLED: bool = False


def install_global_anti_topple(robot) -> bool:
    """Monkey-patch og.sim.step / og.sim.step_physics to force-rewrite the base orientation
    to yaw-only after each physics step. Catches every step path (nav, pick_place inner
    `og.sim.step()` loops, API main loop, post-spawn settle, etc.) so a bumped door/wall
    cannot accumulate roll/pitch into a topple.

    Idempotent: first call installs, subsequent calls are no-ops.
    """
    global _ANTI_TOPPLE_INSTALLED
    if _ANTI_TOPPLE_INSTALLED:
        return False
    if not _NAV_ANTI_TOPPLE:
        return False
    sim = getattr(og, "sim", None)
    if sim is None:
        return False
    _bound_robot = robot

    def _wrap(method_name: str) -> None:
        orig = getattr(sim, method_name, None)
        if not callable(orig):
            return

        def _wrapped(*args, **kwargs):
            ret = orig(*args, **kwargs)
            try:
                _nav_force_robot_upright(_bound_robot)
            except Exception:
                pass
            return ret

        try:
            setattr(sim, method_name, _wrapped)
        except Exception:
            pass

    _wrap("step")
    _wrap("step_physics")
    _ANTI_TOPPLE_INSTALLED = True
    print(
        f"🛡️  [Anti-topple] global hook installed on og.sim.step (tilt> "
        f"{math.degrees(_NAV_ANTI_TOPPLE_TILT_RAD):.0f}° → yaw-only + zero base velocity)",
        flush=True,
    )
    return True


def _nav_impact_emergency_brake(env, robot, after_step_callback=None) -> bool:
    """If currently in non-ground contact, emit N zero-cmd substeps to bleed off momentum.

    Each brake step also force-rewrites the base orientation to yaw-only.
    Returns True if a brake burst ran. Cheap when no contact (single ContactBodies check).
    """
    if _NAV_IMPACT_BRAKE_STEPS <= 0:
        return False
    if not _nav_robot_in_non_ground_contact(robot):
        return False
    zero = np.array([0.0, 0.0], dtype=np.float64)
    for _ in range(_NAV_IMPACT_BRAKE_STEPS):
        if _USE_KINEMATIC_BASE and robot.action_dim > 2:
            kinematic_base_step(env, robot, zero, after_step_callback)
        else:
            action = make_full_nav_action(robot, zero)
            try:
                env.step(action)
            except Exception:
                break
            if after_step_callback:
                after_step_callback()
        _nav_force_robot_upright(robot)
    return True


def kinematic_base_step(env, robot, base_action_2d, after_step_callback=None):
    """
    Kinematic base: integrate [lin_vel, ang_vel] (SI), set_position_orientation, then advance sim.
    Returns the same 5-tuple shape as env.step.
    """
    lin_vel = float(base_action_2d[0])
    ang_vel = float(base_action_2d[1])
    dt = _KINEMATIC_DT

    pos, ori = robot.get_position_orientation()
    x = float(pos[0])
    y = float(pos[1])
    z = float(pos[2])
    yaw = _quat_xyzw_yaw_from_ori(ori)

    mid_yaw = yaw + ang_vel * dt * 0.5
    new_x = x + lin_vel * math.cos(mid_yaw) * dt
    new_y = y + lin_vel * math.sin(mid_yaw) * dt
    new_yaw = yaw + ang_vel * dt

    new_ori = _euler2quat_np(np.array([0.0, 0.0, new_yaw]))
    robot.set_position_orientation(
        position=th.tensor([new_x, new_y, z], dtype=th.float32),
        orientation=th.tensor(new_ori, dtype=th.float32),
    )

    # Avoid env.step(zero): articulated robots may read all DOFs; USD/tensor mismatch can
    # trigger "Failed to get DOF positions". Pose already set; only sim + obs needed.
    zero_action = np.zeros(robot.action_dim, dtype=np.float32)
    action_t = env._convert_action_to_tensor(zero_action)
    og.sim.step()
    obs, reward, terminated, truncated, info = env._post_step(action_t)
    if after_step_callback:
        after_step_callback()
    # og.sim.step() can still resolve residual contact impulses on upper joints; if that tilts
    # the base, snap back to yaw-only so a bumped door cannot accumulate into a topple.
    _nav_force_robot_upright(robot)
    return obs, reward, terminated, truncated, info


def nav_step(env, robot, base_action_2d, after_step_callback=None):
    """
    Single nav substep: kinematic base if enabled, else full env.step.
    base_action_2d: [lin_vel, ang_vel] in SI units. Returns env.step 5-tuple.

    Defensive post-step:
      - force base upright (yaw-only) if it tilted (e.g. bumped a door)
      - if currently in non-ground contact, run a short zero-cmd brake burst so the next
        command does not keep pushing into the obstacle.
    """
    if _USE_KINEMATIC_BASE and robot.action_dim > 2:
        ret = kinematic_base_step(env, robot, base_action_2d, after_step_callback)
    else:
        action = make_full_nav_action(robot, base_action_2d)
        obs, reward, terminated, truncated, info = env.step(action)
        if after_step_callback:
            after_step_callback()
        ret = (obs, reward, terminated, truncated, info)
    _nav_force_robot_upright(robot)
    _nav_impact_emergency_brake(env, robot, after_step_callback)
    return ret


def init_navigation_state(env, robot, scene, floor, goal_pos, goal_yaw_rad=None):
    """
    Build resumable nav state for navigate_step; None if no path.
    Stepwise use: call navigate_step(max_steps_per_call) each tick so the main loop stays responsive.
    """
    robot_pos_ori = robot.get_position_orientation()
    start_pos = robot_pos_ori[0][:2]
    path_waypoints, geodesic_distance = scene.get_shortest_path(
        floor=floor,
        source_world=start_pos,
        target_world=goal_pos[:2],
        entire_path=True,
        robot=None,
    )
    if path_waypoints is None:
        return None
    pl = _normalize_path_waypoints_list(path_waypoints)
    if not pl:
        return None
    path_for_viz = [np.asarray(_as_xy2(w), dtype=np.float64).copy() for w in pl]
    target_pos_np = goal_pos[:2].cpu().numpy() if th.is_tensor(goal_pos) else np.asarray(goal_pos[:2], dtype=np.float64)
    start_xy = start_pos.cpu().numpy() if th.is_tensor(start_pos) else np.asarray(start_pos, dtype=np.float64)
    start_xy = np.asarray(start_xy, dtype=np.float64).reshape(-1)[:2]
    wp0 = _initial_path_waypoint_idx(start_xy, path_waypoints)
    return {
        "path_waypoints": path_waypoints,
        "path_for_viz": path_for_viz,
        "current_waypoint_idx": wp0,
        "step_count": 0,
        "phase": "waypoints",
        "target_pos_np": target_pos_np,
        "goal_yaw_rad": goal_yaw_rad,
        "max_steps": 5000,
        "arrival_threshold": 0.1,
        "rotate_count": 0,
        "max_rotate_steps": 500,
        "printed_init": False,
    }


def _door_nav_stuck_should_fail(state, robot_xy_np) -> bool:
    """
    Near doors, collision noise may prevent tiny-threshold convergence → apparent stall.
    After enough steps, if displacement in repeated windows is ~0, fail instead of in_progress forever.
    """
    cfg_min = state.get("door_nav_stuck_min_steps")
    if cfg_min is None:
        return False
    if state["step_count"] < int(cfg_min):
        return False
    w = int(state.get("door_nav_stuck_window", 50))
    if w <= 0 or state["step_count"] % w != 0:
        return False
    rp = np.asarray(robot_xy_np, dtype=np.float64).reshape(-1)[:2]
    eps = float(state.get("door_nav_stuck_eps_m", 0.02))
    hits_need = int(state.get("door_nav_stuck_hits_needed", 3))
    prev = state.get("_door_stuck_prev_xy")
    if prev is None:
        state["_door_stuck_prev_xy"] = rp.copy()
        state["_door_stuck_hits"] = 0
        return False
    prev = np.asarray(prev, dtype=np.float64).reshape(-1)[:2]
    d = float(np.linalg.norm(rp - prev))
    state["_door_stuck_prev_xy"] = rp.copy()
    if d < eps:
        state["_door_stuck_hits"] = int(state.get("_door_stuck_hits", 0)) + 1
    else:
        state["_door_stuck_hits"] = 0
    if state["_door_stuck_hits"] >= hits_need:
        print(
            f"❌ Nav appears stuck (~{state['step_count']} steps, displacement < {eps}m); "
            f"often at door frames; ending this nav segment."
        )
        return True
    return False


def _door_nav_final_no_progress_should_fail(state, dist_to_goal: float) -> bool:
    """
    Final-approach only: along waypoints, distance to an intermediate vertex can go up then down,
    so no-progress on vertices would false-positive long paths.

    Track best distance to goal; large regression resets baseline; long plateau near goal → fail.
    """
    cfg_min = state.get("door_nav_stuck_min_steps")
    if cfg_min is None:
        return False
    start_after = int(state.get("door_nav_no_progress_min_steps", 120))
    if state["step_count"] < start_after:
        return False
    max_stall = int(state.get("door_nav_no_progress_max_stall", 400))
    improve_m = float(state.get("door_nav_progress_improve_m", 0.03))
    regress_m = float(state.get("door_nav_progress_regress_m", 0.12))
    arrival = float(state.get("arrival_threshold", 0.1))
    d = float(dist_to_goal)

    if "_door_final_best_d" not in state:
        state["_door_final_best_d"] = d
        state["_door_final_prog_stall"] = 0
        return False

    best = float(state.get("_door_final_best_d", d))
    if d < best - improve_m:
        state["_door_final_best_d"] = d
        state["_door_final_prog_stall"] = 0
    elif d > best + regress_m:
        # Pushed back or big detour: reset baseline so stall counter does not stick
        state["_door_final_best_d"] = d
        state["_door_final_prog_stall"] = 0
    else:
        state["_door_final_prog_stall"] = int(state.get("_door_final_prog_stall", 0)) + 1

    if state["_door_final_prog_stall"] < max_stall:
        return False
    if d <= arrival * 1.25:
        state["_door_final_prog_stall"] = 0
        return False
    print(
        f"❌ Final approach stalled (≥{max_stall} steps, dist {d:.2f}m, arrival ~{arrival:.2f}m); ending nav segment."
    )
    return True


def navigate_step(env, robot, scene, floor, nav_state, visualizer, max_steps_per_call=30, after_step_callback=None):
    """
    Resumable nav chunk (up to max_steps_per_call substeps).
    Returns ("in_progress"|"reached"|"failed", nav_state).
    after_step_callback: run after each env substep (e.g. held object), same as navigation_with_pick_place.
    """
    if nav_state is None:
        return "failed", None
    state = dict(nav_state)
    step_limit = state["step_count"] + max_steps_per_call
    stop_action = make_full_nav_action(robot, [0.0, 0.0])
    wp_reach = float(state.get("waypoint_reach_eps", 0.3))

    if not state.get("printed_init"):
        print("\nA* path ready. Stepped nav (other commands can be handled between chunks).")
        state["printed_init"] = True

    # Phase 1: follow polyline waypoints
    while state["phase"] == "waypoints" and state["step_count"] < step_limit and state["step_count"] < state["max_steps"]:
        if state["current_waypoint_idx"] >= len(state["path_waypoints"]):
            state["phase"] = "final"
            break
        wp_idx_before = int(state["current_waypoint_idx"])
        current_waypoint = state["path_waypoints"][wp_idx_before]
        current_waypoint_np = current_waypoint.cpu().numpy() if th.is_tensor(current_waypoint) else np.asarray(current_waypoint)
        rp0 = robot.get_position_orientation()[0][:2]
        rp0_np = rp0.cpu().numpy() if th.is_tensor(rp0) else np.asarray(rp0)
        carrot = _pure_pursuit_carrot(
            rp0_np, state["path_waypoints"], wp_idx_before, _NAV_LOOKAHEAD_M
        )
        track_xy = carrot if carrot is not None else _as_xy2(current_waypoint_np)
        base_cmd = compute_action_to_waypoint(robot, track_xy)
        nav_step(env, robot, base_cmd, after_step_callback)
        robot_pos_after = robot.get_position_orientation()[0][:2]
        robot_pos_after_np = robot_pos_after.cpu().numpy() if th.is_tensor(robot_pos_after) else np.asarray(robot_pos_after)
        state["current_waypoint_idx"] = _advance_waypoint_idx_on_path(
            robot_pos_after_np, state["path_waypoints"], state["current_waypoint_idx"], wp_reach
        )
        state["step_count"] += 1
        if _door_nav_stuck_should_fail(state, robot_pos_after_np):
            for _ in range(10):
                nav_step(env, robot, [0.0, 0.0], after_step_callback)
            return "failed", state
        if visualizer is not None and state["step_count"] % 5 == 0:
            visualizer.update(
                robot_pos=robot_pos_after_np,
                goal_pos=state["target_pos_np"],
                path=state["path_for_viz"],
                current_waypoint=current_waypoint_np,
            )

    # Phase 2: converge to goal XY
    while state["phase"] == "final" and state["step_count"] < step_limit and state["step_count"] < state["max_steps"]:
        robot_pos = robot.get_position_orientation()[0][:2]
        robot_pos_np = robot_pos.cpu().numpy() if th.is_tensor(robot_pos) else np.asarray(robot_pos)
        distance_to_goal = np.linalg.norm(robot_pos_np - state["target_pos_np"])
        if distance_to_goal < state["arrival_threshold"]:
            state["phase"] = "rotate" if state.get("goal_yaw_rad") is not None else "stop"
            break
        base_cmd = compute_action_to_waypoint(robot, state["target_pos_np"])
        nav_step(env, robot, base_cmd, after_step_callback)
        state["step_count"] += 1
        robot_pos_after = robot.get_position_orientation()[0][:2]
        robot_pos_after_np = robot_pos_after.cpu().numpy() if th.is_tensor(robot_pos_after) else np.asarray(robot_pos_after)
        distance_after = float(np.linalg.norm(robot_pos_after_np - state["target_pos_np"]))
        if _door_nav_final_no_progress_should_fail(state, distance_after):
            for _ in range(10):
                nav_step(env, robot, [0.0, 0.0], after_step_callback)
            return "failed", state
        if _door_nav_stuck_should_fail(state, robot_pos_after_np):
            for _ in range(10):
                nav_step(env, robot, [0.0, 0.0], after_step_callback)
            return "failed", state
        if visualizer is not None and state["step_count"] % 5 == 0:
            visualizer.update(
                robot_pos=robot_pos_after_np,
                goal_pos=state["target_pos_np"],
                path=state["path_for_viz"],
                current_waypoint=None,
            )

    # Phase 3: rotate in place to goal yaw
    while state["phase"] == "rotate" and state["rotate_count"] < state["max_rotate_steps"]:
        base_rot = compute_rotate_to_yaw_action(robot, state["goal_yaw_rad"])
        if np.abs(base_rot[1]) < 1e-6:
            state["phase"] = "stop"
            break
        nav_step(env, robot, base_rot, after_step_callback)
        state["rotate_count"] += 1
        if visualizer is not None and state["rotate_count"] % 10 == 0:
            rp = robot.get_position_orientation()[0][:2]
            rp = rp.cpu().numpy() if th.is_tensor(rp) else np.asarray(rp)
            visualizer.update(robot_pos=rp, goal_pos=state["target_pos_np"], path=state["path_for_viz"], current_waypoint=None)

    # Phase 4: stop and done
    if state["phase"] == "stop" or (state["phase"] == "rotate" and state["rotate_count"] >= state["max_rotate_steps"]):
        state["phase"] = "stop"
        for _ in range(10):
            nav_step(env, robot, [0.0, 0.0], after_step_callback)
        state["phase"] = "done"
        return "reached", state

    if state["step_count"] >= state["max_steps"]:
        for _ in range(10):
            nav_step(env, robot, [0.0, 0.0], after_step_callback)
        print(f"❌ Nav timeout (max_steps={state['max_steps']})")
        return "failed", state

    return "in_progress", state


def navigate_to_goal(env, robot, scene, floor, goal_pos, visualizer, goal_yaw_rad=None):
    """Navigate to goal; if goal_yaw_rad (rad) is set, rotate in place after XY arrival. Blocking CLI helper."""
    print("\nPlanning A* path...")
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
        print("❌ No path found. Try opening doors...")
        # Still draw goal on map for debugging
        target_pos_np = goal_pos[:2].cpu().numpy() if th.is_tensor(goal_pos) else np.asarray(goal_pos[:2], dtype=np.float64)
        if visualizer is not None:
            robot_pos_2d = robot.get_position_orientation()[0][:2]
            if th.is_tensor(robot_pos_2d):
                robot_pos_2d = robot_pos_2d.cpu().numpy()
            visualizer.update(
                robot_pos=robot_pos_2d,
                goal_pos=target_pos_np,
                path=None,
                current_waypoint=None,
            )
            print(f"   Goal marked on map: [{target_pos_np[0]:.2f}, {target_pos_np[1]:.2f}] (red star)")
        return False
    
    print(f"✅ Path found. Length: {geodesic_distance:.2f}m, waypoints: {len(path_waypoints)}")
    
    print("\nNavigating (pure pursuit on A* polyline, then final XY converge)...")
    max_steps = 5000
    step_count = 0
    
    pl = _normalize_path_waypoints_list(path_waypoints)
    path_for_viz = [np.asarray(_as_xy2(w), dtype=np.float64).copy() for w in pl] if pl else []
    n_wp = len(pl)
    
    target_pos_np = goal_pos[:2].cpu().numpy() if th.is_tensor(goal_pos) else goal_pos[:2]
    arrival_threshold = 0.1  # meters
    
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
        base_cmd = compute_action_to_waypoint(robot, track_xy)
        nav_step(env, robot, base_cmd)
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
            print(f"Step {step_count}, dist to goal {dg:.2f}m, waypoint {current_waypoint_idx}/{n_wp}")
    
    if current_waypoint_idx >= n_wp and step_count < max_steps:
        print("✅ All polyline vertices passed; driving to final goal...")
        while step_count < max_steps:
            robot_pos = robot.get_position_orientation()[0][:2]
            robot_pos_np = robot_pos.cpu().numpy() if th.is_tensor(robot_pos) else np.asarray(robot_pos)
            distance_to_goal = np.linalg.norm(robot_pos_np - target_pos_np)
            if distance_to_goal < arrival_threshold:
                break
            base_cmd = compute_action_to_waypoint(robot, target_pos_np)
            nav_step(env, robot, base_cmd)
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
                print(f"Step {step_count}, dist to final goal {dgc:.2f}m")
    
    for _ in range(10):
        nav_step(env, robot, [0.0, 0.0])
    
    final_pos = robot.get_position_orientation()[0][:2]
    if th.is_tensor(final_pos):
        final_pos = final_pos.cpu().numpy()
    final_distance = np.linalg.norm(final_pos - target_pos_np)
    
    if final_distance >= arrival_threshold:
        print(f"❌ Did not reach goal. Final distance: {final_distance:.2f}m, steps: {step_count}")
        return False

    if goal_yaw_rad is not None and step_count < max_steps:
        yaw_threshold_rad = 0.05
        max_rotate_steps = 500
        rotate_count = 0
        print(f"🔄 At goal XY, rotating to heading ({math.degrees(goal_yaw_rad):.1f}°)...")
        while rotate_count < max_rotate_steps:
            base_rot = compute_rotate_to_yaw_action(robot, goal_yaw_rad)
            if np.abs(base_rot[1]) < 1e-6:
                break
            nav_step(env, robot, base_rot)
            rotate_count += 1
            if visualizer is not None and rotate_count % 10 == 0:
                robot_pos_after = robot.get_position_orientation()[0][:2]
                if th.is_tensor(robot_pos_after):
                    robot_pos_after = robot_pos_after.cpu().numpy()
                visualizer.update(
                    robot_pos=robot_pos_after,
                    goal_pos=target_pos_np,
                    path=path_for_viz,
                    current_waypoint=None,
                )
        for _ in range(10):
            nav_step(env, robot, [0.0, 0.0])
        final_ori = robot.get_position_orientation()[1]
        final_yaw = _quat_xyzw_yaw_from_ori(final_ori)
        print(f"✅ Final yaw: {math.degrees(final_yaw):.1f}° (target {math.degrees(goal_yaw_rad):.1f}°)")

    print(f"✅ Reached goal. Final distance: {final_distance:.2f}m, steps: {step_count}")
    return True


def set_camera_resolution(robot, image_width=256, image_height=256):
    """
    Set resolution on all VisionSensor instances on the robot.

    Args:
        robot: Robot instance
        image_width: Image width in pixels (default 256)
        image_height: Image height in pixels (default 256)
    """
    from omnigibson.sensors import VisionSensor
    
    print(f"📷 Setting camera resolution to {image_width}x{image_height}...")
    modified_count = 0
    
    for sensor_name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor):
            sensor.image_width = image_width
            sensor.image_height = image_height
            modified_count += 1
            print(f"  ✅ Sensor '{sensor_name}': resolution {image_width}x{image_height}")
    
    if modified_count > 0:
        robot.env.load_observation_space()
        print(f"✅ Updated {modified_count} VisionSensor(s)")
    else:
        print("⚠️  No VisionSensor found on robot")


def main():
    """Interactive nav + doors demo entry point."""
    CAMERA_WIDTH = 512
    CAMERA_HEIGHT = 512
    
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Beechwood_0_int",
            "load_object_categories": None,  # None = load scene defaults (includes doors)
            "trav_map_resolution": 0.05,
            "default_erosion_radius": 0.3,
        },
        "robots": [
            {
                "type": "Turtlebot",
                "obs_modalities": ["scan", "rgb"],
                "action_type": "continuous",
                "action_normalize": True,
                # Option A: set camera res in config (preferred)
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
    
    # Option B: set_camera_resolution(robot, ...) if config path does not apply
    
    og.sim.enable_viewer_camera_teleoperation()
    
    env.reset()
    print(f"Robot initial position: {robot.get_position_orientation()[0]}")
    
    floor = 0
    
    # Snapshot pre-erosion map (avoid stacking erosions)
    original_map_dict = {}
    original_map_dict[floor] = scene.trav_map.floor_map[floor].clone()
    
    erosion_radius_meters = 0.4
    radius_pixel = int(math.ceil(erosion_radius_meters / scene.trav_map.map_resolution))
    kernel = np.ones((radius_pixel, radius_pixel), dtype=np.uint8)
    trav_map_eroded = cv2.erode(original_map_dict[floor].cpu().numpy(), kernel)
    scene.trav_map.floor_map[floor] = th.tensor(trav_map_eroded)
    
    print(f"✅ Map eroded: {erosion_radius_meters}m ≈ {radius_pixel} px kernel")
    
    doors = list_all_doors(scene)
    print_doors(doors)
    
    door_joint_limits = {}
    door_states = {}  # {door_name: "open"|"close"}
    door_map_regions = {}  # {door_name: (min_r, max_r, min_c, max_c)}
    
    # Per-door scale for footprint bbox on grid (<1 shrink, >1 expand)
    door_shrink_factors = {
        3: 1.2,
        4: 1.2,
        8: 0.5,  # shrink footprint so we do not free too many cells near door 8
    }
    
    print("\n📍 Initializing door footprints on trav map...")
    for idx, door in enumerate(doors, 1):
        shrink_factor = door_shrink_factors.get(idx, 1.0)
        
        door_region = get_door_map_region(door, scene.trav_map, floor, shrink_factor=shrink_factor)
        if door_region is not None:
            door_map_regions[door.name] = door_region
            shrink_info = f" (shrink factor {shrink_factor})" if shrink_factor < 1.0 else ""
            print(f"  ✅ Door #{idx} ({door.name}): cells [{door_region[0]}:{door_region[1]}, {door_region[2]}:{door_region[3]}]{shrink_info}")
        else:
            print(f"  ⚠️  Door #{idx} ({door.name}): could not get map region")
    
    skip_doors = {7}  # door 7: known physics/NaN issues
    
    # Per-door hinge targets in degrees: {"open": deg, "close": deg}; negative = opposite swing
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
    
    print(f"\n📐 Door angle presets (deg):")
    for door_idx, positions in door_positions.items():
        print(f"  Door #{door_idx}: close={positions['close']}°, open={positions['open']}°")
    print(f"  Other doors: default open/close behavior")
    
    if skip_doors:
        print(f"\n⚠️  Skipped doors (known issues): {', '.join(f'#{d}' for d in skip_doors)}")
    
    print("\nCreating map visualizer...")
    visualizer = MapVisualizer(
        trav_map_obj=scene.trav_map,
        floor=floor,
        doors=doors,
        door_states=door_states
    )
    print("✅ Map window open (red=closed door, green=open door)")
    
    print("\n" + "="*80)
    print("🚀 Nav + doors (sliding-door style demo)")
    print("="*80)
    print("📖 Commands:")
    print("  - 'nav' — click goal on map, drag to set heading (release to finish, RViz-like)")
    print("  - 'door <index> <open|close>' — single door")
    print("  - 'all open' / 'all close' — all doors")
    print("  - 'list' — print door states")
    print("  - 'quit' / 'exit' — exit")
    print("  - WASD — move viewer camera")
    print("="*80)
    
    print("\n⌨️  Type a command and press Enter...")
    command_buffer = ""
    
    try:
        with NonBlockingInput() as nbi:
            while True:
                og.sim.step()
                
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
                            
                            if user_input.startswith('all'):
                                parts = user_input.split()
                                if len(parts) != 2:
                                    print("❌ Invalid format. Use: all <open|close>")
                                    print("\n⌨️  Type a command and press Enter...")
                                    continue
                                
                                action = parts[1]
                                if action not in ['open', 'close']:
                                    print("❌ Action must be 'open' or 'close'")
                                    print("\n⌨️  Type a command and press Enter...")
                                    continue
                                
                                action_str = "opening" if action == "open" else "closing"
                                print(f"\n🚪 {action_str.capitalize()} all doors...")
                                success_count = 0
                                skip_count = 0
                                
                                for idx, door in enumerate(doors, 1):
                                    if idx in skip_doors:
                                        print(f"  ⏭️  Skipping door #{idx} ({door.name})")
                                        skip_count += 1
                                        continue
                                    
                                    print(f"\nDoor #{idx} ({door.name})...")
                                    if control_door(door, idx, action, scene, floor, visualizer, original_map_dict, door_joint_limits, door_positions, door_states, doors=doors, door_map_regions=door_map_regions):
                                        success_count += 1
                                
                                denom = max(1, len(doors) - skip_count)
                                print(f"\n✅ Done. Set {success_count}/{denom} doors to '{action}'.")
                                if skip_count > 0:
                                    print(f"⏭️  Skipped {skip_count} problematic door(s)")
                                
                                print("\n⌨️  Type a command and press Enter...")
                                continue
                            
                            if user_input.startswith('door'):
                                parts = user_input.split()
                                if len(parts) != 3:
                                    print("❌ Invalid format. Use: door <index> <open|close>")
                                    print("\n⌨️  Type a command and press Enter...")
                                    continue
                                
                                try:
                                    door_idx = int(parts[1]) - 1
                                    action = parts[2]
                                    
                                    if door_idx < 0 or door_idx >= len(doors):
                                        print(f"❌ Door index must be 1..{len(doors)}")
                                    elif action not in ['open', 'close']:
                                        print("❌ Action must be 'open' or 'close'")
                                    elif (door_idx + 1) in skip_doors:
                                        print(f"⚠️  Door #{door_idx + 1} ({doors[door_idx].name}) is skipped (known issue)")
                                    else:
                                        control_door(doors[door_idx], door_idx + 1, action, scene, floor, visualizer, original_map_dict, door_joint_limits, door_positions, door_states, doors=doors, door_map_regions=door_map_regions)
                                except ValueError:
                                    print("❌ Door index must be an integer")
                                
                                print("\n⌨️  Type a command and press Enter...")
                                continue
                            
                            print("❌ Unknown command. Use 'nav', 'door', 'all', 'list', or 'quit'")
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
    
    print("\nClosing map visualizer...")
    visualizer.close()
    
    print("Shutting down simulator...")
    og.shutdown()
    print("✅ Done.")


if __name__ == "__main__":
    main()
