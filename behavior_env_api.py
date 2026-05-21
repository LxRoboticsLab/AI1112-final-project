"""
HTTP API for the behavior environment.
Used by the MCP door-robot server: robot navigation, camera, Pick&Place, and state queries (doors read-only; opened at sim startup).
Interacts with the navigation_with_doors main loop via queues; door and nav actions run on the main thread.
"""

import os
import socket
import sys
import math
import threading
import time
import queue
import json
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# Read by main thread: stores exception text if the API thread fails to start
_server_start_error: Optional[str] = None
_server_start_error_lock = threading.Lock()

# Prefer Flask (no asyncio) to avoid uvicorn auto_loop_factory conflicts under OmniGibson/Isaac Sim
_HAS_FLASK = False
_HAS_FASTAPI = False
app = None

try:
    from flask import Flask, request, jsonify
    from werkzeug.serving import make_server
    _flask_app = Flask("Behavior Env API")
    _HAS_FLASK = True
    app = _flask_app
except ImportError:
    _flask_app = None
    try:
        from fastapi import FastAPI, Body
        from fastapi.responses import JSONResponse
        import uvicorn
        _HAS_FASTAPI = True
        app = FastAPI(title="Behavior Env API", version="0.1.0")
    except ImportError:
        pass

# ============================================================================
# Command queue (refactored)
# Design: single queue + single lock, no pushback.
#   - Normal commands: FIFO append (including pick_place, dequeued when idle in main step)
#   - Cancels: appendleft (highest priority, handled first)
#   - Preempt channel: pick_place retry/preempt uses a separate slot, handled before FIFO when busy
#   - Main loop order: cancel / preempt > FIFO when idle
# ============================================================================
_command_queue: deque = deque()
_command_queue_lock = threading.Lock()

# Preempt channel: while pick_place is running, same-object retry goes here instead of the queue
_preempt_pick_place: Optional[dict] = None
_preempt_pp_lock = threading.Lock()


def _enqueue_command(cmd: tuple) -> None:
    with _command_queue_lock:
        _command_queue.append(cmd)


def _count_queued_pick_place() -> int:
    """Number of pick_place commands in the queue not yet consumed by the main loop."""
    with _command_queue_lock:
        return sum(1 for c in _command_queue if c and c[0] == "pick_place")


def _enqueue_pick_place(object_name: str, target_name: str) -> Tuple[bool, str]:
    """Enqueue pick_place. Returns (ok, error_msg).
    - Same object while running -> preempt channel (LOST retry)
    - running/submitted and different object -> reject (client plan queue avoids overlapping sim jobs)
    - Otherwise -> FIFO enqueue
    """
    global _preempt_pick_place
    payload = {"object_name": object_name, "target_name": target_name}
    with _single_pick_place_lock:
        st = _single_pick_place_state["status"]
        currently_running = st == "running"
        submitted = st == "submitted"
        running_obj = _single_pick_place_state.get("object_name") if currently_running else None
    is_retry = currently_running and running_obj == object_name
    if is_retry:
        with _preempt_pp_lock:
            _preempt_pick_place = payload
        return True, ""
    busy = currently_running or submitted
    if busy:
        return (
            False,
            "PICK_PLACE_BUSY: A Pick&Place is already running or queued; cannot submit another object to the sim. "
            "Use the client plan queue for ordering, or wait until the current task finishes.",
        )
    with _command_queue_lock:
        _command_queue.append(("pick_place", payload))
    with _single_pick_place_lock:
        st2 = _single_pick_place_state["status"]
        if st2 in ("idle", "done"):
            _single_pick_place_state["status"] = "submitted"
            _single_pick_place_state["success"] = None
            _single_pick_place_state["error"] = None
            _single_pick_place_state["phase"] = None
    return True, ""


def _enqueue_cancel(cmd: tuple) -> None:
    """cancel_pick_place / cancel_current_task: prepend to queue head."""
    with _command_queue_lock:
        _command_queue.appendleft(cmd)


def clear_command_queue() -> None:
    """Clear pending commands (on scene reset; avoids stale navigate/pick after reset)."""
    with _command_queue_lock:
        _command_queue.clear()
_result_queue: queue.Queue = queue.Queue()
_state: dict = {
    "doors": [],
    "door_states": {},
    "robot_position": None,
    "objects_graspable": [],
    "objects_placement": [],
    "grasp": {"holding": False, "object_name": None, "grasp_attempt_index": 0},
    "door_world_xy": [],  # [{"id":1,"x":..,"y":..}, ...] for list_doors robot distance
    "skip_door_ids": [],  # 1-based door IDs disabled in sim; list_doors sets controllable=false
}
_state_lock = threading.Lock()

# Single navigation (async): main loop steps; client polls GET /api/robot/navigate_status
_single_nav_lock = threading.Lock()
_single_nav_state: Dict[str, Any] = {"status": "idle", "success": None, "error": None}  # idle | navigating | done


def set_single_nav_result(success: bool, error: Optional[str] = None):
    """Called by main loop when a navigation segment completes."""
    with _single_nav_lock:
        _single_nav_state["status"] = "done"
        _single_nav_state["success"] = success
        _single_nav_state["error"] = error or ""


def get_single_nav_state() -> dict:
    """API: poll single-navigation state."""
    with _single_nav_lock:
        return dict(_single_nav_state)


def set_single_nav_navigating():
    """Called by main loop when starting a navigation segment."""
    with _single_nav_lock:
        _single_nav_state["status"] = "navigating"
        _single_nav_state["success"] = None
        _single_nav_state["error"] = None


# Single Pick&Place (async): like navigate; returns immediately; client polls GET /api/robot/pick_place_status
_single_pick_place_lock = threading.Lock()
_single_pick_place_state: Dict[str, Any] = {
    "status": "idle", "success": None, "error": None,
    "phase": None,  # nav_to_obj | attach | nav_to_target | place | None
    "object_name": None, "target_name": None,
}

# Transport vision: client posts judgment=hold|lost; consecutive HOLD required before place; LOST resets streak
_transport_visual_ack_lock = threading.Lock()
_transport_visual_ack_count: int = 0
_transport_visual_hold_streak: int = 0


def pick_place_transport_visual_gating_enabled() -> bool:
    return os.environ.get("PICK_PLACE_REQUIRE_TRANSPORT_VISUAL", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def get_pick_place_min_transport_visual_frames() -> int:
    """Consecutive HOLD frames required before place (not mixed HOLD/LOST counting)."""
    try:
        n = int(os.environ.get("PICK_PLACE_MIN_TRANSPORT_HOLD_FRAMES", ""))
    except ValueError:
        n = -1
    if n < 0:
        try:
            n = int(os.environ.get("PICK_PLACE_MIN_TRANSPORT_VISUAL_FRAMES", "3"))
        except ValueError:
            n = 3
    return max(0, n)


def reset_transport_visual_ack_count() -> None:
    global _transport_visual_ack_count, _transport_visual_hold_streak
    with _transport_visual_ack_lock:
        _transport_visual_ack_count = 0
        _transport_visual_hold_streak = 0


def reset_episode_api_state() -> None:
    """After scene reset: clear single nav/Pick&Place, preempt channel, and plan index state (main thread)."""
    global _preempt_pick_place
    reset_transport_visual_ack_count()
    with _preempt_pp_lock:
        _preempt_pick_place = None
    with _single_nav_lock:
        _single_nav_state["status"] = "idle"
        _single_nav_state["success"] = None
        _single_nav_state["error"] = None
    with _single_pick_place_lock:
        _single_pick_place_state["status"] = "idle"
        _single_pick_place_state["success"] = None
        _single_pick_place_state["error"] = None
        _single_pick_place_state["phase"] = None
        _single_pick_place_state["object_name"] = None
        _single_pick_place_state["target_name"] = None
    plan_reset_to_idle_after_scene_reset()


def increment_transport_visual_ack_count() -> int:
    """Legacy client: treat as one HOLD."""
    return report_pick_place_transport_visual_judgment("hold")["transport_visual_acks"]


def report_pick_place_transport_visual_judgment(judgment: str) -> dict:
    """
    judgment: hold | lost (case-insensitive).
    hold: total +1, consecutive HOLD +1; lost: total +1, consecutive HOLD reset to 0.
    """
    global _transport_visual_ack_count, _transport_visual_hold_streak
    j = (judgment or "hold").strip().lower()
    if j not in ("hold", "lost"):
        j = "hold"
    with _transport_visual_ack_lock:
        _transport_visual_ack_count += 1
        if j == "lost":
            _transport_visual_hold_streak = 0
        else:
            _transport_visual_hold_streak += 1
        return {
            "transport_visual_acks": _transport_visual_ack_count,
            "transport_visual_hold_streak": _transport_visual_hold_streak,
        }


def get_transport_visual_ack_count() -> int:
    with _transport_visual_ack_lock:
        return _transport_visual_ack_count


def get_transport_visual_hold_streak() -> int:
    with _transport_visual_ack_lock:
        return _transport_visual_hold_streak


def set_single_pick_place_running(object_name: str = "", target_name: str = ""):
    """Called by main loop when starting a Pick&Place segment."""
    reset_transport_visual_ack_count()
    with _single_pick_place_lock:
        _single_pick_place_state["status"] = "running"
        _single_pick_place_state["success"] = None
        _single_pick_place_state["error"] = None
        _single_pick_place_state["phase"] = "nav_to_obj"
        _single_pick_place_state["object_name"] = object_name
        _single_pick_place_state["target_name"] = target_name


def set_single_pick_place_phase(phase: str):
    """Called by main loop to update current Pick&Place phase."""
    with _single_pick_place_lock:
        if _single_pick_place_state["status"] == "running":
            _single_pick_place_state["phase"] = phase


def set_single_pick_place_result(success: bool, error: Optional[str] = None):
    """Called by main loop when a Pick&Place segment completes."""
    with _single_pick_place_lock:
        _single_pick_place_state["status"] = "done"
        _single_pick_place_state["success"] = success
        _single_pick_place_state["error"] = error or ""
        _single_pick_place_state["phase"] = None


def get_single_pick_place_state() -> dict:
    """API: poll single Pick&Place state."""
    with _single_pick_place_lock:
        out = dict(_single_pick_place_state)
    out["queued_pick_place"] = _count_queued_pick_place()
    out["transport_visual_acks"] = get_transport_visual_ack_count()
    out["transport_visual_hold_streak"] = get_transport_visual_hold_streak()
    out["transport_visual_required"] = (
        get_pick_place_min_transport_visual_frames()
        if pick_place_transport_visual_gating_enabled()
        else 0
    )
    return out


# ---------- Execution plan (multi-task, sequential/parallel hints, updatable while running) ----------
# Plan shape: { "plan_id": optional, "steps": [ { "step_id": 0, "parallel": false, "tasks": [ {"type": "navigate"|"get_camera"|"pick_place", "payload": {...}, "task_id": optional} ] } ] }
# - steps run in order; tasks within a step run in order (single robot). parallel=true hints client-side parallelism with other APIs.
# - Example: pick A place B, then C place D, then nav+camera -> 3 steps. Weather+nav -> one step or split on client.
_plan_state: Dict[str, Any] = {
    "plan_id": None,
    "plan": None,           # current plan JSON
    "status": "idle",       # idle | running | completed | cancelled
    "step_index": 0,
    "task_index": 0,
    "results": [],          # [ {"step": i, "task": j, "type": str, "success": bool, ...} ]
    "external_results": [], # client-pushed virtual/external API results
    "assistant_reports": [], # plain-text lines to user: { "text", "timestamp", "task_id"|None }
    "error": None,
    "created_at": None,
}
_plan_lock = threading.Lock()


def _gen_plan_id() -> str:
    return f"plan_{uuid.uuid4().hex[:12]}"


def plan_submit(plan_dict: dict) -> str:
    """API: enqueue plan_submit on command queue; main loop executes. Returns plan_id."""
    plan_id = plan_dict.get("plan_id") or _gen_plan_id()
    steps = plan_dict.get("steps")
    if not steps or not isinstance(steps, list):
        raise ValueError("plan requires a non-empty steps array")
    normalized = {"plan_id": plan_id, "steps": steps}
    _enqueue_command(("plan_submit", normalized))
    return plan_id


def plan_update(cancel: bool = False, append_steps: Optional[list] = None, replace_plan: Optional[dict] = None):
    """API: update plan (cancel / append / replace); main loop applies on next frame."""
    payload = {}
    if cancel:
        payload["cancel"] = True
    if append_steps is not None:
        payload["append_steps"] = list(append_steps)
    if replace_plan is not None:
        payload["replace_plan"] = dict(replace_plan)
    if payload:
        _enqueue_command(("plan_update", payload))


def report_assistant_message(text: str, task_id: Optional[str] = None):
    """Main loop or API: append a plain-text line for the user (e.g. queued, arrived). Visible via GET /api/assistant/reports."""
    if not text or not text.strip():
        return
    with _plan_lock:
        if "assistant_reports" not in _plan_state:
            _plan_state["assistant_reports"] = []
        _plan_state["assistant_reports"].append({
            "text": text.strip(),
            "timestamp": time.time(),
            "task_id": task_id,
        })


def plan_report_external_result(task_id: str, success: bool = True, **data):
    """API: report virtual/external task result; message or text in body is appended to assistant_reports."""
    # Extract user-facing text only; do not dump full data as the report body
    message = data.pop("message", None) or data.pop("text", None)
    if message is not None and not isinstance(message, str):
        message = str(message)
    with _plan_lock:
        ts = time.time()
        if "external_results" not in _plan_state:
            _plan_state["external_results"] = []
        _plan_state["external_results"].append({
            "task_id": str(task_id),
            "success": success,
            "timestamp": ts,
            "message": message,
            **data,
        })
        if message:
            if "assistant_reports" not in _plan_state:
                _plan_state["assistant_reports"] = []
            _plan_state["assistant_reports"].append({
                "text": message,
                "timestamp": ts,
                "task_id": str(task_id),
            })


def plan_get_state() -> dict:
    """Main loop or API: read-only snapshot of plan state."""
    with _plan_lock:
        return {
            "plan_id": _plan_state["plan_id"],
            "plan": _plan_state["plan"],
            "status": _plan_state["status"],
            "step_index": _plan_state["step_index"],
            "task_index": _plan_state["task_index"],
            "results": list(_plan_state["results"]),
            "external_results": list(_plan_state.get("external_results", [])),
            "assistant_reports": list(_plan_state.get("assistant_reports", [])),
            "error": _plan_state["error"],
            "created_at": _plan_state["created_at"],
        }


# --- Called from main loop ---
def plan_set_and_start(plan_dict: dict) -> str:
    """Main loop: set plan and status running; returns plan_id."""
    plan_id = plan_dict.get("plan_id") or _gen_plan_id()
    with _plan_lock:
        _plan_state["plan_id"] = plan_id
        _plan_state["plan"] = dict(plan_dict)
        _plan_state["plan"]["plan_id"] = plan_id
        _plan_state["status"] = "running"
        _plan_state["step_index"] = 0
        _plan_state["task_index"] = 0
        _plan_state["results"] = []
        _plan_state["external_results"] = []
        _plan_state["assistant_reports"] = []
        _plan_state["error"] = None
        _plan_state["created_at"] = time.time()
    return plan_id


def plan_get_next_task() -> Optional[Tuple[int, int, str, dict]]:
    """Main loop: next task as (step_index, task_index, cmd_type, payload) or None."""
    with _plan_lock:
        if _plan_state["status"] != "running" or _plan_state["plan"] is None:
            return None
        steps = _plan_state["plan"].get("steps") or []
        si = _plan_state["step_index"]
        ti = _plan_state["task_index"]
        if si >= len(steps):
            return None
        step = steps[si]
        tasks = step.get("tasks") or []
        if ti >= len(tasks):
            return None
        t = tasks[ti]
        cmd_type = t.get("type")
        payload = t.get("payload") or {}
        if cmd_type not in ("navigate", "get_camera", "pick_place", "cancel_pick_place", "cancel_current_task"):
            return None
        return (si, ti, cmd_type, payload)


def plan_after_task(step_index: int, task_index: int, success: bool, **extra):
    """Main loop: record task result and advance indices."""
    with _plan_lock:
        _plan_state["results"].append({
            "step": step_index,
            "task": task_index,
            "success": success,
            **extra,
        })
        steps = _plan_state["plan"].get("steps") or []
        step = steps[step_index]
        tasks = step.get("tasks") or []
        if task_index + 1 < len(tasks):
            _plan_state["task_index"] = task_index + 1
        else:
            _plan_state["task_index"] = 0
            _plan_state["step_index"] = step_index + 1
        if _plan_state["step_index"] >= len(steps):
            _plan_state["status"] = "completed"


def plan_cancel():
    """Main loop: mark plan as cancelled."""
    with _plan_lock:
        _plan_state["status"] = "cancelled"


def plan_reset_to_idle_after_scene_reset() -> None:
    """After system scene reset: plan to idle; clear plan and assistant_reports (avoid stuck cancelled).
    assistant_reports must clear with server-side counters so dedup does not skip repeated messages."""
    with _plan_lock:
        _plan_state["plan_id"] = None
        _plan_state["plan"] = None
        _plan_state["status"] = "idle"
        _plan_state["step_index"] = 0
        _plan_state["task_index"] = 0
        _plan_state["results"] = []
        _plan_state["external_results"] = []
        _plan_state["assistant_reports"] = []
        _plan_state["error"] = None
        _plan_state["created_at"] = None


def plan_apply_update(upd: dict):
    """Main loop: apply one plan update (cancel / append_steps / replace_plan)."""
    with _plan_lock:
        if upd.get("cancel"):
            _plan_state["status"] = "cancelled"
            return
        if "replace_plan" in upd:
            r = upd["replace_plan"]
            plan_id = r.get("plan_id") or _gen_plan_id()
            _plan_state["plan_id"] = plan_id
            _plan_state["plan"] = dict(r)
            _plan_state["plan"]["plan_id"] = plan_id
            _plan_state["step_index"] = 0
            _plan_state["task_index"] = 0
            _plan_state["results"] = []
            _plan_state["external_results"] = []
            _plan_state["assistant_reports"] = []
            _plan_state["error"] = None
            _plan_state["status"] = "running"
            return
        if "append_steps" in upd and _plan_state["plan"]:
            steps = _plan_state["plan"].get("steps") or []
            steps.extend(upd["append_steps"])
            _plan_state["plan"]["steps"] = steps
            return


def _json_response(data: dict, status: int = 200):
    """Flask JSON response helper."""
    return _flask_app.response_class(
        response=json.dumps(data, ensure_ascii=False),
        status=status,
        mimetype="application/json",
    )


if _HAS_FLASK:
    @_flask_app.route("/api/status", methods=["GET"])
    def get_status():
        with _state_lock:
            doors = _state.get("doors", [])
        return _json_response({"status": "running", "doors": doors})

    @_flask_app.route("/api/doors", methods=["GET"])
    def list_doors():
        with _state_lock:
            doors = _state.get("doors", [])
            door_states = _state.get("door_states", {})
            rp = _state.get("robot_position")
            dxy = _state.get("door_world_xy") or []
            skip_ids = set(_state.get("skip_door_ids") or [])
        xy_by_id = {int(d.get("id", 0)): d for d in dxy if isinstance(d, dict) and d.get("id") is not None}
        out = []
        for i, n in enumerate(doors, 1):
            row: Dict[str, Any] = {"id": i, "name": n, "state": door_states.get(n, "unknown")}
            row["controllable"] = False
            row["note"] = (
                "All doors are opened at startup; POST /api/door/control is disabled."
                if i not in skip_ids
                else "Door skipped in this sim (known physics issue)."
            )
            dw = xy_by_id.get(i)
            if dw is not None:
                row["world_x"] = dw.get("x")
                row["world_y"] = dw.get("y")
                if rp and len(rp) >= 2 and dw.get("x") is not None and dw.get("y") is not None:
                    row["distance_m"] = round(
                        math.hypot(float(rp[0]) - float(dw["x"]), float(rp[1]) - float(dw["y"])),
                        3,
                    )
            out.append(row)
        return _json_response({"doors": out})

    @_flask_app.route("/api/robot/navigate", methods=["POST"])
    def robot_navigate():
        """Queue navigation target; returns 202. Steps in background; poll GET /api/robot/navigate_status."""
        body = request.get_json(force=True, silent=True) or {}
        x, y = body.get("x"), body.get("y")
        floor = body.get("floor", 0)
        goal_yaw_deg = body.get("goal_yaw_deg")
        if x is None or y is None:
            return _json_response({"success": False, "error": "x and y are required"}, 400)
        payload = {"x": float(x), "y": float(y), "floor": int(floor)}
        if goal_yaw_deg is not None:
            payload["goal_yaw_deg"] = float(goal_yaw_deg)
        _enqueue_command(("navigate", payload))
        return _json_response({"success": True, "status": "navigating", "message": "Navigation submitted; poll GET /api/robot/navigate_status."}, 202)

    @_flask_app.route("/api/robot/navigate_status", methods=["GET"])
    def robot_navigate_status():
        """Poll navigation: status idle | navigating | done; when done includes success and error."""
        st = get_single_nav_state()
        return _json_response({"success": True, **st})

    @_flask_app.route("/api/robot/status", methods=["GET"])
    def robot_status():
        with _state_lock:
            pos = _state.get("robot_position")
        if pos is None:
            return _json_response({"success": True, "position": []})
        return _json_response({"success": True, "position": list(pos)})

    @_flask_app.route("/api/camera/image", methods=["GET"])
    def get_camera_image():
        """Robot camera JPEG as base64. Query view=head|gripper (wrist; may fall back to head)."""
        view = request.args.get("view", "head") or "head"
        _enqueue_command(("get_camera", {"view": view}))
        try:
            result = _result_queue.get(timeout=10.0)
            if result.get("success") and not (result.get("image_base64") or "").strip():
                result = {"success": False, "error": "Image data is empty; try again shortly."}
            return _json_response(result)
        except queue.Empty:
            return _json_response({"success": False, "error": "Main loop did not respond within the timeout"}, 504)

    @_flask_app.route("/api/objects", methods=["GET"])
    def get_objects():
        """Graspable object names and placement target names for Pick&Place."""
        with _state_lock:
            graspable = list(_state.get("objects_graspable", []))
            placement = list(_state.get("objects_placement", []))
        return _json_response({"success": True, "objects_graspable": graspable, "objects_placement": placement})

    @_flask_app.route("/api/robot/pick_place", methods=["POST"])
    def robot_pick_place():
        """Queue Pick&Place; returns 202. Poll GET /api/robot/pick_place_status."""
        body = request.get_json(force=True, silent=True) or {}
        object_name = (body.get("object_name") or "").strip()
        target_name = (body.get("target_name") or "").strip()
        if not object_name or not target_name:
            return _json_response({"success": False, "error": "object_name and target_name are required"}, 400)
        ok, err = _enqueue_pick_place(object_name, target_name)
        if not ok:
            return _json_response(
                {"success": False, "error": err, "code": "PICK_PLACE_BUSY"},
                409,
            )
        return _json_response(
            {"success": True, "status": "running", "message": "Pick&Place submitted; poll GET /api/robot/pick_place_status."},
            202,
        )

    @_flask_app.route("/api/robot/pick_place_status", methods=["GET"])
    def robot_pick_place_status():
        """Poll Pick&Place: status idle | running | done; when done includes success and error."""
        st = get_single_pick_place_state()
        return _json_response({"success": True, **st})

    @_flask_app.route("/api/robot/pick_place/cancel", methods=["POST"])
    def robot_pick_place_cancel():
        """Request cancel of current Pick&Place if running."""
        _enqueue_cancel(("cancel_pick_place", {}))
        return _json_response({"success": True, "status": "cancel_requested", "message": "Pick&Place cancel request submitted."}, 202)

    @_flask_app.route("/api/robot/pick_place/transport_visual_ack", methods=["POST"])
    def robot_pick_place_transport_visual_ack():
        """After each transport-phase gripper vision check; JSON judgment hold|lost (default hold)."""
        body = request.get_json(force=True, silent=True) or {}
        j = str(body.get("judgment", "hold") or "hold")
        rep = report_pick_place_transport_visual_judgment(j)
        need = (
            get_pick_place_min_transport_visual_frames()
            if pick_place_transport_visual_gating_enabled()
            else 0
        )
        return _json_response(
            {
                "success": True,
                "transport_visual_acks": rep["transport_visual_acks"],
                "transport_visual_hold_streak": rep["transport_visual_hold_streak"],
                "transport_visual_required": need,
            },
            200,
        )

    @_flask_app.route("/api/task/stop", methods=["POST"])
    def stop_current_task():
        """Stop current task (nav / Pick&Place / current plan step)."""
        _enqueue_cancel(("cancel_current_task", {}))
        return _json_response({"success": True, "status": "stop_requested", "message": "Stop-current-task request submitted."}, 202)

    @_flask_app.route("/api/robot/grasp_status", methods=["GET"])
    def robot_grasp_status():
        """Poll grasp: holding, object_name, grasp_attempt_index (drop sim / retries)."""
        with _state_lock:
            g = dict(_state.get("grasp") or {})
        return _json_response({"success": True, **g})

    @_flask_app.route("/api/plan/status", methods=["GET"])
    def plan_status():
        """Current plan state (poll for live updates). Includes assistant_reports text lines."""
        st = plan_get_state()
        return _json_response({"success": True, **st})

    @_flask_app.route("/api/assistant/reports", methods=["GET"])
    def assistant_reports():
        """Plain-text assistant lines; message/text from external APIs appear here for the UI."""
        st = plan_get_state()
        reports = st.get("assistant_reports") or []
        messages = [r.get("text", "") for r in reports if r.get("text")]
        return _json_response({"success": True, "messages": messages})

    @_flask_app.route("/api/plan/submit", methods=["POST"])
    def plan_submit_endpoint():
        """Submit plan JSON with steps/tasks; returns plan_id; poll GET /api/plan/status."""
        body = request.get_json(force=True, silent=True) or {}
        try:
            plan_id = plan_submit(body)
            return _json_response({"success": True, "plan_id": plan_id, "status": "submitted"}, 202)
        except ValueError as e:
            return _json_response({"success": False, "error": str(e)}, 400)

    @_flask_app.route("/api/plan/update", methods=["POST"])
    def plan_update_endpoint():
        """Update running plan: cancel, append_steps, or replace_plan in body."""
        body = request.get_json(force=True, silent=True) or {}
        cancel = body.get("cancel") is True
        append_steps = body.get("append_steps")
        replace_plan = body.get("replace_plan")
        plan_update(cancel=cancel, append_steps=append_steps, replace_plan=replace_plan)
        return _json_response({"success": True, "message": "Plan update submitted."})

    @_flask_app.route("/api/plan/external_result", methods=["POST"])
    def plan_external_result():
        """Report external/virtual API result; include message or text for user; poll GET /api/assistant/reports."""
        body = request.get_json(force=True, silent=True) or {}
        task_id = body.get("task_id", "external")
        success = body.get("success", True)
        data = {k: v for k, v in body.items() if k not in ("task_id", "success")}
        plan_report_external_result(task_id, success=success, **data)
        return _json_response({"success": True, "message": "External result recorded."})

elif _HAS_FASTAPI:
    from fastapi import Body, Query
    from fastapi.responses import JSONResponse

    @app.get("/api/status")
    def get_status():
        with _state_lock:
            doors = _state.get("doors", [])
        return JSONResponse({"status": "running", "doors": doors})

    @app.get("/api/doors")
    def list_doors():
        with _state_lock:
            doors = _state.get("doors", [])
            door_states = _state.get("door_states", {})
            rp = _state.get("robot_position")
            dxy = _state.get("door_world_xy") or []
            skip_ids = set(_state.get("skip_door_ids") or [])
        xy_by_id = {int(d.get("id", 0)): d for d in dxy if isinstance(d, dict) and d.get("id") is not None}
        out = []
        for i, n in enumerate(doors, 1):
            row: Dict[str, Any] = {"id": i, "name": n, "state": door_states.get(n, "unknown")}
            row["controllable"] = False
            row["note"] = (
                "All doors are opened at startup; POST /api/door/control is disabled."
                if i not in skip_ids
                else "Door skipped in this sim (known physics issue)."
            )
            dw = xy_by_id.get(i)
            if dw is not None:
                row["world_x"] = dw.get("x")
                row["world_y"] = dw.get("y")
                if rp and len(rp) >= 2 and dw.get("x") is not None and dw.get("y") is not None:
                    row["distance_m"] = round(
                        math.hypot(float(rp[0]) - float(dw["x"]), float(rp[1]) - float(dw["y"])),
                        3,
                    )
            out.append(row)
        return JSONResponse({"doors": out})

    @app.post("/api/robot/navigate")
    def robot_navigate(body: dict = Body(default={})):
        """Queue navigation target; returns 202. Steps in background; poll GET /api/robot/navigate_status."""
        x, y = body.get("x"), body.get("y")
        floor = body.get("floor", 0)
        goal_yaw_deg = body.get("goal_yaw_deg")
        if x is None or y is None:
            return JSONResponse({"success": False, "error": "x and y are required"}, status_code=400)
        payload = {"x": float(x), "y": float(y), "floor": int(floor)}
        if goal_yaw_deg is not None:
            payload["goal_yaw_deg"] = float(goal_yaw_deg)
        _enqueue_command(("navigate", payload))
        return JSONResponse(
            {"success": True, "status": "navigating", "message": "Navigation submitted; poll GET /api/robot/navigate_status."},
            status_code=202,
        )

    @app.get("/api/robot/navigate_status")
    def robot_navigate_status():
        """Poll navigation: status idle | navigating | done; when done includes success and error."""
        st = get_single_nav_state()
        return JSONResponse({"success": True, **st})

    @app.get("/api/robot/status")
    def robot_status():
        with _state_lock:
            pos = _state.get("robot_position")
        if pos is None:
            return JSONResponse({"success": True, "position": []})
        return JSONResponse({"success": True, "position": list(pos)})

    @app.get("/api/camera/image")
    def get_camera_image(view: str = Query("head", description="head or gripper (wrist; may fall back if missing)")):
        """Robot camera JPEG as base64."""
        _enqueue_command(("get_camera", {"view": view or "head"}))
        try:
            result = _result_queue.get(timeout=10.0)
            if result.get("success") and not (result.get("image_base64") or "").strip():
                result = {"success": False, "error": "Image data is empty; try again shortly."}
            return JSONResponse(result)
        except queue.Empty:
            return JSONResponse({"success": False, "error": "Main loop did not respond within the timeout"}, status_code=504)

    @app.get("/api/robot/grasp_status")
    def robot_grasp_status():
        """Poll whether the robot is still grasping an object."""
        with _state_lock:
            g = dict(_state.get("grasp") or {})
        return JSONResponse({"success": True, **g})

    @app.get("/api/objects")
    def get_objects():
        """Graspable object names and placement target names for Pick&Place."""
        with _state_lock:
            graspable = list(_state.get("objects_graspable", []))
            placement = list(_state.get("objects_placement", []))
        return JSONResponse({"success": True, "objects_graspable": graspable, "objects_placement": placement})

    @app.post("/api/robot/pick_place")
    def robot_pick_place(body: dict = Body(default={})):
        """Queue Pick&Place; returns 202. Poll GET /api/robot/pick_place_status."""
        object_name = (body.get("object_name") or "").strip()
        target_name = (body.get("target_name") or "").strip()
        if not object_name or not target_name:
            return JSONResponse({"success": False, "error": "object_name and target_name are required"}, status_code=400)
        ok, err = _enqueue_pick_place(object_name, target_name)
        if not ok:
            return JSONResponse(
                {"success": False, "error": err, "code": "PICK_PLACE_BUSY"},
                status_code=409,
            )
        return JSONResponse(
            {"success": True, "status": "running", "message": "Pick&Place submitted; poll GET /api/robot/pick_place_status."},
            status_code=202,
        )

    @app.get("/api/robot/pick_place_status")
    def robot_pick_place_status():
        """Poll Pick&Place: status idle | running | done; when done includes success and error."""
        st = get_single_pick_place_state()
        return JSONResponse({"success": True, **st})

    @app.post("/api/robot/pick_place/cancel")
    def robot_pick_place_cancel():
        """Request cancel of current Pick&Place if running."""
        _enqueue_cancel(("cancel_pick_place", {}))
        return JSONResponse({"success": True, "status": "cancel_requested", "message": "Pick&Place cancel request submitted."}, status_code=202)

    @app.post("/api/robot/pick_place/transport_visual_ack")
    def robot_pick_place_transport_visual_ack(body: dict = Body(default={})):
        """Optional JSON judgment=hold|lost (default hold). Consecutive holds required before place."""
        j = str((body or {}).get("judgment", "hold") or "hold")
        rep = report_pick_place_transport_visual_judgment(j)
        need = (
            get_pick_place_min_transport_visual_frames()
            if pick_place_transport_visual_gating_enabled()
            else 0
        )
        return JSONResponse(
            {
                "success": True,
                "transport_visual_acks": rep["transport_visual_acks"],
                "transport_visual_hold_streak": rep["transport_visual_hold_streak"],
                "transport_visual_required": need,
            },
            status_code=200,
        )

    @app.post("/api/task/stop")
    def stop_current_task():
        """Stop current task (nav / Pick&Place / current plan step)."""
        _enqueue_cancel(("cancel_current_task", {}))
        return JSONResponse({"success": True, "status": "stop_requested", "message": "Stop-current-task request submitted."}, status_code=202)

    @app.get("/api/plan/status")
    def plan_status():
        """Current plan state (poll for live updates). Includes assistant_reports text lines."""
        st = plan_get_state()
        return JSONResponse({"success": True, **st})

    @app.get("/api/assistant/reports")
    def assistant_reports():
        """Plain-text assistant lines; message/text from external APIs appear here for the UI."""
        st = plan_get_state()
        reports = st.get("assistant_reports") or []
        messages = [r.get("text", "") for r in reports if r.get("text")]
        return JSONResponse({"success": True, "messages": messages})

    @app.post("/api/plan/submit")
    def plan_submit_endpoint(body: dict = Body(default={})):
        """Submit plan JSON with steps/tasks; returns plan_id; poll GET /api/plan/status."""
        try:
            plan_id = plan_submit(body)
            return JSONResponse({"success": True, "plan_id": plan_id, "status": "submitted"}, status_code=202)
        except ValueError as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=400)

    @app.post("/api/plan/update")
    def plan_update_endpoint(body: dict = Body(default={})):
        """Update running plan: cancel, append_steps, or replace_plan in body."""
        cancel = body.get("cancel") is True
        append_steps = body.get("append_steps")
        replace_plan = body.get("replace_plan")
        plan_update(cancel=cancel, append_steps=append_steps, replace_plan=replace_plan)
        return JSONResponse({"success": True, "message": "Plan update submitted."})

    @app.post("/api/plan/external_result")
    def plan_external_result(body: dict = Body(default={})):
        """Report external/virtual API result; include message or text for user; poll GET /api/assistant/reports."""
        task_id = body.get("task_id", "external")
        success = body.get("success", True)
        data = {k: v for k, v in body.items() if k not in ("task_id", "success")}
        plan_report_external_result(task_id, success=success, **data)
        return JSONResponse({"success": True, "message": "External result recorded."})

def update_state(
    doors: Optional[list] = None,
    door_states: Optional[dict] = None,
    robot_position: Optional[list] = None,
    objects_graspable: Optional[list] = None,
    objects_placement: Optional[list] = None,
    grasp: Optional[dict] = None,
    door_world_xy: Optional[list] = None,
    skip_door_ids: Optional[list] = None,
):
    """Main loop: push state for GET endpoints."""
    with _state_lock:
        if doors is not None:
            _state["doors"] = list(doors)
        if door_states is not None:
            _state["door_states"] = dict(door_states)
        if robot_position is not None:
            _state["robot_position"] = list(robot_position)
        if objects_graspable is not None:
            _state["objects_graspable"] = list(objects_graspable)
        if objects_placement is not None:
            _state["objects_placement"] = list(objects_placement)
        if grasp is not None:
            _state["grasp"] = dict(grasp)
        if door_world_xy is not None:
            _state["door_world_xy"] = list(door_world_xy)
        if skip_door_ids is not None:
            _state["skip_door_ids"] = [int(x) for x in skip_door_ids]


# ============================================================================
# Command-queue consumption (main loop)
# ============================================================================
_CANCEL_TYPES = frozenset({"cancel_pick_place", "cancel_current_task"})


def pop_preempt_pick_place() -> Optional[dict]:
    """Pop preempt pick_place payload dict or None."""
    global _preempt_pick_place
    with _preempt_pp_lock:
        pp = _preempt_pick_place
        _preempt_pick_place = None
        return pp


def get_next_command(timeout: float = 0.0) -> Optional[tuple]:
    """Pop front command or None; non-blocking (timeout unused)."""
    del timeout
    with _command_queue_lock:
        return _command_queue.popleft() if _command_queue else None


def pop_command_if_type(*types: str) -> Optional[tuple]:
    """Pop front only if its type is in types; otherwise leave queue unchanged."""
    with _command_queue_lock:
        if _command_queue and _command_queue[0][0] in types:
            return _command_queue.popleft()
        return None


def pop_first_matching_command(*types: str) -> Optional[tuple]:
    """Remove and return the first command whose type is in types; preserve order of the rest.
    Used to prioritize get_camera while navigation or Pick&Place is running."""
    if not types:
        return None
    type_set = frozenset(types)
    with _command_queue_lock:
        if not _command_queue:
            return None
        lst = list(_command_queue)
        _command_queue.clear()
        found: Optional[tuple] = None
        for c in lst:
            if found is None and c and c[0] in type_set:
                found = c
            else:
                _command_queue.append(c)
        return found


def peek_command_type() -> Optional[str]:
    """Peek front command type without consuming."""
    with _command_queue_lock:
        return _command_queue[0][0] if _command_queue else None


def has_pending_of_type(*types: str) -> bool:
    """True if any queued command has a type in types (non-consuming)."""
    with _command_queue_lock:
        return any(c[0] in types for c in _command_queue)


def put_result(success: bool, error: Optional[str] = None, **extra):
    """Main loop: push command result for waiting HTTP handlers; extra merged into JSON."""
    out = {"success": success, "error": error or ""}
    out.update(extra)
    _result_queue.put(out)


def _check_port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    """Return True if TCP connect to host:port succeeds within timeout."""
    for _ in range(int(timeout * 10)):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((host, port))
            s.close()
            return True
        except (socket.error, OSError):
            time.sleep(0.1)
        finally:
            try:
                s.close()
            except Exception:
                pass
    return False


def run_server(host: str = "0.0.0.0", port: int = 5001):
    """Start HTTP server in a background thread; prefer Flask under OmniGibson."""
    global _server_start_error
    if _HAS_FLASK:
        from werkzeug.serving import make_server

        def run():
            global _server_start_error
            try:
                server = make_server(host, port, _flask_app, threaded=True)
                server.serve_forever()
            except Exception as e:
                with _server_start_error_lock:
                    _server_start_error = str(e)
                print(f"[Behavior Env API] server thread failed to start: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t
    if _HAS_FASTAPI:
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)

        def run():
            global _server_start_error
            try:
                server.run()
            except Exception as e:
                with _server_start_error_lock:
                    _server_start_error = str(e)
                print(f"[Behavior Env API] server thread failed to start: {e}", file=sys.stderr, flush=True)
                import traceback
                traceback.print_exc(file=sys.stderr)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t
    raise RuntimeError("Install flask or fastapi, e.g. pip install flask  (recommended with OmniGibson)")


def wait_until_ready(port: int, timeout: float = 5.0) -> bool:
    """Block until API thread listens on port; returns whether connect succeeded."""
    ok = _check_port_open("127.0.0.1", port, timeout=timeout)
    if not ok:
        with _server_start_error_lock:
            err = _server_start_error
        if err:
            print(f"[Behavior Env API] port {port} not ready, error: {err}", file=sys.stderr, flush=True)
        else:
            print(
                f"[Behavior Env API] port {port} not ready within {timeout}s; check if another process uses it.",
                file=sys.stderr,
                flush=True,
            )
    return ok
