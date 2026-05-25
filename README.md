# OmniGibson 导航 + 门控 HTTP API

基于 [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) / OmniGibson 的室内导航仿真后端：场景 **Beechwood_0_int**，机器人 **Stretch**，支持 A* 导航、门开关、抓取放置、多步任务计划与 HTTP 远程控制。

> **说明**：本仓库为精简子项目，不含完整 BEHAVIOR-1K 数据集（约 2GB+）。克隆后需按下方步骤安装依赖并同步 `datasets/`。

---

## 目录

- [环境要求](#环境要求)
- [克隆仓库](#克隆仓库)
- [安装 Python 环境（从零开始）](#安装-python-环境从零开始)
- [准备数据与资源](#准备数据与资源)
- [启动服务](#启动服务)
- [端口](#端口)
- [API 功能概览](#api-功能概览)
- [API 详细说明](#api-详细说明)
- [环境变量](#环境变量)
- [仓库结构](#仓库结构)

---

## 环境要求


| 项目     | 说明                                                                          |
| ------ | --------------------------------------------------------------------------- |
| 系统     | Linux x86_64（与 [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) 一致） |
| GPU    | NVIDIA GPU + 驱动，建议 ≥ 8GB 显存                                                 |
| Conda  | Miniconda / Anaconda（用于创建虚拟环境）                                              |
| Python | **3.10**（由 conda 环境 `behavior` 提供）                                          |
| CUDA   | 默认按 **12.4** 安装 PyTorch（可通过 `setup.sh --cuda-version` 修改）                   |
| 磁盘     | 环境 + Isaac Sim 数 GB；`datasets/` 约 2GB（本仓库精简集）                               |
| 网络     | 安装阶段需下载 PyTorch、NVIDIA Isaac Sim wheel、首次 Kit 扩展                            |


> 若你本机已在完整 **BEHAVIOR-1K** 里装好 `conda activate behavior`，可跳到 [已有 BEHAVIOR 环境](#已有-behavior-环境快速路径)。

---

## 克隆仓库

```bash
git clone https://github.com/LxRoboticsLab/AI1112-final-project.git
cd lesson-project
chmod +x setup.sh
```

---

## 安装 Python 环境（从零开始）

流程与官方 **BEHAVIOR-1K** 的 `setup.sh` 一致，本仓库只保留导航 API 所需部分：**conda 环境 → PyTorch → BDDL → OmniGibson → Isaac Sim 4.5 → Flask**。

### 方式 A：一键脚本（推荐）

在仓库根目录执行（需接受 Conda / NVIDIA 条款）：

```bash
./setup.sh --full --accept-conda-tos --accept-nvidia-eula
```

等价于依次执行：`--new-env` → `--omnigibson` → `--api-deps`。

---

## 安装检查清单


| 检查项            | 命令                                                           |
| -------------- | ------------------------------------------------------------ |
| Conda 环境       | `conda activate behavior`                                    |
| Python 3.10    | `python --version`                                           |
| PyTorch + CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| Isaac Sim      | `python -c "import isaacsim"`                                |
| OmniGibson     | `python -c "import omnigibson"`                              |
| Flask          | `python -c "import flask"`                                   |
| 数据集目录          | `test -d datasets/behavior-1k-assets/scenes/Beechwood_0_int` |
| Logo           | `test -f docs/assets/OmniGibson_logo.png`                    |


---

## 资产下载

通过交大云盘下载：

分享内容: [datasets.zip](http://datasets.zip)

链接: [https://pan.sjtu.edu.cn/web/share/b78e44095bf5944d7dd094760caeb744](https://pan.sjtu.edu.cn/web/share/b78e44095bf5944d7dd094760caeb744), 提取码: 434j

下载资产后解压到lesson project文件夹中，格式.../lesson-project/datasets/...。

## 启动服务

```bash
conda activate behavior
cd /path/to/lesson-project
source env.local.sh
python -u navigation_with_doors_api.py
```

启动成功后日志会出现：

- `Behavior API ready: http://127.0.0.1:5001`
- `[BEHAVIOR_ENV_READY] api=http://127.0.0.1:5001 ...`

所有 HTTP 接口前缀为 `**/api**`，默认监听 `**0.0.0.0**`（局域网可访问）。

启动n8n可视化页面，新开终端。

```bash
npx n8n
```
打开 http://localhost:5678 进入可视化页面。

导入My workflow.json。

即可开始与agent交流。

---

## 端口


| 端口       | 环境变量                           | 用途                             |
| -------- | ------------------------------ | ------------------------------ |
| **5001** | `BEHAVIOR_ENV_PORT`（默认 `5001`） | **仿真 + HTTP API**（本仓库唯一对外服务端口） |
| —        | Isaac Sim / Kit                | 由 `isaacsim` 安装决定，非本仓库 HTTP 端口 |


> `env.local.sh` 中的 `BEHAVIOR_NAV_SUPERVISOR_PORT=5002` 为历史远端监督器预留，**当前精简仓库未包含** `remote_nav_env_supervisor.py`，无需开放 5002。

---

## API 功能概览


| 类别         | 能力                                 |
| ---------- | ---------------------------------- |
| 健康 / 状态    | 服务是否运行、门列表、机器人位姿、抓取状态              |
| 导航         | 提交目标点 A* 导航；轮询完成状态                 |
| 相机         | 获取头部或腕部 JPEG（Base64）               |
| 物体         | 列出可抓取物体与可放置目标                      |
| Pick&Place | 抓取并放置；轮询阶段；运输视觉确认；取消               |
| 任务计划       | 提交多步 `steps`/`tasks` 计划；查询进度；更新/取消 |


**异步约定**：`POST` 导航 / Pick&Place / 计划提交 等多返回 **202**，需在后台轮询对应 `GET ..._status` 或 `GET /api/plan/status`。

**同步约定**：`GET /api/camera/image` 会阻塞等待仿真主循环处理（有超时，通常 10–300s）。

---

## API 详细说明

**通用**

- `Content-Type`: `application/json`（POST 请求体）
- 响应体均为 JSON
- 基础 URL：`http://<主机>:5001`

### 健康与状态

#### `GET /api/status`

服务就绪探测。

**响应示例：**

```json
{
  "status": "running",
  "doors": ["door_name_1", "door_name_2"]
}
```

---

#### `GET /api/robot/status`

**响应示例：**

```json
{
  "success": true,
  "position": [1.2, -3.4, 0.05]
}
```

---

#### `GET /api/robot/grasp_status`

**响应示例：**

```json
{
  "success": true,
  "holding": false,
  "object_name": null,
  "grasp_attempt_index": 0
}
```

---

#### `GET /api/objects`

**响应示例：**

```json
{
  "success": true,
  "objects_graspable": ["apple_abc_0", "mug_def_0"],
  "objects_placement": ["countertop_tpuwys_0", "fridge_dszchb_0"]
}
```

---

### 导航

#### `POST /api/robot/navigate`

**请求体：**

```json
{
  "x": 2.5,
  "y": -1.0,
  "floor": 0,
  "goal_yaw_deg": 90.0
}
```


| 字段             | 类型    | 必填  | 说明        |
| -------------- | ----- | --- | --------- |
| `x`, `y`       | float | 是   | 世界坐标目标（米） |
| `floor`        | int   | 否   | 楼层，默认 `0` |
| `goal_yaw_deg` | float | 否   | 到达后朝向（度）  |


**响应（202）：**

```json
{
  "success": true,
  "status": "navigating",
  "message": "Navigation submitted; poll GET /api/robot/navigate_status."
}
```

---

#### `GET /api/robot/navigate_status`

**响应示例：**

```json
{
  "success": true,
  "status": "done",
  "error": ""
}
```

`status` 取值：`idle` | `navigating` | `done`。当 `status` 为 `done` 时，响应里的 `success` 表示**导航是否成功**（非 HTTP 状态），`error` 为失败原因字符串。

---

### 相机

#### `GET /api/camera/image?view=head`


| 查询参数   | 说明                                 |
| ------ | ---------------------------------- |
| `view` | `head`（默认）或 `gripper`（腕部，可能回退到头相机） |


**响应示例（同步，超时约 10s）：**

```json
{
  "success": true,
  "image_base64": "<JPEG Base64 字符串>",
  "view": "head"
}
```

---

### Pick & Place

#### `POST /api/robot/pick_place`

**请求体：**

```json
{
  "object_name": "apple_xxxxx_0",
  "target_name": "countertop_tpuwys_0"
}
```

**响应（202）或忙时（409）：**

```json
{
  "success": false,
  "error": "PICK_PLACE_BUSY: ...",
  "code": "PICK_PLACE_BUSY"
}
```

---

#### `GET /api/robot/pick_place_status`

**响应示例：**

```json
{
  "success": true,
  "status": "running",
  "phase": "nav_to_obj",
  "object_name": "apple_xxxxx_0",
  "target_name": "countertop_tpuwys_0",
  "success": null,
  "error": null
}
```

`status`: `idle` | `submitted` | `running` | `done`；`phase`: `nav_to_obj` | `attach` | `nav_to_target` | `place` 等。

---

#### `POST /api/robot/pick_place/cancel`

无请求体。**响应（202）**。

---

#### `POST /api/robot/pick_place/transport_visual_ack`

运输阶段视觉确认（连续 `hold` 后才允许放置）。

**请求体：**

```json
{
  "judgment": "hold"
}
```

`judgment`: `"hold"` | `"lost"`

**响应：**

```json
{
  "success": true,
  "transport_visual_acks": 1,
  "transport_visual_hold_streak": 1,
  "transport_visual_required": 3
}
```

---

#### `POST /api/task/stop`

停止当前导航 / Pick&Place / 计划步骤。**无请求体，202。**

---

### 任务计划（多步编排）

#### `POST /api/plan/submit`

**请求体：**

```json
{
  "plan_id": "optional_custom_id",
  "steps": [
    {
      "tasks": [
        {
          "type": "navigate",
          "payload": { "x": 1.0, "y": -2.0, "floor": 0 }
        },
        {
          "type": "get_camera",
          "payload": { "view": "head" }
        },
        {
          "type": "pick_place",
          "payload": {
            "object_name": "apple_xxxxx_0",
            "target_name": "table_yyyyy_0"
          }
        }
      ]
    }
  ]
}
```

支持的 `type`：`navigate` | `get_camera` | `pick_place` | `cancel_pick_place` | `cancel_current_task`。

**响应（202）：**

```json
{
  "success": true,
  "plan_id": "plan_a1b2c3d4e5f6",
  "status": "submitted"
}
```

---

#### `GET /api/plan/status`

**响应示例：**

```json
{
  "success": true,
  "plan_id": "plan_a1b2c3d4e5f6",
  "status": "running",
  "step_index": 0,
  "task_index": 1,
  "results": [],
  "assistant_reports": [
    { "text": "[NAV] reached waypoint 2", "timestamp": 1716288000.0, "task_id": null }
  ],
  "external_results": [],
  "error": null
}
```

`status`: `idle` | `running` | `completed` | `cancelled`

---

#### `POST /api/plan/update`

**请求体（字段可选，至少其一）：**

```json
{
  "cancel": true,
  "append_steps": [{ "tasks": [...] }],
  "replace_plan": { "plan_id": "new", "steps": [...] }
}
```

---

#### `POST /api/plan/external_result`

上报外部 API（天气等虚拟任务）结果，文本会进入 `assistant_reports`。

**请求体：**

```json
{
  "task_id": "weather_shanghai",
  "success": true,
  "message": "上海今天晴，25°C"
}
```

---

#### `GET /api/assistant/reports`

**响应：**

```json
{
  "success": true,
  "messages": ["到达目标点", "Pick&Place finished (success)"]
}
```

---

## 环境变量


| 变量                             | 默认                     | 说明                                         |
| ------------------------------ | ---------------------- | ------------------------------------------ |
| `BEHAVIOR_ENV_PORT`            | `5001`                 | HTTP API 端口                                |
| `OMNIGIBSON_DATA_PATH`         | `./datasets`           | 由 `env.local.sh` / `bootstrap_paths.py` 设置 |
| `OMNIGIBSON_APPDATA_PATH`      | `./OmniGibson/appdata` | Kit 缓存与日志                                  |
| `BEHAVIOR_API_READY_TIMEOUT_S` | `30`                   | 等待 API 端口就绪超时（秒）                           |


完整列表见 `navigation_with_doors_api.py` 文件头部常量。

---

## 仓库结构

```
.
├── navigation_with_doors_api.py   # 入口：仿真主循环 + HTTP API
├── navigation_with_doors.py       # 导航、门、可通行地图
├── navigation_with_pick_place.py  # 抓取放置
├── behavior_env_api.py            # Flask/FastAPI 路由与命令队列
├── door_barrier_walls.py
├── spawn_small_objects.py
├── bootstrap_paths.py
├── setup.sh                       # 安装脚本（conda / PyTorch / OmniGibson / Isaac / Flask）
├── requirements-api.txt           # Flask
├── env.local.sh
├── docs/assets/OmniGibson_logo.png
├── datasets/                      # .gitignore，需 sync_minimal_datasets.py
├── OmniGibson/                    # 仿真框架（子集）
├── bddl3/                         # OmniGibson 依赖
└── scripts/sync_minimal_datasets.py
```

---

## 推送到 GitHub 的注意事项

1. **不要提交** `datasets/`、`OmniGibson/appdata/`（已在 `.gitignore`）。
2. 克隆者须按上文 [安装 Python 环境](#安装-python-环境从零开始) 与 [准备数据](#准备数据与资源) 操作。
3. `setup.sh` 需保留可执行权限：`git update-index --chmod=+x setup.sh`。

---

## License

OmniGibson / BEHAVIOR-1K 数据与代码遵循其上游许可证（见 `OmniGibson/LICENSE` 及 BEHAVIOR 数据 EULA）。本仓库自定义导航与 API 代码请以你仓库声明的许可证为准。
