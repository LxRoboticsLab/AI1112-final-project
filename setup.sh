#!/bin/bash
# 精简版 BEHAVIOR-1K 安装脚本：仅 OmniGibson + BDDL + Isaac Sim + API 依赖
# 用法见 README.md 或: ./setup.sh --help
set -e

HELP=false
NEW_ENV=false
OMNIGIBSON=false
API_DEPS=false
CUDA_VERSION="12.4"
ACCEPT_CONDA_TOS=false
ACCEPT_NVIDIA_EULA=false
CONFIRM_NO_CONDA=false
FULL=false

[ "$#" -eq 0 ] && HELP=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) HELP=true; shift ;;
        --new-env) NEW_ENV=true; shift ;;
        --omnigibson) OMNIGIBSON=true; shift ;;
        --api-deps) API_DEPS=true; shift ;;
        --full) FULL=true; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --accept-conda-tos) ACCEPT_CONDA_TOS=true; shift ;;
        --accept-nvidia-eula) ACCEPT_NVIDIA_EULA=true; shift ;;
        --confirm-no-conda) CONFIRM_NO_CONDA=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$FULL" = true ]; then
    NEW_ENV=true
    OMNIGIBSON=true
    API_DEPS=true
    ACCEPT_CONDA_TOS=true
    ACCEPT_NVIDIA_EULA=true
fi

if [ "$HELP" = true ]; then
    cat << 'EOF'
导航 API 环境安装脚本（基于 BEHAVIOR-1K setup.sh 精简）

用法:
  ./setup.sh --full --accept-conda-tos --accept-nvidia-eula
  或分步:
  ./setup.sh --new-env --accept-conda-tos
  conda activate behavior
  ./setup.sh --omnigibson --accept-nvidia-eula
  ./setup.sh --api-deps

选项:
  -h, --help              显示帮助
  --new-env               创建 conda 环境 behavior（Python 3.10 + PyTorch）
  --omnigibson            安装 bddl3、OmniGibson、Isaac Sim 4.5（需已激活 behavior）
  --api-deps              安装 Flask（HTTP API）
  --full                  等价于 --new-env --omnigibson --api-deps（配合 accept 标志）
  --cuda-version VERSION  CUDA 版本，默认 12.4
  --accept-conda-tos      自动接受 Conda 服务条款
  --accept-nvidia-eula    自动接受 NVIDIA Isaac Sim EULA
  --confirm-no-conda      非 conda 环境时跳过确认

安装完成后:
  conda activate behavior
  source env.local.sh
  python scripts/sync_minimal_datasets.py --source /path/to/BEHAVIOR-1K/datasets
  python -u navigation_with_doors_api.py
EOF
    exit 0
fi

WORKDIR=$(cd "$(dirname "$0")" && pwd)
cd "$WORKDIR"

if [ "$NEW_ENV" = false ] && [ -z "$CONDA_PREFIX" ]; then
    if [ "$CONFIRM_NO_CONDA" = false ]; then
        echo ""
        echo "WARNING: 当前不在 conda 环境中: $(which python)"
        echo "建议: ./setup.sh --new-env  或  conda activate behavior"
        read -r -p "继续? [y/N] " response
        [[ "$response" =~ ^[Yy]$ ]] || exit 1
    fi
fi

# --- 创建 conda 环境 + PyTorch ---
if [ "$NEW_ENV" = true ]; then
    command -v conda >/dev/null || { echo "ERROR: 未找到 conda"; exit 1; }
    if [ "$ACCEPT_CONDA_TOS" = true ]; then
        export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    echo ">>> 安装 numpy / setuptools..."
    pip install "numpy<2" "setuptools<=79"

    echo ">>> 安装 PyTorch (CUDA ${CUDA_VERSION})..."
    CUDA_VER_SHORT=$(echo "$CUDA_VERSION" | sed 's/\.//g')
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_VER_SHORT}"
    echo "✓ PyTorch 安装完成"
fi

# --- BDDL + OmniGibson + Isaac Sim ---
if [ "$OMNIGIBSON" = true ]; then
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    [ "$PYTHON_VERSION" != "3.10" ] && { echo "ERROR: 需要 Python 3.10，当前 $PYTHON_VERSION"; exit 1; }

    if [[ -n "$EXP_PATH" || -n "$CARB_APP_PATH" || -n "$ISAAC_PATH" ]]; then
        echo "ERROR: 检测到已有 Isaac Sim 环境变量，请 unset EXP_PATH CARB_APP_PATH ISAAC_PATH 后重试"
        exit 1
    fi

    [ ! -d "$WORKDIR/bddl3" ] && { echo "ERROR: 缺少 bddl3/"; exit 1; }
    [ ! -d "$WORKDIR/OmniGibson" ] && { echo "ERROR: 缺少 OmniGibson/"; exit 1; }

    echo ">>> pip install -e bddl3..."
    pip install -e "$WORKDIR/bddl3"

    echo ">>> pip install -e OmniGibson..."
    pip install -e "$WORKDIR/OmniGibson"

    if [ "$ACCEPT_NVIDIA_EULA" != true ]; then
        echo "ERROR: 安装 Isaac Sim 需要 --accept-nvidia-eula"
        exit 1
    fi
    export OMNI_KIT_ACCEPT_EULA=YES

    if python -c "import isaacsim" 2>/dev/null; then
        echo "✓ Isaac Sim 已安装，跳过"
    else
        NVIDIA_INDEX="https://pypi.nvidia.com"
        ISAAC_PIP_SPECS=(
            "omniverse-kit==106.5.0.162521"
            "isaacsim-kernel==4.5.0.0" "isaacsim-app==4.5.0.0" "isaacsim-core==4.5.0.0"
            "isaacsim-gui==4.5.0.0" "isaacsim-utils==4.5.0.0" "isaacsim-storage==4.5.0.0"
            "isaacsim-asset==4.5.0.0" "isaacsim-sensor==4.5.0.0" "isaacsim-robot-motion==4.5.0.0"
            "isaacsim-robot==4.5.0.0" "isaacsim-benchmark==4.5.0.0" "isaacsim-code-editor==4.5.0.0"
            "isaacsim-ros1==4.5.0.0" "isaacsim-cortex==4.5.0.0" "isaacsim-example==4.5.0.0"
            "isaacsim-replicator==4.5.0.0" "isaacsim-rl==4.5.0.0" "isaacsim-robot-setup==4.5.0.0"
            "isaacsim-ros2==4.5.0.0" "isaacsim-template==4.5.0.0" "isaacsim-test==4.5.0.0"
            "isaacsim==4.5.0.0" "isaacsim-extscache-physics==4.5.0.0"
            "isaacsim-extscache-kit==4.5.0.0" "isaacsim-extscache-kit-sdk==4.5.0.0"
        )
        ISAAC_WHEEL_PKGS=(
            "omniverse_kit-106.5.0.162521" "isaacsim_kernel-4.5.0.0" "isaacsim_app-4.5.0.0"
            "isaacsim_core-4.5.0.0" "isaacsim_gui-4.5.0.0" "isaacsim_utils-4.5.0.0"
            "isaacsim_storage-4.5.0.0" "isaacsim_asset-4.5.0.0" "isaacsim_sensor-4.5.0.0"
            "isaacsim_robot_motion-4.5.0.0" "isaacsim_robot-4.5.0.0" "isaacsim_benchmark-4.5.0.0"
            "isaacsim_code_editor-4.5.0.0" "isaacsim_ros1-4.5.0.0" "isaacsim_cortex-4.5.0.0"
            "isaacsim_example-4.5.0.0" "isaacsim_replicator-4.5.0.0" "isaacsim_rl-4.5.0.0"
            "isaacsim_robot_setup-4.5.0.0" "isaacsim_ros2-4.5.0.0" "isaacsim_template-4.5.0.0"
            "isaacsim_test-4.5.0.0" "isaacsim-4.5.0.0" "isaacsim_extscache_physics-4.5.0.0"
            "isaacsim_extscache_kit-4.5.0.0" "isaacsim_extscache_kit_sdk-4.5.0.0"
        )

        check_glibc_old() { ldd --version 2>&1 | grep -qE "2\.(31|32|33)"; }
        WHEEL_CACHE="$WORKDIR/.isaac_wheels_cache"
        mkdir -p "$WHEEL_CACHE"

        download_one_wheel() {
            local pkg="$1"
            local pkg_name=${pkg%-*}
            local filename="${pkg}-cp310-none-manylinux_2_34_x86_64.whl"
            if check_glibc_old; then
                filename="${pkg}-cp310-none-manylinux_2_31_x86_64.whl"
            fi
            local url="https://pypi.nvidia.com/${pkg_name//_/-}/$filename"
            local dest="$WHEEL_CACHE/$filename"
            if [[ -f "$dest" && -s "$dest" ]]; then
                echo "  ✓ 缓存命中 $filename"
                echo "$dest"
                return 0
            fi
            local attempt
            for attempt in 1 2 3 4 5 6; do
                echo "  下载 $pkg (第 ${attempt}/6 次) ..."
                # -C - 断点续传；-retry 应对 SSL EOF / 瞬时断连
                if curl -fSL \
                    --connect-timeout 60 \
                    --max-time 7200 \
                    --retry 5 --retry-delay 10 --retry-all-errors \
                    -C - "$url" -o "$dest.part" && mv "$dest.part" "$dest"; then
                    echo "  ✓ 完成 $filename"
                    echo "$dest"
                    return 0
                fi
                rm -f "$dest.part"
                echo "  ⚠️  下载失败，${attempt} 秒后重试 ..."
                sleep $((attempt * 5))
            done
            echo "ERROR: 多次重试仍无法下载 $pkg"
            echo "  URL: $url"
            echo "  可手动: curl -C - -o \"$dest\" \"$url\"  然后重新运行 ./setup.sh --omnigibson --accept-nvidia-eula"
            return 1
        }

        install_isaac_via_wheels() {
            echo ">>> 方式 2/2: 分 wheel 下载（支持断点续传，缓存目录 .isaac_wheels_cache/）..."
            local wheel_files=()
            local pkg dest
            for pkg in "${ISAAC_WHEEL_PKGS[@]}"; do
                dest=$(download_one_wheel "$pkg") || return 1
                wheel_files+=("$dest")
            done
            pip install "${wheel_files[@]}"
        }

        pip install --upgrade pip wheel
        echo ">>> 方式 1/2: pip 从 ${NVIDIA_INDEX} 安装 Isaac Sim 4.5（自动重试）..."
        if ! pip install "${ISAAC_PIP_SPECS[@]}" \
            --extra-index-url "$NVIDIA_INDEX" \
            --trusted-host pypi.nvidia.com \
            --retries 10 \
            --timeout 300; then
            echo "⚠️  pip 直连安装未成功，改用手动 wheel 下载（带断点续传）..."
            install_isaac_via_wheels || exit 1
        fi

        python -c "import isaacsim" || { echo "ERROR: isaacsim 导入失败"; exit 1; }
        ISAAC_PATH=$(python -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))")
        if [ -n "$ISAAC_PATH" ] && [ -d "$ISAAC_PATH/extscache" ]; then
            find "$ISAAC_PATH/extscache" -type d -name "websockets" -path "*/pip_prebundle/*" \
                -exec rm -rf {} + 2>/dev/null || true
        fi
        echo "✓ Isaac Sim 安装完成"
    fi

    pip install --force-reinstall cffi==1.17.1
    python -c "import omnigibson; print('✓ omnigibson', omnigibson.__version__ if hasattr(omnigibson,'__version__') else 'ok')"
fi

# --- HTTP API ---
if [ "$API_DEPS" = true ]; then
    echo ">>> 安装 API 依赖 (Flask)..."
    pip install -r "$WORKDIR/requirements-api.txt"
    python -c "import flask; print('✓ flask', flask.__version__)"
fi

echo ""
echo "=== 安装步骤完成 ==="
[ "$NEW_ENV" = true ] && echo "  conda activate behavior"
echo "  source env.local.sh"
echo "  python scripts/sync_minimal_datasets.py --source /path/to/BEHAVIOR-1K/datasets"
echo "  python -u navigation_with_doors_api.py"
