#!/bin/bash
# Ubuntu环境下的数据预处理配置指南脚本（无sudo版本）
# 按照contrast-phys README的指示，完成所有必要的配置和下载
# 本脚本不使用sudo，所有依赖都通过conda/mamba或用户目录安装

set +e  # 不使用set -e，以便更好地处理错误

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Ubuntu环境数据预处理配置指南（无sudo版本）"
echo "============================================================"
echo ""
echo "本脚本将帮助您完成以下步骤："
echo "1. 检查系统依赖（使用conda/mamba，无需sudo）"
echo "2. 安装/编译dlib（如需要，安装到用户目录）"
echo "3. 安装/编译OpenFace（安装到用户目录）"
echo "4. 下载OpenFace模型文件"
echo "5. 验证安装"
echo ""
echo "注意：本脚本不使用sudo，所有依赖都通过conda/mamba安装"
echo "============================================================"

# 检测conda/mamba环境
CONDA_ENV="/home/vt_ai_test1/mamba-envs/ml"
if [ -d "$CONDA_ENV" ]; then
    echo "✅ 找到conda环境: $CONDA_ENV"
    # 激活conda环境（如果可用）
    if command -v conda &> /dev/null || command -v mamba &> /dev/null; then
        echo "✅ conda/mamba可用"
    fi
else
    echo "⚠️  未找到conda环境: $CONDA_ENV"
    echo "   将使用系统Python和pip"
fi

# ============================================================
# 步骤1: 检查系统依赖
# ============================================================
echo ""
echo "步骤1: 检查系统依赖"
echo "============================================================"

MISSING_DEPS=()

# 检查cmake
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake未安装"
    MISSING_DEPS+=("cmake")
else
    echo "✅ cmake已安装: $(cmake --version | head -1)"
fi

# 检查g++
if ! command -v g++ &> /dev/null; then
    echo "❌ g++未安装"
    MISSING_DEPS+=("build-essential")
else
    echo "✅ g++已安装: $(g++ --version | head -1)"
fi

# 检查OpenCV（通过pkg-config）
if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
    echo "⚠️  OpenCV未找到（可能已安装但未配置pkg-config）"
    echo "   如果后续编译失败，需要安装OpenCV开发包"
else
    echo "✅ OpenCV已找到"
fi

# 检查Python依赖
echo ""
echo "检查Python依赖..."
MISSING_PYTHON_DEPS=()
python3 << 'PYTHON_CHECK'
import sys
missing = []
try:
    import cv2
    print("✅ opencv-python")
except ImportError:
    missing.append("opencv-python")
    print("❌ opencv-python")

try:
    import numpy
    print("✅ numpy")
except ImportError:
    missing.append("numpy")
    print("❌ numpy")

try:
    import h5py
    print("✅ h5py")
except ImportError:
    missing.append("h5py")
    print("❌ h5py")

try:
    import pandas
    print("✅ pandas")
except ImportError:
    missing.append("pandas")
    print("❌ pandas")

if missing:
    print(f"\n⚠️  缺少Python包: {', '.join(missing)}")
    sys.exit(1)
PYTHON_CHECK

PYTHON_EXIT=$?
if [ $PYTHON_EXIT -ne 0 ]; then
    echo "⚠️  部分Python依赖缺失"
    echo ""
    echo "安装方式（选择其一）："
    echo ""
    echo "方式1（推荐）: 使用conda/mamba安装（无需sudo）"
    if [ -d "$CONDA_ENV" ]; then
        echo "   conda activate $CONDA_ENV  # 或 mamba activate $CONDA_ENV"
        echo "   conda install -c conda-forge opencv numpy h5py pandas"
        echo "   或者使用environment.yml:"
        echo "   conda env create -f environment.yml"
    else
        echo "   先创建conda环境:"
        echo "   conda create -n contrast-phys python=3.10"
        echo "   conda activate contrast-phys"
        echo "   conda install -c conda-forge opencv numpy h5py pandas"
    fi
    echo ""
    echo "方式2: 使用pip安装（用户目录，无需sudo）"
    echo "   pip3 install --user opencv-python numpy h5py pandas"
    echo ""
    read -p "是否现在使用pip安装到用户目录? (y/n): " install_python_deps
    if [ "$install_python_deps" = "y" ]; then
        pip3 install --user opencv-python numpy h5py pandas
    else
        echo "请手动安装Python依赖后重新运行此脚本"
    fi
fi

# 如果有缺失的系统依赖，提供无sudo的解决方案
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "============================================================"
    echo "⚠️  检测到缺失的系统依赖: ${MISSING_DEPS[@]}"
    echo "============================================================"
    echo ""
    echo "由于无法使用sudo，有以下替代方案："
    echo ""
    echo "方案1（推荐）: 使用conda/mamba安装编译工具（无需sudo）"
    if [ -d "$CONDA_ENV" ]; then
        echo "   conda activate $CONDA_ENV"
        echo "   conda install -c conda-forge cmake compilers"
    else
        echo "   先创建conda环境:"
        echo "   conda create -n contrast-phys python=3.10"
        echo "   conda activate contrast-phys"
        echo "   conda install -c conda-forge cmake compilers"
    fi
    echo ""
    echo "方案2: 如果系统已安装但路径未配置，可以继续尝试编译"
    echo "   OpenFace和dlib的安装脚本会尝试使用conda环境中的工具"
    echo ""
    echo "方案3: 联系系统管理员安装以下包："
    echo "   sudo apt-get install -y ${MISSING_DEPS[@]}"
    echo ""
    read -p "是否继续（即使缺少系统依赖）? (y/n): " continue_anyway
    if [ "$continue_anyway" != "y" ]; then
        echo "请先安装依赖后重新运行此脚本"
        exit 1
    fi
fi

# ============================================================
# 步骤2: 检查并安装dlib（如需要）
# ============================================================
echo ""
echo "============================================================"
echo "步骤2: 检查dlib"
echo "============================================================"

CONDA_ENV="/home/vt_ai_test1/mamba-envs/ml"
LOCAL_PREFIX="$PROJECT_DIR/local"
DLIB_CMAKE_PATH="$LOCAL_PREFIX/lib/cmake/dlib"

if [ -f "$DLIB_CMAKE_PATH/dlibConfig.cmake" ] || [ -f "$DLIB_CMAKE_PATH/dlib-config.cmake" ]; then
    echo "✅ dlib已安装: $DLIB_CMAKE_PATH"
elif [ -d "$CONDA_ENV/lib/cmake/dlib" ]; then
    echo "✅ 找到conda环境中的dlib"
else
    echo "⚠️  dlib未找到"
    echo ""
    read -p "是否现在编译安装dlib? (y/n): " install_dlib
    if [ "$install_dlib" = "y" ]; then
        echo "运行: bash prep/install_dlib.sh"
        bash prep/install_dlib.sh
    else
        echo "⚠️  跳过dlib安装，如果后续OpenFace编译失败，请先安装dlib"
    fi
fi

# ============================================================
# 步骤3: 检查并安装OpenFace
# ============================================================
echo ""
echo "============================================================"
echo "步骤3: 检查OpenFace"
echo "============================================================"

OPENFACE_DIR="$PROJECT_DIR/OpenFace"
CONDA_OPENFACE_BIN="$CONDA_ENV/local/bin/FeatureExtraction"
BUILD_OPENFACE_BIN="$OPENFACE_DIR/build/bin/FeatureExtraction"

if [ -f "$CONDA_OPENFACE_BIN" ]; then
    echo "✅ OpenFace已安装: $CONDA_OPENFACE_BIN"
    OPENFACE_BIN="$CONDA_OPENFACE_BIN"
elif [ -f "$BUILD_OPENFACE_BIN" ]; then
    echo "✅ OpenFace已编译: $BUILD_OPENFACE_BIN"
    OPENFACE_BIN="$BUILD_OPENFACE_BIN"
else
    echo "❌ OpenFace未安装或未编译"
    echo ""
    echo "安装方式（无sudo版本）："
    echo ""
    echo "方式1（推荐）: 使用项目提供的脚本（自动处理conda环境，无需sudo）"
    echo "   bash prep/install_openface_step_by_step.sh"
    echo ""
    echo "⚠️  注意: OpenFace自带的install.sh需要sudo权限，不适用于无sudo环境"
    echo ""
    read -p "是否现在安装OpenFace? (y/n): " install_openface
    
    if [ "$install_openface" = "y" ]; then
        echo "运行安装脚本..."
        bash prep/install_openface_step_by_step.sh
        # 重新检查
        if [ -f "$CONDA_OPENFACE_BIN" ]; then
            OPENFACE_BIN="$CONDA_OPENFACE_BIN"
            echo "✅ OpenFace安装成功: $CONDA_OPENFACE_BIN"
        elif [ -f "$BUILD_OPENFACE_BIN" ]; then
            OPENFACE_BIN="$BUILD_OPENFACE_BIN"
            echo "✅ OpenFace编译成功: $BUILD_OPENFACE_BIN"
        else
            echo "⚠️  OpenFace安装可能未完成，请检查错误信息"
        fi
    else
        echo "⚠️  跳过OpenFace安装，请稍后手动安装"
        OPENFACE_BIN=""
    fi
fi

# ============================================================
# 步骤4: 检查OpenFace模型文件
# ============================================================
echo ""
echo "============================================================"
echo "步骤4: 检查OpenFace模型文件"
echo "============================================================"

# OpenFace模型文件可能在多个位置
MODEL_DIRS=(
    "$OPENFACE_DIR/model"
    "$OPENFACE_DIR/lib/local/LandmarkDetector/model"
    "$OPENFACE_DIR/lib/LandmarkDetector/model"
)
MODEL_DIR=""
for dir in "${MODEL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        MODEL_DIR="$dir"
        break
    fi
done

MODEL_FILES=(
    "main_ceclm_general.txt"
    "main_clnf_general.txt"
    "main_clnf_wild.txt"
    "main_clnf_multi_pie.txt"
    "main_clm_general.txt"
)

MISSING_MODELS=()
if [ -z "$MODEL_DIR" ]; then
    echo "⚠️  未找到model目录，模型文件可能未下载"
    MISSING_MODELS=("${MODEL_FILES[@]}")
else
    echo "✅ 找到model目录: $MODEL_DIR"
    for model_file in "${MODEL_FILES[@]}"; do
        if [ -f "$MODEL_DIR/$model_file" ]; then
            echo "✅ $model_file"
        else
            echo "❌ $model_file 缺失"
            MISSING_MODELS+=("$model_file")
        fi
    done
fi

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "需要下载OpenFace模型文件"
    if [ -f "$OPENFACE_DIR/download_models.sh" ]; then
        read -p "是否现在下载? (y/n): " download_models
        if [ "$download_models" = "y" ]; then
            cd "$OPENFACE_DIR"
            bash download_models.sh
            cd "$PROJECT_DIR"
        fi
    else
        echo "⚠️  download_models.sh不存在，请手动下载模型文件"
    fi
fi

# ============================================================
# 步骤5: 验证安装
# ============================================================
echo ""
echo "============================================================"
echo "步骤5: 验证安装"
echo "============================================================"

if [ -n "$OPENFACE_BIN" ] && [ -f "$OPENFACE_BIN" ]; then
    echo "✅ FeatureExtraction可执行文件存在: $OPENFACE_BIN"
    echo ""
    echo "测试运行（显示帮助信息）:"
    "$OPENFACE_BIN" 2>&1 | head -20 || true
else
    echo "❌ FeatureExtraction未找到"
    echo "请完成OpenFace安装后再继续"
fi

# ============================================================
# 总结和下一步
# ============================================================
echo ""
echo "============================================================"
echo "配置总结"
echo "============================================================"

if [ -n "$OPENFACE_BIN" ] && [ -f "$OPENFACE_BIN" ]; then
    echo "✅ OpenFace已就绪"
    echo ""
    echo "下一步：开始数据预处理"
    echo ""
    echo "方式1: 完整预处理流程（推荐）"
    echo "   bash prep/preprocess_ubfc_complete.sh"
    echo ""
    echo "方式2: 分步预处理"
    echo "   提取landmarks后，可用: python3 preprocess_ubfc.py 生成h5"
    echo ""
    echo "使用OpenFace提取单个视频的landmarks示例:"
    echo "   $OPENFACE_BIN -f <视频文件> -out_dir <输出目录> -2Dfp"
    echo ""
    echo "💡 提示: 如果OpenFace安装在conda环境中，可以添加到PATH:"
    echo "   export PATH=\"$CONDA_ENV/local/bin:\$PATH\""
else
    echo "⚠️  OpenFace未完全配置，请完成安装后再进行预处理"
fi

echo ""
echo "============================================================"
echo "配置指南完成"
echo "============================================================"
