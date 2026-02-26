#!/bin/bash
# OpenFace安装步骤脚本（按照README和Wiki指示）
# 参考: https://github.com/TadasBaltrusaitis/OpenFace/wiki

# 不使用set -e，以便在错误时提供更好的错误信息
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENFACE_DIR="$PROJECT_DIR/OpenFace"

cd "$OPENFACE_DIR"

echo "============================================================"
echo "OpenFace 安装步骤（按照README指示）"
echo "============================================================"

# 步骤1: 下载模型文件（必须先下载）
echo "\n步骤1: 下载模型文件"
echo "="*60
if [ -f "download_models.sh" ]; then
    echo "运行 download_models.sh..."
    bash download_models.sh
    echo "✅ 模型文件下载完成"
else
    echo "❌ download_models.sh 不存在"
    exit 1
fi

# 步骤2: 检查依赖
echo "\n步骤2: 检查系统依赖"
echo "="*60

# 检查cmake
if ! command -v cmake &> /dev/null; then
    echo "❌ cmake未安装，需要安装"
    echo "运行: sudo apt-get install cmake"
    exit 1
else
    echo "✅ cmake已安装: $(cmake --version | head -1)"
fi

# 检查g++
if ! command -v g++ &> /dev/null; then
    echo "❌ g++未安装，需要安装"
    echo "运行: sudo apt-get install build-essential g++"
    exit 1
else
    echo "✅ g++已安装: $(g++ --version | head -1)"
fi

# 检查OpenCV（使用CMake检测，因为OpenCV 4.x不提供pkg-config）
if cmake --find-package -DNAME=OpenCV -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST &> /dev/null; then
    echo "✅ OpenCV已安装（通过CMake检测）"
    # 尝试获取版本
    OPENCV_VERSION=$(cmake --find-package -DNAME=OpenCV -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=VERSION 2>/dev/null || echo "未知版本")
    if [ "$OPENCV_VERSION" != "未知版本" ]; then
        echo "   版本: $OPENCV_VERSION"
    fi
else
    echo "⚠️  OpenCV未找到（通过CMake检测）"
    echo "OpenFace需要OpenCV，可能需要安装"
    echo "运行: bash prep/install_opencv_from_existing.sh"
fi

# 步骤3: 编译OpenFace
echo "\n步骤3: 编译OpenFace"
echo "="*60

if [ -d "build" ]; then
    echo "⚠️  build目录已存在"
    read -p "是否删除并重新编译? (y/n): " rebuild
    if [ "$rebuild" = "y" ]; then
        rm -rf build
    fi
fi

if [ ! -d "build" ]; then
    echo "创建build目录..."
    mkdir -p build
fi

cd build

echo "\n运行cmake配置..."

# 检查conda环境中的依赖
CONDA_ENV="/home/vt_ai_test1/mamba-envs/ml"
CMAKE_ARGS=()

# 检查OpenCV（按优先级：用户指定 > conda环境 > 系统默认）
OPENCV_CMAKE_PATH=""
OPENCV_FOUND=0

# 优先级1: 检查用户是否设置了OpenCV_DIR环境变量或提供了自定义路径
if [ -n "$OPENCV_DIR" ] && [ -f "$OPENCV_DIR/OpenCVConfig.cmake" ]; then
    OPENCV_CMAKE_PATH="$OPENCV_DIR"
    OPENCV_FOUND=1
    echo "✅ 找到用户指定的OpenCV: $OPENCV_CMAKE_PATH"
elif [ -n "$OPENCV_DIR" ] && [ -d "$OPENCV_DIR" ]; then
    # 尝试查找OpenCVConfig.cmake
    if [ -f "$OPENCV_DIR/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
        OPENCV_CMAKE_PATH="$OPENCV_DIR/lib/cmake/opencv4"
        OPENCV_FOUND=1
        echo "✅ 找到用户指定的OpenCV: $OPENCV_CMAKE_PATH"
    elif [ -f "$OPENCV_DIR/share/OpenCV/OpenCVConfig.cmake" ]; then
        OPENCV_CMAKE_PATH="$OPENCV_DIR/share/OpenCV"
        OPENCV_FOUND=1
        echo "✅ 找到用户指定的OpenCV: $OPENCV_CMAKE_PATH"
    fi
fi

# 优先级2: 检查conda环境中的OpenCV
if [ $OPENCV_FOUND -eq 0 ]; then
    OPENCV_CMAKE_PATH="$CONDA_ENV/lib/cmake/opencv4"
    if [ -f "$OPENCV_CMAKE_PATH/OpenCVConfig.cmake" ]; then
        OPENCV_FOUND=1
        echo "✅ 找到conda环境中的OpenCV: $OPENCV_CMAKE_PATH"
    fi
fi

# 优先级3: 检查项目目录下是否有手动放置的OpenCV
if [ $OPENCV_FOUND -eq 0 ]; then
    OPENCV_CUSTOM_PATHS=(
        "$PROJECT_DIR/opencv/build"
        "$PROJECT_DIR/opencv"
        "$PROJECT_DIR/OpenCV"
    )
    for custom_path in "${OPENCV_CUSTOM_PATHS[@]}"; do
        if [ -f "$custom_path/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
            OPENCV_CMAKE_PATH="$custom_path/lib/cmake/opencv4"
            OPENCV_FOUND=1
            echo "✅ 找到项目目录中的OpenCV: $OPENCV_CMAKE_PATH"
            break
        elif [ -f "$custom_path/share/OpenCV/OpenCVConfig.cmake" ]; then
            OPENCV_CMAKE_PATH="$custom_path/share/OpenCV"
            OPENCV_FOUND=1
            echo "✅ 找到项目目录中的OpenCV: $OPENCV_CMAKE_PATH"
            break
        fi
    done
fi

# 设置OpenCV路径
if [ $OPENCV_FOUND -eq 1 ]; then
    export CMAKE_PREFIX_PATH="$(dirname $(dirname $OPENCV_CMAKE_PATH)):$CMAKE_PREFIX_PATH"
    export OpenCV_DIR="$OPENCV_CMAKE_PATH"
    CMAKE_ARGS+=(-D CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" -D OpenCV_DIR="$OpenCV_DIR")
else
    echo "⚠️  未找到OpenCV"
    echo ""
    echo "如果您手动提供了OpenCV，请："
    echo "1. 将OpenCV解压到项目目录（如 opencv/ 或 OpenCV/）"
    echo "2. 或设置环境变量: export OPENCV_DIR=/path/to/opencv"
    echo "3. 或将其安装到conda环境: conda install -c conda-forge opencv"
    echo ""
    echo "OpenCV目录结构应该是："
    echo "  opencv/"
    echo "    lib/cmake/opencv4/OpenCVConfig.cmake"
    echo "    或"
    echo "    share/OpenCV/OpenCVConfig.cmake"
    echo ""
    echo "将尝试使用系统默认路径（可能失败）"
fi

# 检查Boost
BOOST_CMAKE_PATH="$CONDA_ENV/lib/cmake/Boost"
if [ -d "$BOOST_CMAKE_PATH" ]; then
    echo "✅ 找到conda环境中的Boost: $BOOST_CMAKE_PATH"
    CMAKE_ARGS+=(-D Boost_DIR="$BOOST_CMAKE_PATH")
elif [ -d "$CONDA_ENV/share/boost" ]; then
    echo "✅ 找到conda环境中的Boost (share路径)"
    CMAKE_ARGS+=(-D Boost_ROOT="$CONDA_ENV")
fi

# 检查dlib（优先检查本地编译的）
LOCAL_PREFIX="$PROJECT_DIR/local"
DLIB_CMAKE_PATH="$LOCAL_PREFIX/lib/cmake/dlib"
if [ -f "$DLIB_CMAKE_PATH/dlibConfig.cmake" ] || [ -f "$DLIB_CMAKE_PATH/dlib-config.cmake" ]; then
    echo "✅ 找到本地编译的dlib: $DLIB_CMAKE_PATH"
    CMAKE_ARGS+=(-D dlib_DIR="$DLIB_CMAKE_PATH")
    CMAKE_ARGS+=(-D CMAKE_PREFIX_PATH="$LOCAL_PREFIX:$CMAKE_PREFIX_PATH")
elif [ -d "$CONDA_ENV/lib/cmake/dlib" ]; then
    echo "✅ 找到conda环境中的dlib"
    CMAKE_ARGS+=(-D dlib_DIR="$CONDA_ENV/lib/cmake/dlib")
else
    echo "⚠️  未找到dlib，需要先编译安装"
    echo "运行: bash prep/install_dlib.sh"
fi

# 检查GCC 8（优先使用conda环境中的）
CONDA_GCC="$CONDA_ENV/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="$CONDA_ENV/bin/x86_64-conda-linux-gnu-g++"
if [ -f "$CONDA_GCC" ] && [ -f "$CONDA_GXX" ]; then
    GCC_VERSION=$("$CONDA_GCC" --version | head -1)
    echo "✅ 使用conda环境中的GCC: $GCC_VERSION"
    CMAKE_ARGS+=(-D CMAKE_C_COMPILER="$CONDA_GCC" -D CMAKE_CXX_COMPILER="$CONDA_GXX")
    
    # 设置库路径，让编译器使用conda的libstdc++而不是系统的
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$CONDA_ENV/lib:$LIBRARY_PATH"
    export CPLUS_INCLUDE_PATH="$CONDA_ENV/include:$CPLUS_INCLUDE_PATH"
    echo "设置库路径使用conda的libstdc++: $CONDA_ENV/lib"
elif command -v g++-8 &> /dev/null && command -v gcc-8 &> /dev/null; then
    echo "✅ 使用系统g++-8和gcc-8"
    CMAKE_ARGS+=(-D CMAKE_CXX_COMPILER=g++-8 -D CMAKE_C_COMPILER=gcc-8)
else
    echo "⚠️  使用系统默认g++（版本可能不满足要求）"
fi

# 设置安装路径到conda环境的local目录
CONDA_LOCAL="$CONDA_ENV/local"
CMAKE_ARGS+=(-D CMAKE_INSTALL_PREFIX="$CONDA_LOCAL")
echo "📦 OpenFace将安装到: $CONDA_LOCAL"
echo "   二进制文件: $CONDA_LOCAL/bin"
echo "   库文件: $CONDA_LOCAL/lib"
echo "   配置文件: $CONDA_LOCAL/etc"

# 运行cmake配置
echo "\nCMake参数: ${CMAKE_ARGS[@]}"
cmake -D CMAKE_BUILD_TYPE=RELEASE "${CMAKE_ARGS[@]}" ..

CMAKE_EXIT_CODE=$?
if [ $CMAKE_EXIT_CODE -ne 0 ]; then
    echo "\n❌ CMake配置失败！"
    echo "============================================================"
    echo "错误原因：通常是缺少OpenCV依赖"
    echo "============================================================"
    echo "\n解决方案（选择其一）："
    echo "\n方案1（推荐）：使用OpenFace自带的install.sh自动安装所有依赖"
    echo "   cd $OPENFACE_DIR"
    echo "   bash install.sh"
    echo "   注意：这会下载并编译OpenCV 4.1.0，需要sudo权限和较长时间"
    echo "\n方案2：手动安装OpenCV开发包"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y libopencv-dev"
    echo "   注意：Ubuntu 18.04默认仓库可能只有OpenCV 3.x，可能不满足要求"
    echo "\n安装完依赖后，重新运行此脚本："
    echo "   bash prep/install_openface_step_by_step.sh"
    echo "============================================================"
    exit 1
fi

echo "\n开始编译（这可能需要一些时间）..."
make -j$(nproc)

MAKE_EXIT_CODE=$?
if [ $MAKE_EXIT_CODE -ne 0 ]; then
    echo "\n❌ 编译失败！"
    echo "请检查上面的错误信息"
    exit 1
fi

echo "\n✅ 编译完成！"
echo "\n安装OpenFace到conda环境 ($CONDA_LOCAL)..."
make install

INSTALL_EXIT_CODE=$?
if [ $INSTALL_EXIT_CODE -ne 0 ]; then
    echo "\n⚠️  安装失败，但编译成功"
    echo "可以手动安装: cd $OPENFACE_DIR/build && make install"
    # 即使安装失败，也检查build目录中的二进制文件
    BIN_PATH="$OPENFACE_DIR/build/bin/FeatureExtraction"
else
    echo "\n✅ OpenFace已安装到conda环境！"
    # 检查安装后的二进制文件
    BIN_PATH="$CONDA_LOCAL/bin/FeatureExtraction"
fi

# 步骤4: 验证安装
echo "\n步骤4: 验证安装"
echo "="*60

if [ -f "$BIN_PATH" ]; then
    echo "✅ FeatureExtraction工具已找到"
    echo "路径: $BIN_PATH"
    echo "\n测试运行:"
    "$BIN_PATH" || echo "（可能需要视频文件才能完整测试）"
    
    if [ -f "$CONDA_LOCAL/bin/FeatureExtraction" ]; then
        echo "\n📦 OpenFace已安装到conda环境:"
        echo "   二进制文件: $CONDA_LOCAL/bin/FeatureExtraction"
        echo "   使用方式:"
        echo "   export PATH=\"$CONDA_LOCAL/bin:\$PATH\""
        echo "   或者: $CONDA_LOCAL/bin/FeatureExtraction -f <video> -out_dir <dir> -2Dfp"
    fi
else
    echo "❌ FeatureExtraction未找到"
    echo "检查路径: $BIN_PATH"
    exit 1
fi

echo "\n============================================================"
echo "OpenFace安装完成！"
echo "============================================================"
echo "\n📦 安装位置: $CONDA_LOCAL"
echo "   所有文件都在conda环境中，便于管理"
echo "\n下一步："
echo "1. 使用FeatureExtraction提取landmarks:"
if [ -f "$CONDA_LOCAL/bin/FeatureExtraction" ]; then
    echo "   $CONDA_LOCAL/bin/FeatureExtraction -f <video> -out_dir <dir> -2Dfp"
    echo "   或者添加到PATH后直接使用:"
    echo "   export PATH=\"$CONDA_LOCAL/bin:\$PATH\""
    echo "   FeatureExtraction -f <video> -out_dir <dir> -2Dfp"
else
    echo "   $OPENFACE_DIR/build/bin/FeatureExtraction -f <video> -out_dir <dir> -2Dfp"
fi
echo "\n2. 运行预处理脚本生成h5文件:"
echo "   bash prep/preprocess_ubfc_complete.sh"
echo "\n💡 提示: 可以将以下内容添加到 ~/.bashrc 以便永久使用:"
echo "   export PATH=\"$CONDA_LOCAL/bin:\$PATH\""
echo "   export LD_LIBRARY_PATH=\"$CONDA_LOCAL/lib:\$LD_LIBRARY_PATH\""