#!/bin/bash
# 在用户目录下编译安装dlib（不需要sudo权限）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DLIB_VERSION="19.13"
DLIB_DIR="$PROJECT_DIR/dlib-$DLIB_VERSION"
INSTALL_PREFIX="$PROJECT_DIR/local"

cd "$PROJECT_DIR"

echo "============================================================"
echo "编译安装 dlib $DLIB_VERSION"
echo "============================================================"

# 检查是否已下载
if [ ! -d "$DLIB_DIR" ]; then
    echo "下载 dlib $DLIB_VERSION..."
    wget http://dlib.net/files/dlib-$DLIB_VERSION.tar.bz2
    tar xf dlib-$DLIB_VERSION.tar.bz2
    rm dlib-$DLIB_VERSION.tar.bz2
else
    echo "✅ dlib 源码已存在"
fi

cd "$DLIB_DIR"

# 创建build目录
mkdir -p build
cd build

# 使用conda环境中的GCC 8
CONDA_ENV="/home/vt_ai_test1/mamba-envs/ml"
CONDA_GCC="$CONDA_ENV/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="$CONDA_ENV/bin/x86_64-conda-linux-gnu-g++"

if [ -f "$CONDA_GCC" ] && [ -f "$CONDA_GXX" ]; then
    echo "使用conda环境中的GCC 8编译dlib"
    cmake -D CMAKE_C_COMPILER="$CONDA_GCC" \
          -D CMAKE_CXX_COMPILER="$CONDA_GXX" \
          -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
          ..
else
    echo "使用系统默认编译器"
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
          ..
fi

echo "开始编译dlib（这可能需要几分钟）..."
make -j$(nproc)

echo "安装dlib到 $INSTALL_PREFIX"
make install

echo "============================================================"
echo "✅ dlib 安装完成！"
echo "安装路径: $INSTALL_PREFIX"
echo "============================================================"
echo "\n下一步：更新OpenFace的CMake配置以使用这个dlib"
echo "运行: bash prep/install_openface_step_by_step.sh"
