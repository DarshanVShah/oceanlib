#!/usr/bin/env bash
# Cross-compiles the test suite for aarch64 and runs it under QEMU user-mode
# emulation, to exercise the NEON kernel without needing physical ARM hardware.
#
# Run this from inside a Linux environment with the aarch64 cross toolchain
# and qemu-user-static installed (on Ubuntu/Debian:
#   sudo apt-get install g++-aarch64-linux-gnu qemu-user-static cmake ninja-build
# On Windows this means running it inside WSL2, not in PowerShell/cmd.
#
# See ARCHITECTURE.md ADR-018 for what this does and does not prove.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build-aarch64-qemu"

command -v aarch64-linux-gnu-g++ >/dev/null || {
    echo "error: aarch64-linux-gnu-g++ not found. Install g++-aarch64-linux-gnu." >&2
    exit 1
}
command -v qemu-aarch64-static >/dev/null || {
    echo "error: qemu-aarch64-static not found. Install qemu-user-static." >&2
    exit 1
}

TOOLCHAIN_FILE="${BUILD_DIR}.toolchain.cmake"
mkdir -p "$(dirname "${TOOLCHAIN_FILE}")"
cat > "${TOOLCHAIN_FILE}" <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_CROSSCOMPILING_EMULATOR qemu-aarch64-static)
EOF

# Statically linked: qemu-aarch64-static needs the aarch64 dynamic linker
# (/lib/ld-linux-aarch64.so.1), which does not exist on an x86_64 host unless
# a full aarch64 sysroot is installed. Linking statically sidesteps that
# entirely rather than requiring a sysroot just to run a test binary once.
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS=-static \
    -DOCEAN_BUILD_TESTS=ON -DOCEAN_BUILD_BENCH=OFF \
    -DOCEAN_BUILD_EXAMPLES=OFF -DOCEAN_BUILD_VIEWER=OFF

cmake --build "${BUILD_DIR}" -j"$(nproc)"

echo
echo "=== running the aarch64 test binary under qemu-aarch64-static ==="
echo
qemu-aarch64-static "${BUILD_DIR}/tests/ocean_tests" "$@"
