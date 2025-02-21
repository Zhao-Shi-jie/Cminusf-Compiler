# 创建 mlir 库，包含 Dialect.cpp 和 MLIRGen.cpp
add_library(mlir
    Dialect.cpp
    MLIRGen.cpp
)

# 将 mlir 库链接到 LLVM 和 MLIR 库
target_link_libraries(mlir
    LLVM
    MLIR
    common  # 如果需要链接 common 库
)

# 包含 MLIR 头文件
target_include_directories(mlir PUBLIC
    ${PROJECT_SOURCE_DIR}/include/mlir
    ${CMAKE_CURRENT_BINARY_DIR}/include/mlir
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

# 如果需要 TableGen 目标
add_public_tablegen_target(OpsIncGen)

# 安装配置
install(
    TARGETS mlir
    ARCHIVE DESTINATION lib
)
