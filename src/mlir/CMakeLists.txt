add_library(mlir
    Dialect.cpp
    MLIRGen.cpp
)
add_dependencies(mlir OpsIncGen)
# 将 mlir 库链接到 LLVM 和 MLIR 库
target_link_libraries(mlir
    # LLVM 组件
    ${llvm_libs}
        
    # MLIR 组件
    MLIRAnalysis
    MLIRDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms

    # 其他依赖库
    common
)

# 包含 MLIR 头文件
target_include_directories(mlir PUBLIC
    ${PROJECT_SOURCE_DIR}/include/mlir
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include/ 
)

# 安装配置
install(
    TARGETS mlir
    ARCHIVE DESTINATION lib
)
