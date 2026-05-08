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
    MLIRCallInterfaces
    MLIRDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
    MLIRSupport

    # 其他依赖库
    common
)

add_library(cminusf_passes
    LoweringPasses.cpp
)
add_dependencies(cminusf_passes OpsIncGen)
target_link_libraries(cminusf_passes
    mlir
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRControlFlowDialect
    MLIRFuncToLLVM
    MLIRFuncDialect
    MLIRIndexToLLVM
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefToLLVM
    MLIRMemRefDialect
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRSCFDialect
    MLIRPass
    MLIRIR
    MLIRTransforms
    MLIRSupport
    ${llvm_libs}
)
target_include_directories(cminusf_passes PUBLIC
    ${PROJECT_SOURCE_DIR}/include/mlir
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include/
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
