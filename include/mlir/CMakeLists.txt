set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(CminusfOps.h.inc -gen-op-decls)
mlir_tablegen(CminusfOps.cpp.inc -gen-op-defs)
mlir_tablegen(CminusfDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CminusfDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(OpsIncGen)

# 设置 MLIR TableGen 生成规则
# set(LLVM_TARGET_DEFINITIONS Ops.td)

# # 生成方言声明（必须指定 -dialect）
# mlir_tablegen(CminusfDialect.h.inc -gen-dialect-decls -dialect=cminusf)
# mlir_tablegen(CminusfDialect.cpp.inc -gen-dialect-defs -dialect=cminusf)

# # 生成操作声明和定义（必须指定 -dialect）
# mlir_tablegen(CminusfOps.h.inc -gen-op-decls -dialect=cminusf)
# mlir_tablegen(CminusfOps.cpp.inc -gen-op-defs -dialect=cminusf)

# # 将生成文件添加到公共目标
# add_public_tablegen_target(OpsIncGen)