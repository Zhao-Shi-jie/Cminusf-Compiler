#ifndef CMINUSF_MLIRGEN_H
#define CMINUSF_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace cminusf {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace cminusf

#endif // CMINUSF_MLIRGEN_H