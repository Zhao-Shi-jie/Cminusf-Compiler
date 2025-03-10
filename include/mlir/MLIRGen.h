#ifndef CMINUSF_MLIRGEN_H
#define CMINUSF_MLIRGEN_H

#include "ast.hpp"
#include <cstddef>
#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;

namespace cminusf {
// struct ASTNode;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, std::unique_ptr<ASTNode> node);
} // namespace cminusf

} // namespace mlir

#endif // CMINUSF_MLIRGEN_H