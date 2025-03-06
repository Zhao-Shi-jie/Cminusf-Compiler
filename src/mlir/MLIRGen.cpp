#include "MLIRGen.h"
#include "Dialect.h"
#include "ast.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir::cminusf;
using namespace cminusf;

namespace {

class MLIRGenImpl {};

} // namespace

namespace cminusf {
mlir::ModuleOp mlirGen(mlir::MLIRContext &context, std::unique_ptr<ASTNode> root) {
    // return MLIRGenImpl(context).mlirGen(std::move(root));
}
} // namespace cminusf
