#include "Dialect.h"
#include "ast.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::cminusf;

namespace {

class MLIRGenImpl {};

} // namespace

ModuleOp mlirGen(MLIRContext &context, std::unique_ptr<ASTNode> root) {
    // return MLIRGenImpl(context).mlirGen(std::move(root));
}