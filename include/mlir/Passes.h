#ifndef CMINUSF_MLIR_PASSES_H_
#define CMINUSF_MLIR_PASSES_H_

#include <memory>

namespace mlir {
class Pass;

namespace cminusf {

std::unique_ptr<Pass> createLowerCminusfToStandardPass();
std::unique_ptr<Pass> createLowerStandardToLLVMDialectPass();
std::unique_ptr<Pass> createPrintCminusfOpCountPass();
std::unique_ptr<Pass> createCminusfConstantPropagationPass();
std::unique_ptr<Pass> createCminusfConstantFoldingPass();
void registerCminusfPasses();

} // namespace cminusf
} // namespace mlir

#endif // CMINUSF_MLIR_PASSES_H_
