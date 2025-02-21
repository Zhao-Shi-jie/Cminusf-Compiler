#include "ast.h"
#include "mlir/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <iostream>
#include <memory>

using namespace llvm;
using namespace mlir;
using namespace cminusf;

static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "CminusF compiler\n");

    // 读取输入文件
    std::ifstream inputFile(InputFilename);
    if (!inputFile.is_open()) {
        errs() << "Error: could not open input file " << InputFilename << "\n";
        return 1;
    }

    std::string input((std::istreambuf_iterator<char>(inputFile)),
                      std::istreambuf_iterator<char>());

    // 解析输入文件为 AST
    syntax_tree *tree = parse(input.c_str());
    if (!tree) {
        errs() << "Error: failed to parse input file " << InputFilename << "\n";
        return 1;
    }

    // 创建 AST
    std::unique_ptr<ASTNode> ast = std::make_unique<AST>(tree);

    // 初始化 MLIR 上下文
    mlir::MLIRContext context;
    context.getOrLoadDialect<CminusfDialect>();

    // 生成 MLIR 模块
    mlir::ModuleOp module = mlirGen(context, std::move(ast));

    // 验证 MLIR 模块
    if (failed(mlir::verify(module))) {
        errs() << "Error: MLIR module verification failed\n";
        return 1;
    }

    // 输出 MLIR 模块
    module.print(outs());

    return 0;
}