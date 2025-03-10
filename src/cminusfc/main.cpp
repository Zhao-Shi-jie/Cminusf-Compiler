#include "ast.hpp"
#include "mlir/Dialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/MLIRGen.h"
#include "mlir/Parser/Parser.h"
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
#include <string>

using namespace llvm;
using namespace mlir;
using namespace cminusf;

// 命令行选项定义
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

static cl::opt<bool> EmitAst("emit-ast", cl::desc("Emit AST"), cl::init(false));
static cl::opt<bool> EmitMlir("emit-mlir", cl::desc("Emit MLIR"), cl::init(false));
static cl::opt<bool> EmitLlvm("emit-llvm", cl::desc("Emit LLVM IR"), cl::init(false));
static cl::opt<bool> EmitObj("emit-obj", cl::desc("Emit object file"), cl::init(false));
static cl::opt<bool> RunJit("run-jit", cl::desc("Run using JIT"), cl::init(false));
static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
                                           cl::init("-"));

// 前向声明
extern "C" {
#include "syntax_tree.h"
extern syntax_tree *parse(const char *input);
}

// 声明 MLIRGen 函数 (从 MLIRGen.cpp)
namespace cminusf {
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, std::unique_ptr<ASTNode> root);
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "CminusF compiler\n");

    // 读取输入文件
    std::ifstream inputFile(InputFilename);
    if (!inputFile.is_open()) {
        errs() << "Error: could not open input file " << InputFilename << "\n";
        return 1;
    }

    std::string input((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
    inputFile.close();

    // 解析输入文件为语法树
    syntax_tree *tree = parse(input.c_str());
    if (!tree) {
        errs() << "Error: failed to parse input file " << InputFilename << "\n";
        return 1;
    }

    // 创建 AST
    auto ast = std::make_unique<AST>(tree);

    // 如果只需要输出 AST，则在这里处理
    if (EmitAst) {
        // 简单的 AST 打印实现 - 这里直接使用语法树的打印函数
        // outs() << "AST for " << InputFilename << ":\n";
        // print_syntax_tree(outs().os(), tree);
        // del_syntax_tree(tree);
        // return 0;
    }

    // 获取根节点
    ASTProgram *root = ast->get_root();
    auto rootNode = std::unique_ptr<ASTNode>(static_cast<ASTNode *>(root));

    // 初始化 MLIR 上下文
    mlir::MLIRContext context;
    context.getOrLoadDialect<CminusfDialect>();

    // 生成 MLIR 模块
    auto moduleRef = mlirGen(context, std::move(rootNode));

    if (!moduleRef) {
        errs() << "Error: failed to generate MLIR module\n";
        return 1;
    }

    mlir::ModuleOp module = moduleRef.get();

    // 验证 MLIR 模块
    if (failed(mlir::verify(module))) {
        errs() << "Error: MLIR module verification failed\n";
        return 1;
    }

    // 确定输出目标
    std::error_code EC;
    raw_ostream *OS = &outs();
    std::unique_ptr<raw_fd_ostream> outputFile;

    if (OutputFilename != "-") {
        outputFile = std::make_unique<raw_fd_ostream>(OutputFilename, EC, sys::fs::OF_None);
        if (EC) {
            errs() << "Error: could not open output file " << OutputFilename << ": " << EC.message() << "\n";
            return 1;
        }
        OS = outputFile.get();
    }

    // 如果需要输出 MLIR 或默认情况
    if (EmitMlir || (!EmitLlvm && !EmitObj && !RunJit)) {
        module.print(*OS);
        return 0;
    }

    // 创建 PassManager
    mlir::PassManager pm(&context);

    // 添加一些转换和优化 Pass
    pm.addPass(mlir::createCanonicalizerPass());

    // 执行 PassManager
    if (failed(pm.run(module))) {
        errs() << "Error: Pass pipeline failed\n";
        return 1;
    }

    // 如果需要生成 LLVM IR
    if (EmitLlvm) {
        // 在实际项目中，这里应该添加 MLIR 到 LLVM IR 的转换代码
        *OS << "// LLVM IR for " << InputFilename << "\n";
        *OS << "// Note: MLIR to LLVM IR conversion not fully implemented yet\n";
        // 输出 LLVM IR
        module.print(*OS);
        return 0;
    }

    // 如果需要生成目标文件
    if (EmitObj) {
        // 在实际项目中，这里应该添加 MLIR 到目标代码的转换代码
        errs() << "Generating object file: " << OutputFilename << "\n";
        errs() << "Note: MLIR to object code conversion not fully implemented yet\n";
        // 输出目标文件
        module.print(*OS);
        return 0;
    }

    // 如果需要使用 JIT 运行
    if (RunJit) {
        errs() << "Running using JIT...\n";
        errs() << "Note: JIT execution not fully implemented yet\n";
        // 执行 JIT 代码
        return 0;
    }

    // 默认情况下输出 MLIR
    module.print(*OS);

    return 0;
}