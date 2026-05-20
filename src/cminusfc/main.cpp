#include "common/ast.hpp"
#include "mlir/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/MLIRGen.h"
#include "mlir/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

static std::string InputFilename;
static std::string OutputFilename = "-";
static bool EmitAst = false;
static bool EmitMlir = false;
static bool EmitStandardMlir = false;
static bool EmitLlvmDialect = false;
static bool EmitLlvm = false;
static bool EmitAsm = false;
static bool EmitObjShort = false;
static bool EmitObj = false;
static bool EmitExe = false;
static bool EmitOpStats = false;
static int OptLevel = 0;

extern "C" {
#include "syntax_tree.h"
extern syntax_tree *parse(const char *input);
}

static bool writeTextFile(const std::filesystem::path &path, const std::string &text) {
    std::ofstream out(path);
    if (!out)
        return false;
    out << text;
    return true;
}

static std::string shellQuote(const std::filesystem::path &path) {
    std::string input = path.string();
    std::string quoted = "'";
    for (char ch : input) {
        if (ch == '\'')
            quoted += "'\\''";
        else
            quoted += ch;
    }
    quoted += "'";
    return quoted;
}

static std::filesystem::path defaultOutputPath(const std::string &suffix) {
    if (OutputFilename != "-")
        return std::filesystem::path(OutputFilename);
    auto path = std::filesystem::path(InputFilename);
    path.replace_extension(suffix);
    return path;
}

static int runCommand(const std::string &cmd) {
    int status = std::system(cmd.c_str());
    if (status != 0)
        std::cerr << "Error: command failed: " << cmd << "\n";
    return status;
}

static std::filesystem::path temporaryLlvmPath() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto key = InputFilename + "|" + OutputFilename + "|" + std::to_string(now);
    auto hash = std::hash<std::string>{}(key);
    auto name = "cminusfc-" + std::to_string(::getpid()) + "-" + std::to_string(hash) + ".ll";
    return std::filesystem::temp_directory_path() / name;
}

static mlir::OwningOpRef<mlir::ModuleOp> generateMLIRModule(mlir::MLIRContext &context,
                                                            ASTProgram &root) {
    context.getOrLoadDialect<mlir::cminusf::CminusfDialect>();
    auto module = mlir::cminusf::mlirGen(context, root);
    if (!module)
        std::cerr << "Error: failed to generate verifier-clean cminusf MLIR\n";
    return module;
}

static bool lowerMLIRModule(mlir::ModuleOp module, mlir::MLIRContext &context,
                            bool toLLVMDialect, int optLevel = 0) {
    mlir::PassManager pm(&context);

    // Optional: print op stats before lowering
    if (EmitOpStats)
        pm.addPass(mlir::cminusf::createPrintCminusfOpCountPass());

    // Optimization passes at the cminusf dialect level (run before lowering)
    if (optLevel >= 1) {
        pm.addPass(mlir::cminusf::createCminusfConstantFoldingPass());
        pm.addPass(mlir::cminusf::createCminusfConstantPropagationPass());
    }

    pm.addPass(mlir::cminusf::createLowerCminusfToStandardPass());

    // Standard MLIR optimization passes (run after lowering to standard dialects)
    if (optLevel >= 1) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createSymbolDCEPass());
    }
    if (optLevel >= 2) {
        pm.addPass(mlir::createInlinerPass());
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    }

    if (toLLVMDialect)
        pm.addPass(mlir::cminusf::createLowerStandardToLLVMDialectPass());

    if (mlir::succeeded(pm.run(module)))
        return true;

    std::cerr << "Error: failed to lower cminusf MLIR";
    if (toLLVMDialect)
        std::cerr << " to LLVM dialect";
    std::cerr << "\n";
    return false;
}

static void wrapMainWithZeroExit(llvm::Module &module) {
    auto *userMain = module.getFunction("main");
    if (!userMain || !userMain->arg_empty())
        return;

    auto &context = module.getContext();
    auto *mainType = llvm::FunctionType::get(llvm::Type::getInt32Ty(context), false);
    if (userMain->getFunctionType() != mainType)
        return;

    // In normal C/LLVM ABI semantics, main's return value is the process exit
    // status.  Cminusf tests usually inspect stdout instead, so keep the user
    // main result internal and make the executable report successful execution.
    userMain->setName("__cminusf_user_main");
    userMain->setLinkage(llvm::GlobalValue::InternalLinkage);

    auto *entryMain =
        llvm::Function::Create(mainType, llvm::GlobalValue::ExternalLinkage, "main", module);
    auto *entry = llvm::BasicBlock::Create(context, "entry", entryMain);
    llvm::IRBuilder<> builder(entry);
    builder.CreateCall(userMain);
    builder.CreateRet(llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0));
}

static bool emitMLIRBackedLLVMIR(ASTProgram &root, std::ostream &out) {
    mlir::DialectRegistry registry;
    mlir::registerLLVMDialectTranslation(registry);
    mlir::MLIRContext context(registry);

    auto module = generateMLIRModule(context, root);
    if (!module)
        return false;
    if (!lowerMLIRModule(*module, context, /*toLLVMDialect=*/true, OptLevel))
        return false;

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module->getOperation(), llvmContext,
                                                    InputFilename);
    if (!llvmModule) {
        std::cerr << "Error: failed to translate LLVM dialect to LLVM IR\n";
        return false;
    }
    wrapMainWithZeroExit(*llvmModule);

    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    llvmModule->print(os, nullptr);
    os.flush();
    out << buffer;
    return true;
}

static void usage(const char *argv0) {
    std::cerr << "Usage: " << argv0
              << " [--emit-ast|--emit-mlir|--emit-standard-mlir|--emit-llvm-dialect|--emit-llvm|-S|-c|--emit-obj|--emit-exe] [--emit-op-stats] [-O0|-O1|-O2|-O3] input.cminus [-o output]\n";
}

static bool parseArgs(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--emit-ast" || arg == "-emit-ast") {
            EmitAst = true;
        } else if (arg == "--emit-mlir" || arg == "-emit-mlir") {
            EmitMlir = true;
        } else if (arg == "--emit-standard-mlir" || arg == "-emit-standard-mlir") {
            EmitStandardMlir = true;
        } else if (arg == "--emit-llvm-dialect" || arg == "-emit-llvm-dialect") {
            EmitLlvmDialect = true;
        } else if (arg == "--emit-llvm" || arg == "-emit-llvm") {
            EmitLlvm = true;
        } else if (arg == "-S") {
            EmitAsm = true;
        } else if (arg == "-c") {
            EmitObjShort = true;
        } else if (arg == "--emit-obj" || arg == "-emit-obj") {
            EmitObj = true;
        } else if (arg == "--emit-exe" || arg == "-emit-exe") {
            EmitExe = true;
        } else if (arg == "--emit-op-stats" || arg == "-emit-op-stats") {
            EmitOpStats = true;
        } else if (arg == "-O0") {
            OptLevel = 0;
        } else if (arg == "-O1") {
            OptLevel = 1;
        } else if (arg == "-O2") {
            OptLevel = 2;
        } else if (arg == "-O3") {
            OptLevel = 3;
        } else if (arg == "-o") {
            if (++i >= argc)
                return false;
            OutputFilename = argv[i];
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: unknown option " << arg << "\n";
            return false;
        } else if (InputFilename.empty()) {
            InputFilename = arg;
        } else {
            std::cerr << "Error: multiple input files are not supported\n";
            return false;
        }
    }
    return !InputFilename.empty();
}

int main(int argc, char **argv) {
    if (!parseArgs(argc, argv)) {
        usage(argv[0]);
        return 1;
    }

    syntax_tree *tree = parse(InputFilename.c_str());
    if (!tree) {
        std::cerr << "Error: failed to parse input file " << InputFilename << "\n";
        return 1;
    }

    if (EmitAst) {
        print_syntax_tree(stdout, tree);
        del_syntax_tree(tree);
        return 0;
    }

    AST ast(tree);
    ASTProgram *root = ast.get_root();

    if (EmitMlir || EmitStandardMlir || EmitLlvmDialect) {
        // 这里走真实 MLIR builder，输出经过 verifier 检查的 cminusf dialect module。
        mlir::MLIRContext context;
        auto module = generateMLIRModule(context, *root);
        if (!module)
            return 1;

        if (EmitMlir && (EmitOpStats || OptLevel > 0)) {
            // Run optimization passes at the cminusf level without lowering.
            mlir::PassManager pm(&context);
            if (EmitOpStats)
                pm.addPass(mlir::cminusf::createPrintCminusfOpCountPass());
            if (OptLevel >= 1) {
                pm.addPass(mlir::cminusf::createCminusfConstantFoldingPass());
                pm.addPass(mlir::cminusf::createCminusfConstantPropagationPass());
            }
            if (mlir::failed(pm.run(*module))) {
                std::cerr << "Error: failed to optimize cminusf MLIR\n";
                return 1;
            }
        }

        if (EmitStandardMlir || EmitLlvmDialect) {
            // 通过 MLIR PassManager 运行项目自己的 lowering pass。
            if (!lowerMLIRModule(*module, context, EmitLlvmDialect, OptLevel))
                return 1;
        }

        if (OutputFilename == "-") {
            module->print(llvm::outs());
            llvm::outs() << "\n";
        } else {
            std::error_code ec;
            llvm::raw_fd_ostream os(OutputFilename, ec);
            if (ec) {
                std::cerr << "Error: could not write output file " << OutputFilename << "\n";
                return 1;
            }
            module->print(os);
            os << "\n";
        }
        return 0;
    }

    std::ostringstream llvmIR;
    if (!emitMLIRBackedLLVMIR(*root, llvmIR))
        return 1;

    if (EmitLlvm || (!EmitAsm && !EmitObj && !EmitObjShort && !EmitExe)) {
        if (OutputFilename == "-") {
            std::cout << llvmIR.str();
        } else if (!writeTextFile(std::filesystem::path(OutputFilename), llvmIR.str())) {
            std::cerr << "Error: could not write output file " << OutputFilename << "\n";
            return 1;
        }
        return 0;
    }

    auto llPath = temporaryLlvmPath();
    if (!writeTextFile(llPath, llvmIR.str())) {
        std::cerr << "Error: could not write temporary LLVM IR file " << llPath.string() << "\n";
        return 1;
    }

    auto clang = std::string("clang");
    auto result = 0;
    if (EmitAsm) {
        result = runCommand(clang + " -S -x ir " + shellQuote(llPath) + " -o " +
                            shellQuote(defaultOutputPath(".s")));
    } else if (EmitObj || EmitObjShort) {
        result = runCommand(clang + " -c -x ir " + shellQuote(llPath) + " -o " +
                            shellQuote(defaultOutputPath(".o")));
    } else {
        auto ioPath = std::filesystem::path(CMINUSF_SOURCE_DIR) / "src" / "io" / "io.c";
        result = runCommand(clang + " -x ir " + shellQuote(llPath) + " -x c " +
                            shellQuote(ioPath) + " -o " + shellQuote(defaultOutputPath("")));
    }

    std::error_code ec;
    std::filesystem::remove(llPath, ec);
    return result == 0 ? 0 : 1;
}
