// #include "MLIRGen.h"
// #include "Dialect.h"
// #include "ast.hpp"

// #include "mlir/IR/Builders.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/IR/MLIRContext.h"
// #include "mlir/IR/Verifier.h"

// using namespace mlir::cminusf;
// using namespace cminusf;

// namespace {

// class MLIRGenImpl {};

// } // namespace

// namespace cminusf {
// mlir::ModuleOp mlirGen(mlir::MLIRContext &context, std::unique_ptr<ASTNode> root) {
//     // return MLIRGenImpl(context).mlirGen(std::move(root));
// }
// } // namespace cminusf

#include "MLIRGen.h"
#include "Dialect.h"
#include "ast.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <memory>

using namespace mlir::cminusf;
// using namespace cminusf;

namespace {

/// Implementation of a simple MLIR emission from the C-minus-f AST.
class MLIRGenImpl {
  public:
    MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    /// Public API: convert the AST for a C-minus-f module to an MLIR Module operation.
    mlir::ModuleOp mlirGen(std::unique_ptr<ASTNode> &root) {
        // Create an empty MLIR module
        theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

        // Process the AST root node
        if (auto *program = dynamic_cast<ASTProgram *>(root.get())) {
            processProgram(*program);
        } else {
            llvm::errs() << "Error: Expected ASTProgram as root node\n";
            return nullptr;
        }

        // Verify the module after construction
        if (failed(mlir::verify(theModule))) {
            theModule.emitError("module verification error");
            return nullptr;
        }

        return theModule;
    }

  private:
    /// The module being constructed
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function
    mlir::OpBuilder builder;

    /// Symbol table for variable and function lookup
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

    /// Map C-minus-f type to MLIR type
    mlir::Type getMLIRType(CminusType type) {
        switch (type) {
        case TYPE_INT:
            return builder.getI32Type();
        case TYPE_FLOAT:
            return builder.getF32Type();
        case TYPE_VOID:
            return builder.getNoneType();
        default:
            llvm::errs() << "Unknown C-minus-f type\n";
            return nullptr;
        }
    }

    /// Process the program node (root of AST)
    void processProgram(ASTProgram &program) {
        // Set insertion point to the module body
        builder.setInsertionPointToEnd(theModule.getBody());

        // Process all declarations (variables and functions)
        for (auto &decl : program.declarations) {
            if (auto varDecl = std::dynamic_pointer_cast<ASTVarDeclaration>(decl)) {
                processGlobalVarDecl(*varDecl);
            } else if (auto funDecl = std::dynamic_pointer_cast<ASTFunDeclaration>(decl)) {
                processFunDecl(*funDecl);
            }
        }
    }

    /// Process global variable declaration
    void processGlobalVarDecl(ASTVarDeclaration &varDecl) {
        mlir::Type type = getMLIRType(varDecl.type);

        // Check if variable is an array
        if (varDecl.num) {
            // Create array variable
            mlir::StringAttr nameAttr = builder.getStringAttr(varDecl.id);
            mlir::IntegerAttr sizeAttr = builder.getI32IntegerAttr(varDecl.num->i_val);

            // Create var_decl operation for array
            builder.create<VarDeclOp>(builder.getUnknownLoc(), nameAttr, type, sizeAttr);
        } else {
            // Create scalar variable
            mlir::StringAttr nameAttr = builder.getStringAttr(varDecl.id);

            // Create var_decl operation for scalar
            builder.create<VarDeclOp>(builder.getUnknownLoc(), nameAttr, type);
        }
    }

    /// Process function declaration
    void processFunDecl(ASTFunDeclaration &funDecl) {
        // Create a function type based on return type and parameters
        std::vector<mlir::Type> paramTypes;
        for (auto &param : funDecl.params) {
            paramTypes.push_back(getMLIRType(param->type));
        }

        // Determine return type
        mlir::Type returnType = getMLIRType(funDecl.type);
        mlir::FunctionType funcType;

        if (funDecl.type == TYPE_VOID) {
            // Void function has no results
            funcType = builder.getFunctionType(paramTypes, std::nullopt);
        } else {
            // Function with a return value
            funcType = builder.getFunctionType(paramTypes, returnType);
        }

        // Create the function operation
        auto loc = builder.getUnknownLoc();
        auto funcOp = builder.create<FunDeclOp>(loc, funDecl.id, funcType);

        // Create entry block in function body
        auto &entryBlock = funcOp.getBody().emplaceBlock();
        builder.setInsertionPointToStart(&entryBlock);

        // Create a new scope for the function
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> functionScope(symbolTable);

        // Add parameters to symbol table
        for (size_t i = 0; i < funDecl.params.size(); i++) {
            auto &param = funDecl.params[i];
            auto paramType = getMLIRType(param->type);

            // Create function arguments
            auto arg = entryBlock.addArgument(paramType, loc);
            symbolTable.insert(param->id, arg);
        }

        // Process function body
        if (funDecl.compound_stmt) {
            processCompoundStmt(*funDecl.compound_stmt);
        }

        // Add implicit return if needed
        if (builder.getBlock()->empty() || !llvm::isa<ReturnOp>(builder.getBlock()->back())) {
            if (funDecl.type == TYPE_VOID) {
                builder.create<ReturnOp>(loc, nullptr);
            } else {
                // Non-void function should have explicit returns in all code paths
                llvm::errs() << "Warning: Non-void function '" << funDecl.id
                             << "' missing return statement\n";
                builder.create<ReturnOp>(loc, nullptr);
            }
        }
    }

    /// Process compound statement (block of code)
    void processCompoundStmt(ASTCompoundStmt &compoundStmt) {
        // Create a new scope for local variables
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> blockScope(symbolTable);

        // Process local variable declarations
        for (auto &varDecl : compoundStmt.local_declarations) {
            processLocalVarDecl(*varDecl);
        }

        // Process statements
        for (auto &stmt : compoundStmt.statement_list) {
            processStmt(*stmt);
        }
    }

    /// Process local variable declaration
    void processLocalVarDecl(ASTVarDeclaration &varDecl) {
        mlir::Type type = getMLIRType(varDecl.type);
        auto loc = builder.getUnknownLoc();

        // Create var_decl operation and register in symbol table
        if (varDecl.num) {
            // Array variable
            mlir::StringAttr nameAttr = builder.getStringAttr(varDecl.id);
            mlir::IntegerAttr sizeAttr = builder.getI32IntegerAttr(varDecl.num->i_val);

            auto varOp = builder.create<VarDeclOp>(loc, nameAttr, type, sizeAttr);
            symbolTable.insert(varDecl.id, varOp.getResult());
        } else {
            // Scalar variable
            mlir::StringAttr nameAttr = builder.getStringAttr(varDecl.id);

            auto varOp = builder.create<VarDeclOp>(loc, nameAttr, type);
            symbolTable.insert(varDecl.id, varOp.getResult());
        }
    }

    /// Process statement
    void processStmt(ASTStatement &stmt) {
        auto loc = builder.getUnknownLoc();

        if (auto *exprStmt = dynamic_cast<ASTExpressionStmt *>(&stmt)) {
            // Expression statement
            if (exprStmt->expression) {
                processExpression(*exprStmt->expression);
            }
        } else if (auto *selStmt = dynamic_cast<ASTSelectionStmt *>(&stmt)) {
            // Selection statement (if-else)
            mlir::Value condValue = processExpression(*selStmt->expression);

            // Create if operation with then and else regions
            auto ifOp = builder.create<IfOp>(
                loc, condValue,
                [&](mlir::OpBuilder &builder, mlir::Location loc) {
                    // Then branch
                    this->builder.setInsertionPoint(builder.getBlock(), builder.getBlock()->begin());
                    processStmt(*selStmt->if_statement);
                    builder.create<YieldOp>(loc);
                },
                [&](mlir::OpBuilder &builder, mlir::Location loc) {
                    // Else branch (if exists)
                    this->builder.setInsertionPoint(builder.getBlock(), builder.getBlock()->begin());
                    if (selStmt->else_statement) {
                        processStmt(*selStmt->else_statement);
                    }
                    builder.create<YieldOp>(loc);
                });

            // Reset insertion point after if operation
            builder.setInsertionPointAfter(ifOp);

        } else if (auto *iterStmt = dynamic_cast<ASTIterationStmt *>(&stmt)) {
            // Iteration statement (while)
            mlir::Value condValue = processExpression(*iterStmt->expression);

            // Create while operation
            auto whileOp =
                builder.create<WhileOp>(loc, condValue, [&](mlir::OpBuilder &builder, mlir::Location loc) {
                    // While body
                    this->builder.setInsertionPoint(builder.getBlock(), builder.getBlock()->begin());
                    processStmt(*iterStmt->statement);
                    builder.create<YieldOp>(loc);
                });

            // Reset insertion point after while operation
            builder.setInsertionPointAfter(whileOp);

        } else if (auto *returnStmt = dynamic_cast<ASTReturnStmt *>(&stmt)) {
            // Return statement
            if (returnStmt->expression) {
                // Return with value
                mlir::Value retVal = processExpression(*returnStmt->expression);
                builder.create<ReturnOp>(loc, retVal);
            } else {
                // Void return
                builder.create<ReturnOp>(loc, nullptr);
            }
        } else if (auto *compoundStmt = dynamic_cast<ASTCompoundStmt *>(&stmt)) {
            // Compound statement
            processCompoundStmt(*compoundStmt);
        } else {
            llvm::errs() << "Unknown statement type\n";
        }
    }

    /// Process expression and return the resulting value
    mlir::Value processExpression(ASTExpression &expr) {
        auto loc = builder.getUnknownLoc();

        if (auto *assignExpr = dynamic_cast<ASTAssignExpression *>(&expr)) {
            // Assignment expression
            mlir::Value lhsVar = processVar(*assignExpr->var);
            mlir::Value rhsExpr = processExpression(*assignExpr->expression);

            return builder.create<AssignOp>(loc, lhsVar, rhsExpr);

        } else if (auto *simpleExpr = dynamic_cast<ASTSimpleExpression *>(&expr)) {
            // Simple expression (comparison)

            // If only left side exists, just return it
            if (!simpleExpr->additive_expression_r) {
                return processAdditiveExpression(*simpleExpr->additive_expression_l);
            }

            // Process both sides of comparison
            mlir::Value lhs = processAdditiveExpression(*simpleExpr->additive_expression_l);
            mlir::Value rhs = processAdditiveExpression(*simpleExpr->additive_expression_r);

            // Map RelOp to CmpPredicate
            std::string predStr;
            switch (simpleExpr->op) {
            case OP_LE:
                predStr = "le";
                break;
            case OP_LT:
                predStr = "lt";
                break;
            case OP_GT:
                predStr = "gt";
                break;
            case OP_GE:
                predStr = "ge";
                break;
            case OP_EQ:
                predStr = "eq";
                break;
            case OP_NEQ:
                predStr = "ne";
                break;
            default:
                llvm::errs() << "Unknown comparison operator\n";
                break;
            }

            return builder.create<CmpOp>(loc, lhs, rhs, builder.getStringAttr(predStr));
        }

        // else if (auto *var = dynamic_cast<ASTVar *>(&expr)) {
        //     // Variable reference
        //     return processVar(*var);

        // } else if (auto *call = dynamic_cast<ASTCall *>(&expr)) {
        //     // Function call
        //     std::vector<mlir::Value> arguments;

        //     for (auto &arg : call->args) {
        //         arguments.push_back(processExpression(*arg));
        //     }

        //     return builder.create<CallOp>(loc, llvm::StringRef(call->id), arguments);
        // }

        llvm::errs() << "Unknown expression type\n";
        return nullptr;
    }

    /// Process variable reference and return the resulting value
    mlir::Value processVar(ASTVar &var) {
        // Look up variable in symbol table
        if (auto value = symbolTable.lookup(var.id)) {
            return value;
        } else {
            llvm::errs() << "Unknown variable: " << var.id << "\n";
            return nullptr;
        }
    }

    /// Process additive expression and return the resulting value
    mlir::Value processAdditiveExpression(ASTAdditiveExpression &addExpr) {
        auto loc = builder.getUnknownLoc();

        // Process term
        mlir::Value result = processTerm(*addExpr.term);

        // If there's no left side, just return the term
        if (!addExpr.additive_expression) {
            return result;
        }

        // Process left side (recursive)
        mlir::Value lhs = processAdditiveExpression(*addExpr.additive_expression);

        // Create binary operation
        std::string opStr;
        switch (addExpr.op) {
        case OP_PLUS:
            opStr = "add";
            break;
        case OP_MINUS:
            opStr = "sub";
            break;
        }

        return builder.create<BinaryOp>(loc, builder.getStringAttr(opStr), lhs, result);
    }

    /// Process term and return the resulting value
    mlir::Value processTerm(ASTTerm &term) {
        auto loc = builder.getUnknownLoc();

        // Process factor
        mlir::Value result = processFactor(*term.factor);

        // If there's no left side, just return the factor
        if (!term.term) {
            return result;
        }

        // Process left side (recursive)
        mlir::Value lhs = processTerm(*term.term);

        // Create binary operation
        std::string opStr;
        switch (term.op) {
        case OP_MUL:
            opStr = "mul";
            break;
        case OP_DIV:
            opStr = "div";
            break;
        }

        return builder.create<BinaryOp>(loc, builder.getStringAttr(opStr), lhs, result);
    }

    /// Process factor and return the resulting value
    mlir::Value processFactor(ASTFactor &factor) {
        auto loc = builder.getUnknownLoc();

        if (auto *num = dynamic_cast<ASTNum *>(&factor)) {
            // Numeric literal
            if (num->type == TYPE_INT) {
                return builder.create<ConstantOp>(loc, num->i_val);
            } else if (num->type == TYPE_FLOAT) {
                return builder.create<ConstantOp>(loc, num->f_val);
            }
        } else if (auto *var = dynamic_cast<ASTVar *>(&factor)) {
            // Variable reference
            return processVar(*var);
        } else if (auto *call = dynamic_cast<ASTCall *>(&factor)) {
            // Function call
            std::vector<mlir::Value> arguments;

            for (auto &arg : call->args) {
                arguments.push_back(processExpression(*arg));
            }

            return builder.create<CallOp>(loc, llvm::StringRef(call->id), arguments).getResult();
        } else if (auto *expr = dynamic_cast<ASTExpression *>(&factor)) {
            // Expression
            return processExpression(*expr);
        }

        llvm::errs() << "Unknown factor type\n";
        return nullptr;
    }
};

} // namespace

namespace mlir {
namespace cminusf {

/// The public API for generating MLIR from the C-minus-f AST
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, std::unique_ptr<ASTNode> root) {
    MLIRGenImpl impl(context);
    return impl.mlirGen(root);
}

} // namespace cminusf
} // namespace mlir