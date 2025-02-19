#include "ast/AST.h"
#include "mlir/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
using namespace mlir::cminusf;

namespace {

class MLIRGenImpl {
  public:
    MLIRGenImpl(MLIRContext &context) : builder(&context) {}

    ModuleOp mlirGen(std::unique_ptr<ASTNode> root) {
        theModule = ModuleOp::create(builder.getUnknownLoc());
        mlirGenProgram(static_cast<ASTProgram *>(root.get()));
        return theModule;
    }

  private:
    void mlirGenProgram(ASTProgram *program) {
        for (auto &decl : program->declarations) {
            mlirGen(decl.get());
        }
    }

    void mlirGen(ASTNode *node) {
        if (auto *varDecl = dynamic_cast<ASTVarDeclaration *>(node)) {
            mlirGenVarDeclaration(varDecl);
        } else if (auto *funDecl = dynamic_cast<ASTFunDeclaration *>(node)) {
            mlirGenFunDeclaration(funDecl);
        } else if (auto *stmt = dynamic_cast<ASTStatement *>(node)) {
            mlirGenStatement(stmt);
        } else if (auto *expr = dynamic_cast<ASTExpression *>(node)) {
            mlirGenExpression(expr);
        }
    }

    void mlirGenVarDeclaration(ASTVarDeclaration *varDecl) {
        auto type = getType(varDecl->type);
        builder.create<Cminusf_VarDeclarationOp>(
            builder.getUnknownLoc(), type, varDecl->id,
            varDecl->num ? llvm::Optional<int32_t>(varDecl->num->i_val) : llvm::None);
    }

    void mlirGenFunDeclaration(ASTFunDeclaration *funDecl) {
        auto funcType = builder.getFunctionType({}, {builder.getI32Type()});
        auto funcOp = builder.create<Cminusf_FunDeclarationOp>(builder.getUnknownLoc(), funDecl->id,
                                                               funcType);
        auto &entryBlock = *funcOp.addEntryBlock();
        builder.setInsertionPointToStart(&entryBlock);

        for (auto &stmt : funDecl->compound_stmt->statement_list) {
            mlirGen(stmt.get());
        }

        builder.create<Cminusf_ReturnOp>(builder.getUnknownLoc(), llvm::None);
    }

    void mlirGenStatement(ASTStatement *stmt) {
        if (auto *exprStmt = dynamic_cast<ASTExpressionStmt *>(stmt)) {
            mlirGenExpressionStmt(exprStmt);
        } else if (auto *compStmt = dynamic_cast<ASTCompoundStmt *>(stmt)) {
            mlirGenCompoundStmt(compStmt);
        } else if (auto *selStmt = dynamic_cast<ASTSelectionStmt *>(stmt)) {
            mlirGenSelectionStmt(selStmt);
        } else if (auto *iterStmt = dynamic_cast<ASTIterationStmt *>(stmt)) {
            mlirGenIterationStmt(iterStmt);
        } else if (auto *retStmt = dynamic_cast<ASTReturnStmt *>(stmt)) {
            mlirGenReturnStmt(retStmt);
        }
    }

    void mlirGenExpressionStmt(ASTExpressionStmt *exprStmt) {
        if (exprStmt->expression) {
            auto exprValue = mlirGenExpression(exprStmt->expression.get());
            builder.create<Cminusf_ExpressionStmtOp>(builder.getUnknownLoc(), exprValue);
        }
    }

    void mlirGenCompoundStmt(ASTCompoundStmt *compStmt) {
        builder.create<Cminusf_CompoundStmtOp>(builder.getUnknownLoc());
        for (auto &localDecl : compStmt->local_declarations) {
            mlirGenVarDeclaration(localDecl.get());
        }
        for (auto &stmt : compStmt->statement_list) {
            mlirGen(stmt.get());
        }
    }

    void mlirGenSelectionStmt(ASTSelectionStmt *selStmt) {
        auto condValue = mlirGenExpression(selStmt->expression.get());
        auto thenValue = mlirGenStatement(selStmt->if_statement.get());
        Optional<Value> elseValue =
            selStmt->else_statement ? mlirGenStatement(selStmt->else_statement.get()) : llvm::None;
        builder.create<Cminusf_SelectionStmtOp>(builder.getUnknownLoc(), condValue, thenValue,
                                                elseValue);
    }

    void mlirGenIterationStmt(ASTIterationStmt *iterStmt) {
        auto condValue = mlirGenExpression(iterStmt->expression.get());
        auto bodyValue = mlirGenStatement(iterStmt->statement.get());
        builder.create<Cminusf_IterationStmtOp>(builder.getUnknownLoc(), condValue, bodyValue);
    }

    void mlirGenReturnStmt(ASTReturnStmt *retStmt) {
        Optional<Value> retValue =
            retStmt->expression ? mlirGenExpression(retStmt->expression.get()) : llvm::None;
        builder.create<Cminusf_ReturnOp>(builder.getUnknownLoc(), retValue);
    }

    Value mlirGenExpression(ASTExpression *expr) {
        if (auto *assignExpr = dynamic_cast<ASTAssignExpression *>(expr)) {
            return mlirGenAssignExpression(assignExpr);
        } else if (auto *simpleExpr = dynamic_cast<ASTSimpleExpression *>(expr)) {
            return mlirGenSimpleExpression(simpleExpr);
        } else if (auto *addExpr = dynamic_cast<ASTAdditiveExpression *>(expr)) {
            return mlirGenAdditiveExpression(addExpr);
        } else if (auto *term = dynamic_cast<ASTTerm *>(expr)) {
            return mlirGenTerm(term);
        } else if (auto *call = dynamic_cast<ASTCall *>(expr)) {
            return mlirGenCall(call);
        } else if (auto *var = dynamic_cast<ASTVar *>(expr)) {
            return mlirGenVar(var);
        } else if (auto *num = dynamic_cast<ASTNum *>(expr)) {
            return mlirGenNum(num);
        }
        return nullptr;
    }

    Value mlirGenAssignExpression(ASTAssignExpression *assignExpr) {
        auto lhs = mlirGenVar(assignExpr->var.get());
        auto rhs = mlirGenExpression(assignExpr->expression.get());
        return builder.create<Cminusf_AssignOp>(builder.getUnknownLoc(), lhs, rhs);
    }

    Value mlirGenSimpleExpression(ASTSimpleExpression *simpleExpr) {
        auto lhs = mlirGenExpression(simpleExpr->additive_expression_l.get());
        auto rhs = mlirGenExpression(simpleExpr->additive_expression_r.get());
        return builder.create<Cminusf_SimpleOp>(builder.getUnknownLoc(), lhs, rhs,
                                                static_cast<int32_t>(simpleExpr->op));
    }

    Value mlirGenAdditiveExpression(ASTAdditiveExpression *addExpr) {
        auto lhs = mlirGenExpression(addExpr->additive_expression.get());
        auto rhs = mlirGenExpression(addExpr->term.get());
        return builder.create<Cminusf_AdditiveOp>(builder.getUnknownLoc(), lhs, rhs,
                                                  static_cast<int32_t>(addExpr->op));
    }

    Value mlirGenTerm(ASTTerm *term) {
        auto lhs = mlirGenExpression(term->term.get());
        auto rhs = mlirGenExpression(term->factor.get());
        return builder.create<Cminusf_TermOp>(builder.getUnknownLoc(), lhs, rhs,
                                              static_cast<int32_t>(term->op));
    }

    Value mlirGenCall(ASTCall *call) {
        SmallVector<Value, 4> args;
        for (auto &arg : call->args) {
            args.push_back(mlirGenExpression(arg.get()));
        }
        return builder.create<Cminusf_CallOp>(builder.getUnknownLoc(), call->id, args);
    }

    Value mlirGenVar(ASTVar *var) {
        // Assume variables are already declared and can be looked up by name
        // You might need a symbol table or context to look up variables
        // For now, we return a dummy value
        return builder.create<Cminusf_ConstantOp>(builder.getUnknownLoc(), 0);
    }

    Value mlirGenNum(ASTNum *num) {
        if (num->type == TYPE_INT) {
            return builder.create<Cminusf_ConstantOp>(builder.getUnknownLoc(), num->i_val);
        } else {
            // Handle floating point numbers if necessary
            return builder.create<Cminusf_ConstantOp>(builder.getUnknownLoc(),
                                                      static_cast<int32_t>(num->f_val));
        }
    }

    Type getType(CminusType type) {
        switch (type) {
        case TYPE_INT:
            return builder.getIntegerType(32);
        case TYPE_FLOAT:
            return builder.getF32Type();
        case TYPE_VOID:
            return builder.getNoneType();
        }
        return builder.getNoneType();
    }

    MLIRContext &context;
    OpBuilder builder;
    ModuleOp theModule;
};

} // namespace

ModuleOp mlirGen(MLIRContext &context, std::unique_ptr<ASTNode> root) {
    return MLIRGenImpl(context).mlirGen(std::move(root));
}