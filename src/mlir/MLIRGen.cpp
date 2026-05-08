#include "MLIRGen.h"
#include "Dialect.h"
#include "ast.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <vector>

using namespace mlir::cminusf;

namespace {

struct Symbol {
    mlir::Value ref;
    mlir::Type elementType;
    bool isArray = false;
};

class MLIRGenImpl {
  public:
    explicit MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(ASTProgram &program) {
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
        theModule = module;

        builder.setInsertionPointToEnd(module.getBody());
        declareRuntimeFunctions();

        for (auto &decl : program.declarations) {
            // MLIR/LLVM is built with RTTI disabled in this project, so this
            // path uses ASTKind instead of dynamic_cast/dynamic_pointer_cast.
            if (decl->getKind() == ASTKind::VarDeclaration)
                emitGlobal(static_cast<ASTVarDeclaration &>(*decl));
            else if (decl->getKind() == ASTKind::FunDeclaration)
                emitFunction(static_cast<ASTFunDeclaration &>(*decl));
        }

        if (failed(mlir::verify(module)))
            return nullptr;
        return module;
    }

  private:
    mlir::ModuleOp theModule;
    mlir::OpBuilder builder;
    llvm::ScopedHashTable<llvm::StringRef, Symbol> symbols;
    mlir::Type currentReturnType;

    mlir::Location loc() { return builder.getUnknownLoc(); }

    mlir::Type scalarType(CminusType type) {
        if (type == TYPE_FLOAT)
            return builder.getF32Type();
        if (type == TYPE_VOID)
            return {};
        return builder.getI32Type();
    }

    mlir::MemRefType scalarRefType(mlir::Type elementType) {
        // Cminusf variables are l-values.  Represent every scalar variable as a
        // one-element memref so load/store lowering can be uniform.
        return mlir::MemRefType::get({1}, elementType);
    }

    mlir::Type arrayRefType(mlir::Type elementType, int64_t size) {
        return mlir::MemRefType::get({size}, elementType);
    }

    mlir::Type arrayParamType(mlir::Type elementType) {
        // Array parameters decay to a dynamically-sized memref reference.
        return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
    }

    void declareRuntimeFunctions() {
        // Runtime declarations are normal cminusf functions.  Calls verify by
        // looking up these symbols before the user program is lowered further.
        createFunctionDecl("input", {}, {builder.getI32Type()}, /*withBody=*/false);
        createFunctionDecl("output", {builder.getI32Type()}, {}, /*withBody=*/false);
        createFunctionDecl("outputFloat", {builder.getF32Type()}, {}, /*withBody=*/false);
        createFunctionDecl("neg_idx_except", {}, {}, /*withBody=*/false);
    }

    FunDeclOp createFunctionDecl(llvm::StringRef name, llvm::ArrayRef<mlir::Type> inputs,
                                 llvm::ArrayRef<mlir::Type> results, bool withBody) {
        auto type = builder.getFunctionType(inputs, results);
        auto func = builder.create<FunDeclOp>(loc(), name, type);
        if (!withBody)
            return func;

        auto *entry = new mlir::Block();
        for (mlir::Type input : inputs)
            entry->addArgument(input, loc());
        func.getBody().push_back(entry);
        return func;
    }

    void emitGlobal(ASTVarDeclaration &decl) {
        auto elementType = scalarType(decl.type);
        auto type = decl.num ? arrayRefType(elementType, decl.num->i_val) : scalarRefType(elementType);
        builder.create<GlobalOp>(loc(), decl.id, type);
    }

    void emitFunction(ASTFunDeclaration &decl) {
        std::vector<mlir::Type> inputs;
        for (auto &param : decl.params) {
            auto elementType = scalarType(param->type);
            inputs.push_back(param->isarray ? arrayParamType(elementType) : elementType);
        }

        std::vector<mlir::Type> results;
        auto returnType = scalarType(decl.type);
        if (decl.id == "main" && !returnType)
            returnType = builder.getI32Type();
        if (returnType)
            results.push_back(returnType);

        auto func = createFunctionDecl(decl.id, inputs, results, /*withBody=*/true);
        currentReturnType = returnType;

        mlir::OpBuilder::InsertionGuard guard(builder);
        auto &entry = func.getBody().front();
        builder.setInsertionPointToStart(&entry);

        llvm::ScopedHashTableScope<llvm::StringRef, Symbol> functionScope(symbols);
        for (size_t i = 0; i < decl.params.size(); ++i) {
            auto &param = decl.params[i];
            auto elementType = scalarType(param->type);
            auto arg = entry.getArgument(i);

            if (param->isarray) {
                // Array parameters are already references; no local var op is needed.
                symbols.insert(param->id, {arg, elementType, true});
            } else {
                // Scalar parameters are copied into a local l-value slot.
                auto slot = builder.create<VarOp>(loc(), scalarRefType(elementType),
                                                  builder.getStringAttr(param->id));
                builder.create<StoreOp>(loc(), arg, slot.getResult());
                symbols.insert(param->id, {slot.getResult(), elementType, false});
            }
        }

        if (decl.compound_stmt)
            emitCompound(*decl.compound_stmt);

        if (entry.empty() || !entry.back().hasTrait<mlir::OpTrait::IsTerminator>())
            emitDefaultReturn();
    }

    void emitCompound(ASTCompoundStmt &stmt) {
        llvm::ScopedHashTableScope<llvm::StringRef, Symbol> scope(symbols);
        for (auto &decl : stmt.local_declarations)
            emitLocal(*decl);
        for (auto &statement : stmt.statement_list)
            emitStatement(*statement);
    }

    void emitLocal(ASTVarDeclaration &decl) {
        auto elementType = scalarType(decl.type);
        auto refType = decl.num ? arrayRefType(elementType, decl.num->i_val) : scalarRefType(elementType);
        auto var = builder.create<VarOp>(loc(), refType, builder.getStringAttr(decl.id));
        symbols.insert(decl.id, {var.getResult(), elementType, decl.num != nullptr});
    }

    void emitStatement(ASTStatement &stmt) {
        if (stmt.getKind() == ASTKind::ExpressionStmt) {
            auto *exprStmt = static_cast<ASTExpressionStmt *>(&stmt);
            if (exprStmt->expression)
                emitExpression(*exprStmt->expression);
            return;
        }
        if (stmt.getKind() == ASTKind::CompoundStmt) {
            emitCompound(static_cast<ASTCompoundStmt &>(stmt));
            return;
        }
        if (stmt.getKind() == ASTKind::ReturnStmt) {
            auto *ret = static_cast<ASTReturnStmt *>(&stmt);
            if (ret->expression) {
                auto value = castValue(emitExpression(*ret->expression), currentReturnType);
                builder.create<ReturnOp>(loc(), value);
            } else {
                emitDefaultReturn();
            }
            return;
        }
        if (stmt.getKind() == ASTKind::SelectionStmt) {
            auto *sel = static_cast<ASTSelectionStmt *>(&stmt);
            auto cond = castValue(emitExpression(*sel->expression), builder.getI32Type());
            auto ifOp = builder.create<IfOp>(loc(), cond);
            fillRegion(ifOp.getThenBranch(), [&] { emitStatement(*sel->if_statement); });
            fillRegion(ifOp.getElseBranch(), [&] {
                if (sel->else_statement)
                    emitStatement(*sel->else_statement);
            });
            return;
        }
        if (stmt.getKind() == ASTKind::IterationStmt) {
            auto *iter = static_cast<ASTIterationStmt *>(&stmt);
            auto whileOp = builder.create<WhileOp>(loc());
            fillRegion(whileOp.getCondition(), [&] {
                auto cond = castValue(emitExpression(*iter->expression), builder.getI32Type());
                builder.create<YieldOp>(loc(), cond);
            });
            fillRegion(whileOp.getBody(), [&] { emitStatement(*iter->statement); });
        }
    }

    template <typename BodyBuilder>
    void fillRegion(mlir::Region &region, BodyBuilder bodyBuilder) {
        mlir::Block &block = region.empty() ? region.emplaceBlock() : region.front();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&block);
        bodyBuilder();
        if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
            builder.create<YieldOp>(loc());
    }

    void emitDefaultReturn() {
        if (!currentReturnType) {
            builder.create<ReturnOp>(loc(), nullptr);
        } else if (currentReturnType.isF32()) {
            auto zero = builder.create<ConstantOp>(loc(), 0.0f);
            builder.create<ReturnOp>(loc(), zero.getResult());
        } else {
            auto zero = builder.create<ConstantOp>(loc(), 0);
            builder.create<ReturnOp>(loc(), zero.getResult());
        }
    }

    mlir::Value emitExpression(ASTExpression &expr) {
        if (expr.getKind() == ASTKind::AssignExpression) {
            auto *assign = static_cast<ASTAssignExpression *>(&expr);
            auto lhs = emitVarRef(*assign->var);
            auto rhs = castValue(emitExpression(*assign->expression), lhs.elementType);
            builder.create<StoreOp>(loc(), rhs, lhs.ref);
            return rhs;
        }
        return emitSimple(*static_cast<ASTSimpleExpression *>(&expr));
    }

    mlir::Value emitSimple(ASTSimpleExpression &expr) {
        auto lhs = emitAdditive(*expr.additive_expression_l);
        if (!expr.additive_expression_r)
            return lhs;
        auto rhs = emitAdditive(*expr.additive_expression_r);
        auto common = commonType(lhs.getType(), rhs.getType());
        lhs = castValue(lhs, common);
        rhs = castValue(rhs, common);
        return builder.create<CmpOp>(loc(), lhs, rhs, relName(expr.op)).getResult();
    }

    mlir::Value emitAdditive(ASTAdditiveExpression &expr) {
        auto rhs = emitTerm(*expr.term);
        if (!expr.additive_expression)
            return rhs;
        auto lhs = emitAdditive(*expr.additive_expression);
        auto common = commonType(lhs.getType(), rhs.getType());
        lhs = castValue(lhs, common);
        rhs = castValue(rhs, common);
        return builder
            .create<BinaryOp>(loc(), expr.op == OP_PLUS ? "add" : "sub", lhs, rhs)
            .getResult();
    }

    mlir::Value emitTerm(ASTTerm &term) {
        auto rhs = emitFactor(*term.factor);
        if (!term.term)
            return rhs;
        auto lhs = emitTerm(*term.term);
        auto common = commonType(lhs.getType(), rhs.getType());
        lhs = castValue(lhs, common);
        rhs = castValue(rhs, common);
        return builder
            .create<BinaryOp>(loc(), term.op == OP_MUL ? "mul" : "div", lhs, rhs)
            .getResult();
    }

    mlir::Value emitFactor(ASTFactor &factor) {
        if (factor.getKind() == ASTKind::Num) {
            auto *num = static_cast<ASTNum *>(&factor);
            if (num->type == TYPE_FLOAT)
                return builder.create<ConstantOp>(loc(), num->f_val).getResult();
            return builder.create<ConstantOp>(loc(), num->i_val).getResult();
        }
        if (factor.getKind() == ASTKind::Var) {
            auto ref = emitVarRef(static_cast<ASTVar &>(factor));
            return builder.create<LoadOp>(loc(), ref.elementType, ref.ref).getResult();
        }
        if (factor.getKind() == ASTKind::Call)
            return emitCall(static_cast<ASTCall &>(factor));
        return emitExpression(static_cast<ASTExpression &>(factor));
    }

    Symbol emitVarRef(ASTVar &var) {
        Symbol symbol;
        if (symbols.count(var.id)) {
            symbol = symbols.lookup(var.id);
        } else {
            auto global = theModule.lookupSymbol<GlobalOp>(var.id);
            auto refType = global ? global.getType()
                                  : scalarRefType(builder.getI32Type());
            auto element = refType.cast<mlir::MemRefType>().getElementType();
            auto varOp = builder.create<VarOp>(loc(), refType, builder.getStringAttr(var.id));
            // Global identifiers are l-value references, not declarations of a
            // fresh local slot.  Mark them so the lowering pass emits
            // memref.get_global instead of memref.alloc.
            if (global)
                varOp->setAttr("global_ref", builder.getUnitAttr());
            symbol = {varOp, element, true};
        }

        if (!var.expression)
            return symbol;

        auto index = castValue(emitExpression(*var.expression), builder.getI32Type());
        auto sub = builder.create<SubscriptOp>(loc(), scalarRefType(symbol.elementType), symbol.ref, index);
        return {sub.getResult(), symbol.elementType, false};
    }

    mlir::Value emitCall(ASTCall &call) {
        std::vector<mlir::Value> args;
        auto callee = theModule.lookupSymbol<FunDeclOp>(call.id);
        auto inputTypes = callee ? callee.getFunctionType().getInputs() : llvm::ArrayRef<mlir::Type>{};

        for (size_t i = 0; i < call.args.size(); ++i) {
            if (i < inputTypes.size() && inputTypes[i].isa<mlir::MemRefType>()) {
                if (auto *var = extractVariable(*call.args[i])) {
                    args.push_back(emitVarRef(*var).ref);
                    continue;
                }
            }
            args.push_back(emitExpression(*call.args[i]));
        }

        auto callOp = builder.create<CallOp>(loc(), call.id, args);
        if (callOp.getNumResults() == 0)
            return {};
        return callOp.getResult();
    }

    mlir::Value castValue(mlir::Value value, mlir::Type targetType) {
        if (!value || value.getType() == targetType)
            return value;
        // Keep the source dialect small: mixed numeric expressions are marked
        // with MLIR's builtin cast placeholder, then materialized as concrete
        // arith casts by the cminusf-to-standard lowering pass.
        return builder.create<mlir::UnrealizedConversionCastOp>(loc(), targetType, value)
            .getResult(0);
    }

    mlir::Type commonType(mlir::Type lhs, mlir::Type rhs) {
        if (lhs.isF32() || rhs.isF32())
            return builder.getF32Type();
        return builder.getI32Type();
    }

    static ASTVar *extractVariable(ASTExpression &expr) {
        if (expr.getKind() != ASTKind::SimpleExpression)
            return nullptr;
        auto *simple = static_cast<ASTSimpleExpression *>(&expr);
        if (simple->additive_expression_r)
            return nullptr;
        auto *add = simple->additive_expression_l.get();
        if (!add || add->additive_expression)
            return nullptr;
        auto *term = add->term.get();
        if (!term || term->term)
            return nullptr;
        if (term->factor->getKind() != ASTKind::Var)
            return nullptr;
        return static_cast<ASTVar *>(term->factor.get());
    }

    static llvm::StringRef relName(RelOp op) {
        switch (op) {
        case OP_LE:
            return "le";
        case OP_LT:
            return "lt";
        case OP_GT:
            return "gt";
        case OP_GE:
            return "ge";
        case OP_EQ:
            return "eq";
        case OP_NEQ:
            return "ne";
        }
        return "eq";
    }
};

} // namespace

namespace mlir {
namespace cminusf {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ASTProgram &root) {
    MLIRGenImpl impl(context);
    return impl.mlirGen(root);
}

} // namespace cminusf
} // namespace mlir
