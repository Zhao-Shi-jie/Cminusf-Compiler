#include "Passes.h"
#include "Dialect.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

namespace mlir {
namespace cminusf {
namespace {

static mlir::arith::CmpIPredicate toIntegerPredicate(CmpPredicate predicate) {
    switch (predicate) {
    case CmpPredicate::eq:
        return mlir::arith::CmpIPredicate::eq;
    case CmpPredicate::ne:
        return mlir::arith::CmpIPredicate::ne;
    case CmpPredicate::lt:
        return mlir::arith::CmpIPredicate::slt;
    case CmpPredicate::le:
        return mlir::arith::CmpIPredicate::sle;
    case CmpPredicate::gt:
        return mlir::arith::CmpIPredicate::sgt;
    case CmpPredicate::ge:
        return mlir::arith::CmpIPredicate::sge;
    }
    return mlir::arith::CmpIPredicate::eq;
}

static mlir::arith::CmpFPredicate toFloatPredicate(CmpPredicate predicate) {
    switch (predicate) {
    case CmpPredicate::eq:
        return mlir::arith::CmpFPredicate::OEQ;
    case CmpPredicate::ne:
        return mlir::arith::CmpFPredicate::ONE;
    case CmpPredicate::lt:
        return mlir::arith::CmpFPredicate::OLT;
    case CmpPredicate::le:
        return mlir::arith::CmpFPredicate::OLE;
    case CmpPredicate::gt:
        return mlir::arith::CmpFPredicate::OGT;
    case CmpPredicate::ge:
        return mlir::arith::CmpFPredicate::OGE;
    }
    return mlir::arith::CmpFPredicate::OEQ;
}

static Value castScalar(Location loc, Value value, Type targetType,
                        ConversionPatternRewriter &rewriter) {
    if (!value || value.getType() == targetType)
        return value;
    if (value.getType().isInteger(32) && targetType.isF32())
        return rewriter.create<mlir::arith::SIToFPOp>(loc, targetType, value);
    if (value.getType().isF32() && targetType.isInteger(32))
        return rewriter.create<mlir::arith::FPToSIOp>(loc, targetType, value);
    return value;
}

static Value normalizeCondition(Location loc, Value condition,
                                ConversionPatternRewriter &rewriter) {
    if (condition.getType().isInteger(1))
        return condition;
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(condition.getType(), 0));
    return rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, condition, zero);
}

static bool eraseUnreachableFunctionBlocks(ModuleOp module) {
    bool changed = false;
    module.walk([&](mlir::func::FuncOp func) {
        auto &body = func.getBody();
        if (body.empty())
            return;

        bool localChanged = true;
        while (localChanged) {
            localChanged = false;
            for (Block &block : llvm::make_early_inc_range(body)) {
                if (&block == &body.front() || !block.hasNoPredecessors())
                    continue;
                // Drop use-def edges before erasing an unreachable block.  This
                // mirrors MLIR's region cleanup utilities but keeps the pass
                // self-contained for the small CFGs generated here.
                block.dropAllDefinedValueUses();
                block.erase();
                changed = true;
                localChanged = true;
            }
        }
    });
    return changed;
}

static LogicalResult eraseDeadSubscripts(ModuleOp module) {
    bool hasLiveSubscript = false;
    module.walk([&](SubscriptOp op) {
        if (!op->use_empty()) {
            hasLiveSubscript = true;
            return;
        }
        op.erase();
    });
    return failure(hasLiveSubscript);
}

struct ConstantLowering : public OpConversionPattern<ConstantOp> {
    using OpConversionPattern<ConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ConstantOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto value = op.getValue();
        if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
            rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, intAttr);
            return success();
        }
        if (auto floatAttr = value.dyn_cast<FloatAttr>()) {
            rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, floatAttr);
            return success();
        }
        return failure();
    }
};

struct UnrealizedCastLowering : public OpConversionPattern<UnrealizedConversionCastOp> {
    using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (adaptor.getInputs().size() != 1 || op.getOutputs().size() != 1)
            return failure();
        auto casted = castScalar(op.getLoc(), adaptor.getInputs().front(),
                                 op.getOutputs().front().getType(), rewriter);
        if (casted.getType() != op.getOutputs().front().getType())
            return failure();
        rewriter.replaceOp(op, casted);
        return success();
    }
};

struct FunDeclLowering : public OpConversionPattern<FunDeclOp> {
    using OpConversionPattern<FunDeclOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(FunDeclOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getSymName(),
                                                        op.getFunctionType());
        if (op.getBody().empty())
            func.setPrivate();

        // Move the existing function body instead of rebuilding it so block
        // arguments, nested regions, and source operation order stay intact.
        rewriter.inlineRegionBefore(op.getBody(), func.getBody(), func.end());
        rewriter.eraseOp(op);
        return success();
    }
};

struct CallLowering : public OpConversionPattern<CallOp> {
    using OpConversionPattern<CallOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CallOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        SmallVector<Value> args(adaptor.getArgs());
        auto module = op->getParentOfType<ModuleOp>();
        FunctionType calleeType;
        if (auto func = module.lookupSymbol<FunDeclOp>(op.getCallee()))
            calleeType = func.getFunctionType();
        else if (auto func = module.lookupSymbol<mlir::func::FuncOp>(op.getCallee()))
            calleeType = func.getFunctionType();

        if (calleeType) {
            for (auto [index, expected] : llvm::enumerate(calleeType.getInputs())) {
                if (index >= args.size())
                    break;
                if (args[index].getType() == expected)
                    continue;
                if (args[index].getType().isa<MemRefType>() && expected.isa<MemRefType>())
                    args[index] = rewriter.create<mlir::memref::CastOp>(op.getLoc(), expected, args[index]);
            }
        }

        rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, op.getCallee(), op.getResultTypes(), args);
        return success();
    }
};

struct BinaryLowering : public OpConversionPattern<BinaryOp> {
    using OpConversionPattern<BinaryOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(BinaryOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();
        lhs = castScalar(op.getLoc(), lhs, op.getResult().getType(), rewriter);
        rhs = castScalar(op.getLoc(), rhs, op.getResult().getType(), rewriter);
        bool isFloat = lhs.getType().isa<FloatType>();

        // The custom dialect has one binary op.  The lowering selects the
        // concrete arithmetic op after type resolution has happened in MLIRGen.
        switch (op.getOpType()) {
        case BinaryOpType::add:
            if (isFloat)
                rewriter.replaceOpWithNewOp<mlir::arith::AddFOp>(op, lhs, rhs);
            else
                rewriter.replaceOpWithNewOp<mlir::arith::AddIOp>(op, lhs, rhs);
            return success();
        case BinaryOpType::sub:
            if (isFloat)
                rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(op, lhs, rhs);
            else
                rewriter.replaceOpWithNewOp<mlir::arith::SubIOp>(op, lhs, rhs);
            return success();
        case BinaryOpType::mul:
            if (isFloat)
                rewriter.replaceOpWithNewOp<mlir::arith::MulFOp>(op, lhs, rhs);
            else
                rewriter.replaceOpWithNewOp<mlir::arith::MulIOp>(op, lhs, rhs);
            return success();
        case BinaryOpType::div:
            if (isFloat)
                rewriter.replaceOpWithNewOp<mlir::arith::DivFOp>(op, lhs, rhs);
            else
                rewriter.replaceOpWithNewOp<mlir::arith::DivSIOp>(op, lhs, rhs);
            return success();
        }
        return failure();
    }
};

struct CmpLowering : public OpConversionPattern<CmpOp> {
    using OpConversionPattern<CmpOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto lhs = adaptor.getLhs();
        auto rhs = adaptor.getRhs();
        Type commonType = op.getLhs().getType().isF32() || op.getRhs().getType().isF32()
                              ? Type(rewriter.getF32Type())
                              : Type(rewriter.getI32Type());
        lhs = castScalar(op.getLoc(), lhs, commonType, rewriter);
        rhs = castScalar(op.getLoc(), rhs, commonType, rewriter);
        Value cmp;
        if (lhs.getType().isa<FloatType>()) {
            cmp = rewriter.create<mlir::arith::CmpFOp>(op.getLoc(),
                                                       toFloatPredicate(op.getPredicate()), lhs, rhs);
        } else {
            cmp = rewriter.create<mlir::arith::CmpIOp>(op.getLoc(),
                                                       toIntegerPredicate(op.getPredicate()), lhs, rhs);
        }

        // Cminusf conditions are i32.  Standard arith comparisons produce i1,
        // so extend the predicate result back to the source language shape.
        rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(op, rewriter.getI32Type(), cmp);
        return success();
    }
};

struct VarLowering : public OpConversionPattern<VarOp> {
    using OpConversionPattern<VarOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(VarOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto type = op.getResult().getType().dyn_cast<MemRefType>();
        if (!type)
            return failure();
        if (op->hasAttr("global_ref")) {
            // cminusf.var normally declares a local slot, but global references
            // are represented with the same ref-shaped value in the source
            // dialect.  Preserve that distinction when entering standard MLIR.
            rewriter.replaceOpWithNewOp<mlir::memref::GetGlobalOp>(op, type, op.getVarName());
            return success();
        }
        rewriter.replaceOpWithNewOp<mlir::memref::AllocOp>(op, type);
        return success();
    }
};

struct GlobalLowering : public OpConversionPattern<GlobalOp> {
    using OpConversionPattern<GlobalOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(GlobalOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        auto type = op.getType().cast<MemRefType>();
        auto zero = rewriter.getZeroAttr(type.getElementType());
        auto tensorType = RankedTensorType::get(type.getShape(), type.getElementType());
        Attribute initial = DenseElementsAttr::get(tensorType, zero);

        // Keep globals in the memref dialect first; the standard memref-to-LLVM
        // pipeline can later translate them into LLVM globals.
        rewriter.replaceOpWithNewOp<mlir::memref::GlobalOp>(
            op, op.getSymName(), rewriter.getStringAttr("private"), type, initial,
            /*constant=*/false, IntegerAttr{});
        return success();
    }
};

struct SubscriptLowering : public OpConversionPattern<SubscriptOp> {
    using OpConversionPattern<SubscriptOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(SubscriptOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (!op->use_empty())
            return failure();
        rewriter.eraseOp(op);
        return success();
    }
};

struct LoadLowering : public OpConversionPattern<LoadOp> {
    using OpConversionPattern<LoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (auto subscript = op.getRef().getDefiningOp<SubscriptOp>()) {
            Value base = rewriter.getRemappedValue(subscript.getBase());
            Value index = rewriter.getRemappedValue(subscript.getIndex());
            if (!base || !index)
                return failure();
            if (!index.getType().isIndex())
                index = rewriter.create<mlir::arith::IndexCastOp>(
                    op.getLoc(), rewriter.getIndexType(), index);
            // Lower array element reads directly.  This avoids materializing a
            // one-element memref.subview that must later be expanded before
            // memref-to-LLVM conversion.
            rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, base, ValueRange{index});
            return success();
        }

        auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, adaptor.getRef(), ValueRange{zero});
        return success();
    }
};

struct StoreLowering : public OpConversionPattern<StoreOp> {
    using OpConversionPattern<StoreOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (auto subscript = op.getRef().getDefiningOp<SubscriptOp>()) {
            Value base = rewriter.getRemappedValue(subscript.getBase());
            Value index = rewriter.getRemappedValue(subscript.getIndex());
            if (!base || !index)
                return failure();
            if (!index.getType().isIndex())
                index = rewriter.create<mlir::arith::IndexCastOp>(
                    op.getLoc(), rewriter.getIndexType(), index);
            auto baseType = base.getType().cast<MemRefType>();
            auto value = castScalar(op.getLoc(), adaptor.getValue(), baseType.getElementType(),
                                    rewriter);
            // Lower array element writes directly for the same reason as the
            // load case above: direct indexed memref ops have robust LLVM
            // lowering and preserve the source array access clearly.
            rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, base,
                                                               ValueRange{index});
            return success();
        }

        auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        auto refType = adaptor.getRef().getType().cast<MemRefType>();
        auto value = castScalar(op.getLoc(), adaptor.getValue(), refType.getElementType(), rewriter);
        rewriter.replaceOpWithNewOp<mlir::memref::StoreOp>(op, value, adaptor.getRef(),
                                                           ValueRange{zero});
        return success();
    }
};

struct ReturnLowering : public OpConversionPattern<ReturnOp> {
    using OpConversionPattern<ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (!isa<mlir::func::FuncOp>(op->getBlock()->getParentOp()))
            return failure();
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

struct YieldLowering : public OpConversionPattern<YieldOp> {
    using OpConversionPattern<YieldOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (!isa<mlir::scf::IfOp>(op->getBlock()->getParentOp()))
            return failure();
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, adaptor.getOperands());
        return success();
    }
};

struct IfLowering : public OpConversionPattern<IfOp> {
    using OpConversionPattern<IfOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(IfOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (op->getParentOfType<WhileOp>() &&
            (!op.getThenBranch().getOps<YieldOp>().empty() ||
             !op.getElseBranch().getOps<YieldOp>().empty()))
            return failure();

        Value condition = normalizeCondition(op.getLoc(), adaptor.getCondition(), rewriter);

        if (!op.getThenBranch().getOps<ReturnOp>().empty() ||
            !op.getElseBranch().getOps<ReturnOp>().empty())
            return lowerToCFG(op, condition, rewriter);

        auto scfIf = rewriter.create<mlir::scf::IfOp>(op.getLoc(), TypeRange{}, condition,
                                                      /*addThenBlock=*/false,
                                                      /*addElseBlock=*/false);
        // Move regions wholesale so nested operations are still rewritten by
        // the conversion driver after the structured control-flow op is formed.
        rewriter.inlineRegionBefore(op.getThenBranch(), scfIf.getThenRegion(),
                                    scfIf.getThenRegion().end());
        rewriter.inlineRegionBefore(op.getElseBranch(), scfIf.getElseRegion(),
                                    scfIf.getElseRegion().end());
        rewriter.eraseOp(op);
        return success();
    }

  private:
    static LogicalResult lowerToCFG(IfOp op, Value condition,
                                    ConversionPatternRewriter &rewriter) {
        Block *sourceBlock = op->getBlock();
        Region *parentRegion = sourceBlock->getParent();
        if (!isa<mlir::func::FuncOp>(parentRegion->getParentOp()))
            return failure();

        auto afterIf = std::next(Block::iterator(op));
        Block *continueBlock = rewriter.splitBlock(sourceBlock, afterIf);
        Block *thenBlock = &op.getThenBranch().front();
        Block *elseBlock = &op.getElseBranch().front();

        // Move cminusf branch regions into the function CFG.  Region-local
        // yields become branches to the continuation block; cminusf.return is
        // left in place and then lowered to func.return by ReturnLowering.
        rewriter.inlineRegionBefore(op.getThenBranch(), *parentRegion,
                                    continueBlock->getIterator());
        rewriter.inlineRegionBefore(op.getElseBranch(), *parentRegion,
                                    continueBlock->getIterator());

        rewriteBranchTerminator(thenBlock, continueBlock, rewriter);
        rewriteBranchTerminator(elseBlock, continueBlock, rewriter);

        rewriter.setInsertionPoint(op);
        rewriter.create<mlir::cf::CondBranchOp>(op.getLoc(), condition, thenBlock,
                                                ValueRange{}, elseBlock, ValueRange{});
        rewriter.eraseOp(op);

        if (continueBlock->empty()) {
            rewriter.eraseBlock(continueBlock);
        } else {
            rewriter.setInsertionPointToEnd(continueBlock);
        }
        return success();
    }

    static void rewriteBranchTerminator(Block *block, Block *continueBlock,
                                        ConversionPatternRewriter &rewriter) {
        Operation *terminator = block->getTerminator();
        if (isa<ReturnOp>(terminator) || isa<mlir::func::ReturnOp>(terminator))
            return;
        if (auto yield = dyn_cast<YieldOp>(terminator)) {
            rewriter.setInsertionPoint(yield);
            rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), continueBlock);
            rewriter.eraseOp(yield);
        }
    }
};

struct WhileLowering : public OpConversionPattern<WhileOp> {
    using OpConversionPattern<WhileOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(WhileOp op, OpAdaptor,
                                  ConversionPatternRewriter &rewriter) const final {
        if (needsCFGLowering(op))
            return lowerToCFG(op, rewriter);

        auto scfWhile = rewriter.create<mlir::scf::WhileOp>(
            op.getLoc(), TypeRange{}, ValueRange{},
            [](OpBuilder &builder, Location loc, ValueRange) {
                // Build a verifier-valid shell first; the real condition body
                // is moved in below and this placeholder block is erased.
                auto condition = builder.create<mlir::arith::ConstantOp>(
                    loc, builder.getBoolAttr(false));
                builder.create<mlir::scf::ConditionOp>(loc, condition, ValueRange{});
            },
            [](OpBuilder &builder, Location loc, ValueRange) {
                builder.create<mlir::scf::YieldOp>(loc);
            });
        Block *placeholderBefore = &scfWhile.getBefore().front();
        Block *placeholderAfter = &scfWhile.getAfter().front();

        rewriter.inlineRegionBefore(op.getCondition(), scfWhile.getBefore(),
                                    scfWhile.getBefore().begin());
        rewriter.inlineRegionBefore(op.getBody(), scfWhile.getAfter(), scfWhile.getAfter().begin());

        auto conditionYield = cast<YieldOp>(scfWhile.getBefore().front().getTerminator());
        rewriter.setInsertionPoint(conditionYield);
        Value condition = conditionYield.getValue();
        condition = normalizeCondition(conditionYield.getLoc(), condition, rewriter);
        rewriter.create<mlir::scf::ConditionOp>(conditionYield.getLoc(), condition, ValueRange{});
        rewriter.eraseOp(conditionYield);

        auto bodyYield = dyn_cast<YieldOp>(scfWhile.getAfter().front().getTerminator());
        if (!bodyYield)
            return failure();
        rewriter.setInsertionPoint(bodyYield);
        rewriter.create<mlir::scf::YieldOp>(bodyYield.getLoc());
        rewriter.eraseOp(bodyYield);
        rewriter.eraseBlock(placeholderBefore);
        rewriter.eraseBlock(placeholderAfter);

        rewriter.eraseOp(op);
        return success();
    }

  private:
    static bool needsCFGLowering(WhileOp op) {
        // SCF loop regions cannot directly contain function returns, and a
        // nested cminusf.if inside a loop needs a function-level continuation
        // block when it is lowered to CFG.  Use CFG lowering for those cases.
        return !op.getCondition().getOps<ReturnOp>().empty() ||
               !op.getBody().getOps<ReturnOp>().empty() ||
               !op.getBody().getOps<IfOp>().empty();
    }

    static LogicalResult lowerToCFG(WhileOp op, ConversionPatternRewriter &rewriter) {
        Block *sourceBlock = op->getBlock();
        Region *parentRegion = sourceBlock->getParent();
        if (!isa<mlir::func::FuncOp>(parentRegion->getParentOp()))
            return failure();

        auto afterLoop = std::next(Block::iterator(op));
        Block *afterBlock = rewriter.splitBlock(sourceBlock, afterLoop);
        Block *conditionBlock = &op.getCondition().front();
        Block *bodyBlock = &op.getBody().front();

        // Splice the loop regions into the surrounding function CFG.  The
        // condition terminator becomes a conditional branch, and the body
        // terminator jumps back to the condition block.
        rewriter.inlineRegionBefore(op.getCondition(), *parentRegion,
                                    afterBlock->getIterator());
        rewriter.inlineRegionBefore(op.getBody(), *parentRegion, afterBlock->getIterator());

        auto conditionYield = cast<YieldOp>(conditionBlock->getTerminator());
        rewriter.setInsertionPoint(conditionYield);
        Value condition = normalizeCondition(conditionYield.getLoc(), conditionYield.getValue(),
                                             rewriter);
        rewriter.create<mlir::cf::CondBranchOp>(conditionYield.getLoc(), condition, bodyBlock,
                                                ValueRange{}, afterBlock, ValueRange{});
        rewriter.eraseOp(conditionYield);

        Operation *bodyTerminator = bodyBlock->getTerminator();
        if (auto yield = dyn_cast<YieldOp>(bodyTerminator)) {
            rewriter.setInsertionPoint(yield);
            rewriter.create<mlir::cf::BranchOp>(yield.getLoc(), conditionBlock);
            rewriter.eraseOp(yield);
        } else if (!isa<ReturnOp, mlir::func::ReturnOp>(bodyTerminator)) {
            return failure();
        }

        rewriter.setInsertionPoint(op);
        rewriter.create<mlir::cf::BranchOp>(op.getLoc(), conditionBlock);
        rewriter.eraseOp(op);
        return success();
    }
};

class LowerCminusfToStandardPass
    : public PassWrapper<LowerCminusfToStandardPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCminusfToStandardPass)

    StringRef getArgument() const final { return "lower-cminusf-to-standard"; }
    StringRef getDescription() const final {
        return "Lower the cminusf dialect to standard MLIR dialects";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<CminusfDialect, mlir::arith::ArithDialect,
                        mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
    }

    void runOnOperation() final {
        ModuleOp module = getOperation();

        ConversionTarget target(getContext());
        target.addLegalDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                               mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                               mlir::scf::SCFDialect>();
        target.addLegalOp<ModuleOp>();
        target.addIllegalOp<FunDeclOp, CallOp, GlobalOp, ConstantOp, BinaryOp, CmpOp, VarOp,
                            LoadOp, StoreOp, UnrealizedConversionCastOp>();
        // Subscript is a transient l-value marker.  Load/store lowering consumes
        // it into direct indexed memref operations; dead markers are erased
        // immediately after conversion.
        target.addLegalOp<SubscriptOp>();
        target.addDynamicallyLegalOp<IfOp>([](IfOp op) {
            // Keep only the loop-local structured cases that need more work.
            // Function-level conditionals, including early returns, are lowered
            // to CFG blocks by IfLowering.
            return op->getParentOfType<WhileOp>() &&
                   (!op.getThenBranch().getOps<YieldOp>().empty() ||
                    !op.getElseBranch().getOps<YieldOp>().empty());
        });
        target.addIllegalOp<WhileOp>();
        target.addDynamicallyLegalOp<YieldOp>(
            [](YieldOp op) { return !isa<mlir::scf::IfOp>(op->getBlock()->getParentOp()); });
        target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
            return !isa<mlir::func::FuncOp>(op->getBlock()->getParentOp());
        });

        RewritePatternSet patterns(&getContext());
        patterns.add<FunDeclLowering, CallLowering, GlobalLowering, ConstantLowering,
                     UnrealizedCastLowering,
                     BinaryLowering, CmpLowering, VarLowering, SubscriptLowering, LoadLowering,
                     StoreLowering, IfLowering, WhileLowering, YieldLowering, ReturnLowering>(
            &getContext());

        // The transient cminusf.subscript marker is kept legal while load/store
        // patterns consume it into direct indexed memref operations.
        if (failed(applyPartialConversion(module, target, std::move(patterns))) ||
            failed(eraseDeadSubscripts(module)))
            signalPassFailure();
    }
};

class LowerStandardToLLVMDialectPass
    : public PassWrapper<LowerStandardToLLVMDialectPass, OperationPass<ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStandardToLLVMDialectPass)

    StringRef getArgument() const final { return "lower-standard-to-llvm-dialect"; }
    StringRef getDescription() const final {
        return "Lower standard MLIR dialects produced by cminusf to the LLVM dialect";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    void runOnOperation() final {
        MLIRContext *context = &getContext();
        ModuleOp module = getOperation();

        eraseUnreachableFunctionBlocks(module);

        RewritePatternSet scfPatterns(context);
        populateSCFToControlFlowConversionPatterns(scfPatterns);
        ConversionTarget scfTarget(*context);
        scfTarget.addLegalDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                                  mlir::LLVM::LLVMDialect>();
        scfTarget.addLegalOp<ModuleOp>();
        scfTarget.addIllegalDialect<mlir::scf::SCFDialect>();
        if (failed(applyPartialConversion(module, scfTarget, std::move(scfPatterns)))) {
            module.emitError("failed to lower SCF to control-flow before LLVM conversion");
            signalPassFailure();
            return;
        }

        eraseUnreachableFunctionBlocks(module);

        LowerToLLVMOptions options(context);
        // Match the x86_64 host ABI this project targets through clang.  The
        // memref lowering below uses LLVM descriptors rather than bare pointers,
        // which keeps array parameters explicit and ABI-stable inside MLIR.
        options.overrideIndexBitwidth(64);
        options.useOpaquePointers = true;
        LLVMTypeConverter typeConverter(context, options);

        RewritePatternSet patterns(context);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);
        mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

        LLVMConversionTarget target(*context);
        target.addLegalOp<ModuleOp>();
        // These casts are inserted while block/function signatures are being
        // converted.  A reconciliation pass erases the no-op chains afterwards.
        target.addLegalOp<UnrealizedConversionCastOp>();

        if (failed(applyFullConversion(module, target, std::move(patterns)))) {
            module.emitError("failed to finalize standard MLIR to LLVM dialect");
            signalPassFailure();
            return;
        }

        module.walk([&](mlir::LLVM::LLVMFuncOp func) {
            if (!func.getBody().empty())
                return;
            // Func declarations must be private before FuncToLLVM conversion,
            // but the resulting LLVM declarations should be external so clang
            // can resolve them from src/io/io.c or libc.
            if (func.getSymName() == "input" || func.getSymName() == "output" ||
                func.getSymName() == "outputFloat" || func.getSymName() == "neg_idx_except")
                func->removeAttr(SymbolTable::getVisibilityAttrName());
            func.setLinkage(mlir::LLVM::Linkage::External);
        });

        RewritePatternSet castPatterns(context);
        populateReconcileUnrealizedCastsPatterns(castPatterns);
        if (failed(applyPatternsAndFoldGreedily(module, std::move(castPatterns)))) {
            module.emitError("failed to reconcile LLVM conversion casts");
            signalPassFailure();
        }
    }
};

struct PrintCminusfOpCountPass
    : public PassWrapper<PrintCminusfOpCountPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintCminusfOpCountPass)

  StringRef getArgument() const final { return "print-cminusf-op-count"; }
  StringRef getDescription() const final {
    return "Print the count of each cminusf dialect operation in the module";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    llvm::StringMap<unsigned> opCounts;

    module.walk([&](Operation *op) {
      if (op->getDialect() ==
          module.getContext()->getLoadedDialect<CminusfDialect>())
        opCounts[op->getName().getStringRef()]++;
    });

    if (opCounts.empty()) {
      llvm::outs() << "No cminusf operations found.\n";
      return;
    }

    llvm::outs() << "Cminusf op counts for module:\n";
    for (const auto &entry : opCounts)
      llvm::outs() << "  " << entry.first().str() << ": " << entry.second << "\n";
  }
};

struct CminusfConstantPropagationPass
    : public PassWrapper<CminusfConstantPropagationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CminusfConstantPropagationPass)

  StringRef getArgument() const final { return "cminusf-const-prop"; }
  StringRef getDescription() const final {
    return "Propagate constants through variables in the cminusf dialect";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // For each VarOp that is stored once with a constant and never stored
    // again, replace all loads with a copy of that constant.
    // Only processes VarOps that are true local slots (not global refs).
    for (auto func : module.getOps<FunDeclOp>()) {
      SmallVector<VarOp> varsToProcess;
      func.walk([&](VarOp var) {
        if (var->getAttr("global_ref"))
          return;
        varsToProcess.push_back(var);
      });

      for (auto var : varsToProcess) {
        if (var->use_empty())
          continue;

        // Collect all stores and their values.
        SmallVector<ConstantOp> storedConstants;
        bool hasNonConstantStore = false;
        for (auto &use : var->getUses()) {
          if (auto storeOp = dyn_cast<StoreOp>(use.getOwner())) {
            if (auto constOp = storeOp.getValue().getDefiningOp<ConstantOp>())
              storedConstants.push_back(constOp);
            else
              hasNonConstantStore = true;
          }
        }

        // Only propagate when the variable has exactly one store and it's a
        // constant.  This avoids complex cases like store-then-modify.
        if (storedConstants.size() != 1 || hasNonConstantStore)
          continue;

        // Collect loads before modifying anything.
        SmallVector<LoadOp> loadsToReplace;
        for (auto &use : var->getUses()) {
          if (auto loadOp = dyn_cast<LoadOp>(use.getOwner()))
            loadsToReplace.push_back(loadOp);
        }

        ConstantOp constOp = storedConstants.front();
        for (auto loadOp : loadsToReplace) {
          OpBuilder builder(loadOp);
          Value newConst;
          auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
          auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>();
          if (intAttr)
            newConst = builder.create<ConstantOp>(loadOp.getLoc(), static_cast<int>(intAttr.getInt()));
          else if (floatAttr)
            newConst = builder.create<ConstantOp>(loadOp.getLoc(),
                                                  floatAttr.getValue().convertToFloat());
          if (!newConst)
            continue;
          loadOp.replaceAllUsesWith(newConst);
          loadOp.erase();
        }
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// Constant Folding Pass
//===----------------------------------------------------------------------===//

// Fold binary operations where both operands are constants.
// e.g., cminusf.binary add, constant(2), constant(3)  -->  constant(5)
struct FoldConstantBinaryOp : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsConst = op.getLhs().getDefiningOp<ConstantOp>();
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (!lhsConst || !rhsConst)
      return failure();

    Attribute lhsVal = lhsConst.getValue();
    Attribute rhsVal = rhsConst.getValue();

    bool lhsInt = lhsVal.isa<IntegerAttr>();
    bool rhsInt = rhsVal.isa<IntegerAttr>();
    bool lhsFloat = lhsVal.isa<FloatAttr>();
    bool rhsFloat = rhsVal.isa<FloatAttr>();

    if (!(lhsInt || lhsFloat) || !(rhsInt || rhsFloat))
      return failure();

    Location loc = op.getLoc();
    Value result;

    if (lhsInt && rhsInt) {
      int64_t l = lhsVal.cast<IntegerAttr>().getInt();
      int64_t r = rhsVal.cast<IntegerAttr>().getInt();
      int64_t v;
      switch (op.getOpType()) {
      case BinaryOpType::add: v = l + r; break;
      case BinaryOpType::sub: v = l - r; break;
      case BinaryOpType::mul: v = l * r; break;
      case BinaryOpType::div:
        if (r == 0) return failure();
        v = l / r; break;
      default: return failure();
      }
      result = rewriter.create<ConstantOp>(loc, static_cast<int>(v));
    } else {
      float l = lhsFloat ? lhsVal.cast<FloatAttr>().getValue().convertToFloat()
                         : static_cast<float>(lhsVal.cast<IntegerAttr>().getInt());
      float r = rhsFloat ? rhsVal.cast<FloatAttr>().getValue().convertToFloat()
                         : static_cast<float>(rhsVal.cast<IntegerAttr>().getInt());
      float v;
      switch (op.getOpType()) {
      case BinaryOpType::add: v = l + r; break;
      case BinaryOpType::sub: v = l - r; break;
      case BinaryOpType::mul: v = l * r; break;
      case BinaryOpType::div:
        if (r == 0.0f) return failure();
        v = l / r; break;
      default: return failure();
      }
      result = rewriter.create<ConstantOp>(loc, v);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Fold identity operations: add/sub with 0, mul/div with 1
struct FoldBinaryIdentityOp : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (!rhsConst)
      return failure();

    auto attr = rhsConst.getValue();
    bool isZero = false, isOne = false;

    if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
      isZero = (intAttr.getInt() == 0);
      isOne = (intAttr.getInt() == 1);
    } else if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
      isZero = floatAttr.getValue().isZero();
      isOne = floatAttr.getValue().isExactlyValue(1.0f);
    }

    if (!isZero && !isOne)
      return failure();

    switch (op.getOpType()) {
    case BinaryOpType::add:
    case BinaryOpType::sub:
      if (isZero) {
        rewriter.replaceOp(op, op.getLhs());
        return success();
      }
      break;
    case BinaryOpType::mul:
    case BinaryOpType::div:
      if (isOne) {
        rewriter.replaceOp(op, op.getLhs());
        return success();
      }
      break;
    }
    return failure();
  }
};

// Fold comparisons where both operands are the same SSA value.
// cmp(eq, %x, %x) -> constant(1), cmp(ne, %x, %x) -> constant(0), etc.
struct FoldCmpSameOperand : public OpRewritePattern<CmpOp> {
  using OpRewritePattern<CmpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return failure();

    Location loc = op.getLoc();
    Value result;
    switch (op.getPredicate()) {
    case CmpPredicate::eq:
    case CmpPredicate::le:
    case CmpPredicate::ge:
      result = rewriter.create<ConstantOp>(loc, 1);
      break;
    case CmpPredicate::ne:
    case CmpPredicate::lt:
    case CmpPredicate::gt:
      result = rewriter.create<ConstantOp>(loc, 0);
      break;
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Fold comparisons where both sides are constants.
struct FoldConstantCmpOp : public OpRewritePattern<CmpOp> {
  using OpRewritePattern<CmpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsConst = op.getLhs().getDefiningOp<ConstantOp>();
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (!lhsConst || !rhsConst)
      return failure();

    Attribute lhsVal = lhsConst.getValue();
    Attribute rhsVal = rhsConst.getValue();

    Location loc = op.getLoc();
    bool ret;
    if (lhsVal.isa<IntegerAttr>() && rhsVal.isa<IntegerAttr>()) {
      int64_t l = lhsVal.cast<IntegerAttr>().getInt();
      int64_t r = rhsVal.cast<IntegerAttr>().getInt();
      ret = evalCmp(op.getPredicate(), l, r);
    } else {
      float l = lhsVal.isa<FloatAttr>()
                    ? lhsVal.cast<FloatAttr>().getValue().convertToFloat()
                    : static_cast<float>(lhsVal.cast<IntegerAttr>().getInt());
      float r = rhsVal.isa<FloatAttr>()
                    ? rhsVal.cast<FloatAttr>().getValue().convertToFloat()
                    : static_cast<float>(rhsVal.cast<IntegerAttr>().getInt());
      ret = evalCmp(op.getPredicate(), l, r);
    }
    Value folded = rewriter.create<ConstantOp>(loc, ret ? 1 : 0);
    rewriter.replaceOp(op, folded);
    return success();
  }

  static bool evalCmp(CmpPredicate pred, int64_t l, int64_t r) {
    switch (pred) {
    case CmpPredicate::eq: return l == r;
    case CmpPredicate::ne: return l != r;
    case CmpPredicate::lt: return l < r;
    case CmpPredicate::le: return l <= r;
    case CmpPredicate::gt: return l > r;
    case CmpPredicate::ge: return l >= r;
    }
    return false;
  }
  static bool evalCmp(CmpPredicate pred, float l, float r) {
    switch (pred) {
    case CmpPredicate::eq: return l == r;
    case CmpPredicate::ne: return l != r;
    case CmpPredicate::lt: return l < r;
    case CmpPredicate::le: return l <= r;
    case CmpPredicate::gt: return l > r;
    case CmpPredicate::ge: return l >= r;
    }
    return false;
  }
};

struct CminusfConstantFoldingPass
    : public PassWrapper<CminusfConstantFoldingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CminusfConstantFoldingPass)

  StringRef getArgument() const final { return "cminusf-const-fold"; }
  StringRef getDescription() const final {
    return "Apply constant folding and algebraic simplifications on cminusf ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<CminusfDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConstantBinaryOp, FoldBinaryIdentityOp, FoldCmpSameOperand,
                 FoldConstantCmpOp>(&getContext());

    GreedyRewriteConfig config;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerCminusfToStandardPass() {
    return std::make_unique<LowerCminusfToStandardPass>();
}

std::unique_ptr<Pass> createLowerStandardToLLVMDialectPass() {
    return std::make_unique<LowerStandardToLLVMDialectPass>();
}

std::unique_ptr<Pass> createPrintCminusfOpCountPass() {
    return std::make_unique<PrintCminusfOpCountPass>();
}

std::unique_ptr<Pass> createCminusfConstantPropagationPass() {
    return std::make_unique<CminusfConstantPropagationPass>();
}

std::unique_ptr<Pass> createCminusfConstantFoldingPass() {
    return std::make_unique<CminusfConstantFoldingPass>();
}

void registerCminusfPasses() {
    PassRegistration<LowerCminusfToStandardPass>();
    PassRegistration<LowerStandardToLLVMDialectPass>();
    PassRegistration<PrintCminusfOpCountPass>();
    PassRegistration<CminusfConstantPropagationPass>();
    PassRegistration<CminusfConstantFoldingPass>();
}

} // namespace cminusf
} // namespace mlir
