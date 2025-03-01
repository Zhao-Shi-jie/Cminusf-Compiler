#include "Dialect.h"
#include "ast.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include <cstdint>

using namespace mlir;
using namespace mlir::cminusf;

#include "mlir/CminusfDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CminusfDialect
//===----------------------------------------------------------------------===//

void CminusfDialect::initialize() {
    // 注册所有自定义操作
    addOperations<
#define GET_OP_LIST
#include "mlir/CminusfOps.cpp.inc"
        >();
}

///===----------------------------------------------------------------------===//
// Cminusf Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation.
    if (FunctionType funcType = type.dyn_cast<FunctionType>()) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc, result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all of the types are the same, print the type directly.
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(), [=](Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type.
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
// 支持 int32_t 和 float 的构建函数
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, int value) {
    ConstantOp::build(builder, state, builder.getI32Type(), builder.getI32IntegerAttr(value));
}
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, float value) {
    ConstantOp::build(builder, state, builder.getF32Type(), builder.getF32FloatAttr(value));
}

mlir::ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
    Attribute valueAttr;
    Type type;
    if (parser.parseAttribute(valueAttr, "value", result.attributes) || parser.parseColonType(type))
        return failure();

    if (!valueAttr.isa<IntegerAttr>() && !valueAttr.isa<FloatAttr>()) {
        return parser.emitError(parser.getNameLoc(), "expected integer or float attribute");
    }
    // 检查属性类型与结果类型是否一致
    if (auto intAttr = valueAttr.dyn_cast<mlir::IntegerAttr>()) {
        // 如果属性是整数，结果类型必须是 i32
        if (!type.isInteger(32)) {
            return parser.emitError(parser.getNameLoc())
                   << "integer attribute requires i32 result type, but got " << type;
        }
    } else if (auto floatAttr = valueAttr.dyn_cast<mlir::FloatAttr>()) {
        // 如果属性是浮点，结果类型必须是 f32
        if (!type.isF32()) {
            return parser.emitError(parser.getNameLoc())
                   << "float attribute requires f32 result type, but got " << type;
        }
    }

    result.addTypes(type);
    return success();
}

mlir::LogicalResult ConstantOp::verify() {
    if (!getValue().isa<FloatAttr>() && !getValue().isa<IntegerAttr>()) {
        return emitError("value must be either an integer (i32) or floating-point (f32)");
    }
    return success();
}

//===----------------------------------------------------------------------===//
// VarDeclarationOp
//===----------------------------------------------------------------------===//

void VarDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, StringAttr var_name,
                      Type var_type) {
    state.addAttribute("var_name", var_name);
    state.addAttribute("var_type_attr", TypeAttr::get(var_type));
    // 结果类型与参数类型一致
    state.addTypes(var_type);
}

void VarDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::StringAttr var_name, mlir::Type var_type,
                      /*optional*/ mlir::IntegerAttr size) {
    state.addAttribute("var_name", var_name);
    state.addAttribute("var_type_attr", TypeAttr::get(var_type));
    state.addAttribute("size", size);
    // 结果类型是 MemRef
    state.addTypes(MemRefType::get({size.getInt()}, var_type));
}

LogicalResult VarDeclOp::verify() {
    // 获取基本类型
    Attribute varTypeAttr = getVarTypeAttrAttr();
    Type baseType;

    if (auto intAttr = varTypeAttr.dyn_cast<IntegerAttr>()) {
        baseType = intAttr.getType();
        if (baseType.getIntOrFloatBitWidth() != 32) {
            return emitError("Integer is not 32 bit, but ") << baseType.getIntOrFloatBitWidth();
        }
    } else if (auto floatAttr = varTypeAttr.dyn_cast<FloatAttr>()) {
        baseType = floatAttr.getType();
        if (baseType.getIntOrFloatBitWidth() != 32) {
            return emitError("Float is not 32 bit, but ") << baseType.getIntOrFloatBitWidth();
        }
    } else {
        // 检查基本类型是否有效（根据您的类型系统调整）
        // if (!baseType.isSignlessInteger(32) && !baseType.isF32()) {
        //     return emitOpError("variable's type should be i32 or f32, but got ") << baseType;
        // }
        baseType = varTypeAttr.dyn_cast<TypedAttr>().getType();
        return emitOpError("variable's type should be i32 or f32, but got ") << baseType;
    }

    // 获取结果类型（可能是基本类型或MemRefType）
    Type resultType = getType();

    // 检查数组情况
    if (auto sizeAttr = getSizeAttr()) {
        int64_t size = sizeAttr.getInt();
        if (size <= 0) {
            return emitOpError("size should more than 0");
        }
        // 如果有size属性，结果类型必须是MemRefType
        auto memRefType = resultType.dyn_cast<MemRefType>();
        if (!memRefType)
            return emitOpError("当指定size属性时，结果类型必须是memref类型");

        // 验证MemRefType的元素类型与指定的变量类型一致
        if (memRefType.getElementType() != baseType)
            return emitOpError("memref元素类型 ")
                   << memRefType.getElementType() << " 与指定的变量类型不匹配 " << baseType;

        // 验证MemRefType的维度与size属性一致
        if (memRefType.getRank() != 1)
            return emitOpError("数组变量的memref必须是一维的");

        // 验证MemRefType的维度大小与size属性一致
        if (memRefType.getDimSize(0) != size)
            return emitOpError("memref维度大小 ")
                   << memRefType.getDimSize(0) << " 与size属性不匹配 " << size;
    } else {
        // 如果没有size属性，结果类型必须与变量类型相同
        if (resultType != baseType)
            return emitOpError("结果类型 ") << resultType << " 与指定的变量类型不匹配 " << baseType;
    }

    return success();
}

//===----------------------------------------------------------------------===//
// FunDeclarationOp
//===----------------------------------------------------------------------===//

void FunDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                      mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    // 使用 FunctionOpInterface 提供的便捷方法来填充我们的 Cminusf_FunDeclOp 状态
    // 并创建一个入口块
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

ParseResult FunDeclOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // 调用 FunctionOpInterface 提供的工具方法来解析函数操作
    auto buildFuncType = [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
                            llvm::ArrayRef<mlir::Type> results,
                            mlir::function_interface_impl::VariadicFlag,
                            std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name),
        buildFuncType, getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FunDeclOp::print(mlir::OpAsmPrinter &p) {
    // 调用 FunctionOpInterface 提供的工具方法来打印函数操作
    mlir::function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
                                                   getFunctionTypeAttrName().getValue(),
                                                   getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

void AssignOp::print(OpAsmPrinter &p) {
    p << " " << getVar() << " = " << getExpr();
    // p.printOptionalAttrDict((*this)->getAttrs());
    p << " : " << getResult().getType();
}

ParseResult AssignOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand var, expr;
    Type type;

    if (parser.parseOperand(var) || parser.parseEqual() || parser.parseOperand(expr) ||
        parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type))
        return failure();

    if (parser.resolveOperand(var, type, result.operands) ||
        parser.resolveOperand(expr, type, result.operands))
        return failure();

    result.addTypes(type);
    return success();
}

LogicalResult AssignOp::verify() {
    // 检查变量和表达式类型是否匹配
    if (getVar().getType() != getExpr().getType())
        return emitOpError("变量类型 ")
               << getVar().getType() << " 与表达式类型 " << getExpr().getType() << " 不匹配";

    // 检查结果类型是否与变量类型匹配
    if (getResult().getType() != getVar().getType())
        return emitOpError("结果类型 ")
               << getResult().getType() << " 与变量类型 " << getVar().getType() << " 不匹配";

    return success();
}

LogicalResult AssignOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                         ValueRange operands, DictionaryAttr attributes,
                                         RegionRange regions,
                                         SmallVectorImpl<Type> &inferredReturnTypes) {
    // 确保有两个操作数
    if (operands.size() != 2)
        return emitOptionalError(location, "需要两个操作数（变量和表达式）");

    // 检查类型匹配
    auto varType = operands[0].getType();
    auto exprType = operands[1].getType();
    if (varType != exprType)
        return emitOptionalError(location, "变量类型与表达式类型不匹配");

    // 结果类型与变量类型一致
    inferredReturnTypes.push_back(varType);
    return success();
}

//===----------------------------------------------------------------------===//
// CompoundStmtOp
//===----------------------------------------------------------------------===//

void Cminusf_CompoundStmtOp::build(OpBuilder &builder, OperationState &state) { state.addRegion(); }

static void print(OpAsmPrinter &printer, Cminusf_CompoundStmtOp op) {
    printer << "cminusf.compound";
    printer.printRegion(op.getBody(), false, true);
}

static ParseResult parseCminusfCompoundStmtOp(OpAsmParser &parser, OperationState &result) {
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
    return success();
}

static LogicalResult verify(Cminusf_CompoundStmtOp op) { return success(); }

//===----------------------------------------------------------------------===//
// ExpressionStmtOp
//===----------------------------------------------------------------------===//

void Cminusf_ExpressionStmtOp::build(OpBuilder &builder, OperationState &state, Value expr) {
    state.addOperands(expr);
}

static void print(OpAsmPrinter &printer, Cminusf_ExpressionStmtOp op) {
    printer << "cminusf.exprstmt ";
    printer.printOperand(op.getOperands().front());
}

static ParseResult parseCminusfExpressionStmtOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType expr;
    Type type;
    if (parser.parseOperand(expr) || parser.parseColonType(type) ||
        parser.resolveOperand(expr, type, result.operands))
        return failure();
    return success();
}

static LogicalResult verify(Cminusf_ExpressionStmtOp op) { return success(); }

//===----------------------------------------------------------------------===//
// SelectionStmtOp
//===----------------------------------------------------------------------===//

void Cminusf_SelectionStmtOp::build(OpBuilder &builder, OperationState &state, Value cond,
                                    Value then, Optional<Value> elseOp) {
    state.addOperands(cond);
    state.addOperands(then);
    if (elseOp.hasValue())
        state.addOperands(elseOp.getValue());
}

static void print(OpAsmPrinter &printer, Cminusf_SelectionStmtOp op) {
    printer << "cminusf.if ";
    printer.printOperand(op.getOperands().front());
    printer << " : ";
    printer.printOperand(op.getOperands()[1]);
    if (op.getNumOperands() == 3) {
        printer << " else ";
        printer.printOperand(op.getOperands()[2]);
    }
}

static ParseResult parseCminusfSelectionStmtOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType cond, then, elseOp;
    Type type;
    if (parser.parseOperand(cond) || parser.parseColonType(type) ||
        parser.resolveOperand(cond, type, result.operands) || parser.parseOperand(then) ||
        parser.parseColonType(type) || parser.resolveOperand(then, type, result.operands))
        return failure();
    if (!parser.parseOptionalKeyword("else")) {
        if (parser.parseOperand(elseOp) || parser.parseColonType(type) ||
            parser.resolveOperand(elseOp, type, result.operands))
            return failure();
    }
    return success();
}

static LogicalResult verify(Cminusf_SelectionStmtOp op) { return success(); }

//===----------------------------------------------------------------------===//
// IterationStmtOp
//===----------------------------------------------------------------------===//

void Cminusf_IterationStmtOp::build(OpBuilder &builder, OperationState &state, Value cond,
                                    Value body) {
    state.addOperands(cond);
    state.addOperands(body);
}

static void print(OpAsmPrinter &printer, Cminusf_IterationStmtOp op) {
    printer << "cminusf.while ";
    printer.printOperand(op.getOperands().front());
    printer << " : ";
    printer.printOperand(op.getOperands()[1]);
}

static ParseResult parseCminusfIterationStmtOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType cond, body;
    Type type;
    if (parser.parseOperand(cond) || parser.parseColonType(type) ||
        parser.resolveOperand(cond, type, result.operands) || parser.parseOperand(body) ||
        parser.parseColonType(type) || parser.resolveOperand(body, type, result.operands))
        return failure();
    return success();
}

static LogicalResult verify(Cminusf_IterationStmtOp op) { return success(); }

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

void Cminusf_AssignOp::build(OpBuilder &builder, OperationState &state, Value var, Value expr) {
    auto varType = var.getType();
    auto exprType = expr.getType();

    // 确保变量类型和表达式类型匹配
    if (varType != exprType) {
        emitError("变量类型 ") << varType << " 与表达式类型 " << exprType << " 不匹配";
    }

    // 设置操作的类型
    state.addTypes(varType);
    state.addOperands({var, expr});
}

//===----------------------------------------------------------------------===//
// SimpleOp
//===----------------------------------------------------------------------===//

void Cminusf_SimpleOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs,
                             int32_t op) {
    state.addOperands(lhs);
    state.addOperands(rhs);
    state.addAttribute("op", builder.getI32IntegerAttr(op));
    state.addTypes(lhs.getType());
}

static void print(OpAsmPrinter &printer, Cminusf_SimpleOp op) {
    printer << "cminusf.simple ";
    printer.printOperand(op.getOperands().front());
    printer << " " << op.op() << " ";
    printer.printOperand(op.getOperands()[1]);
}

static ParseResult parseCminusfSimpleOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType lhs, rhs;
    IntegerAttr opAttr;
    Type type;
    if (parser.parseOperand(lhs) || parser.parseOperand(rhs) ||
        parser.parseAttribute(opAttr, "op", result.attributes) || parser.parseColonType(type) ||
        parser.resolveOperands({lhs, rhs}, type, result.operands))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_SimpleOp op) { return success(); }

//===----------------------------------------------------------------------===//
// AdditiveOp
//===----------------------------------------------------------------------===//

void Cminusf_AdditiveOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs,
                               int32_t op) {
    state.addOperands(lhs);
    state.addOperands(rhs);
    state.addAttribute("op", builder.getI32IntegerAttr(op));
    state.addTypes(lhs.getType());
}

static void print(OpAsmPrinter &printer, Cminusf_AdditiveOp op) {
    printer << "cminusf.additive ";
    printer.printOperand(op.getOperands().front());
    printer << " " << op.op() << " ";
    printer.printOperand(op.getOperands()[1]);
}

static ParseResult parseCminusfAdditiveOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType lhs, rhs;
    IntegerAttr opAttr;
    Type type;
    if (parser.parseOperand(lhs) || parser.parseOperand(rhs) ||
        parser.parseAttribute(opAttr, "op", result.attributes) || parser.parseColonType(type) ||
        parser.resolveOperands({lhs, rhs}, type, result.operands))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_AdditiveOp op) { return success(); }

//===----------------------------------------------------------------------===//
// TermOp
//===----------------------------------------------------------------------===//

void Cminusf_TermOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs,
                           int32_t op) {
    state.addOperands(lhs);
    state.addOperands(rhs);
    state.addAttribute("op", builder.getI32IntegerAttr(op));
    state.addTypes(lhs.getType());
}

static void print(OpAsmPrinter &printer, Cminusf_TermOp op) {
    printer << "cminusf.term ";
    printer.printOperand(op.getOperands().front());
    printer << " " << op.op() << " ";
    printer.printOperand(op.getOperands()[1]);
}

static ParseResult parseCminusfTermOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType lhs, rhs;
    IntegerAttr opAttr;
    Type type;
    if (parser.parseOperand(lhs) || parser.parseOperand(rhs) ||
        parser.parseAttribute(opAttr, "op", result.attributes) || parser.parseColonType(type) ||
        parser.resolveOperands({lhs, rhs}, type, result.operands))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_TermOp op) { return success(); }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void Cminusf_CallOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                           ArrayRef<Value> args) {
    state.addAttribute("name", builder.getStringAttr(name));
    state.addOperands(args);
    state.addTypes(builder.getI32Type());
}

static void print(OpAsmPrinter &printer, Cminusf_CallOp op) {
    printer << "cminusf.call " << op.name() << "(";
    printer.printOperands(op.getOperands());
    printer << ")";
}

static ParseResult parseCminusfCallOp(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    SmallVector<OpAsmParser::OperandType, 4> args;
    Type type;
    if (parser.parseAttribute(nameAttr, "name", result.attributes) ||
        parser.parseOperandList(args) || parser.parseColonType(type) ||
        parser.resolveOperands(args, type, result.operands))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_CallOp op) { return success(); }

#define GET_OP_CLASSES
#include "mlir/CminusfOps.cpp.inc"