//===- CminusfOpsDialect.cpp - Cminusf dialect and ops implementation --------------===//
//
// This file implements the dialect and operations in the Cminusf dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect.h"
#include "ast.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

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

CminusfDialect::CminusfDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<CminusfDialect>()) {
    initialize();
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
void Cminusf_ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                               int32_t value) {
    build(builder, state, builder.getI32IntegerAttr(value), builder.getI32Type());
}
void Cminusf_ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, float value) {
    build(builder, state, builder.getF32FloatAttr(value), builder.getF32Type());
}
void Cminusf_ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                               mlir::Attribute value, mlir::Type type) {
    state.addAttribute("value", value);
    state.addTypes(type);
}

void Cminusf_ConstantOp::print(OpAsmPrinter &printer) {
    printer << " ";
    printer.printAttributeWithoutType(getValueAttr());
    printer << " : " << getResult().getType();
}

mlir::ParseResult Cminusf_ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
    Attribute valueAttr;
    Type type;
    if (parser.parseAttribute(valueAttr, "value", result.attributes) || parser.parseColonType(type))
        return failure();

    if (!valueAttr.isa<IntegerSetAttr>() && !valueAttr.isa<FloatAttr>()) {
        return parser.emitError(parser.getNameLoc(), "expected integer or float attribute");
    }
    // 新增逻辑：检查属性类型与结果类型是否一致 ---------------
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

mlir::LogicalResult Cminusf_ConstantOp::verify(Cminusf_ConstantOp op) {
    auto valueAttr = op.value();
    auto valueType = valueAttr.getType();

    // 校验属性类型是否为 i32 或 f32
    if (!(valueType.isInteger(32) || valueType.isF32())) {
        return op.emitOpError("attribute must be a 32-bit integer or float");
    }

    // 校验结果类型与属性类型一致
    if (valueType != op.getType()) {
        return op.emitOpError("result type must match the attribute type");
    }

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// VarDeclarationOp
//===----------------------------------------------------------------------===//

void Cminusf_VarDeclarationOp::build(OpBuilder &builder, OperationState &state, Type type,
                                     StringRef name, Optional<int32_t> size) {
    state.addAttribute("name", builder.getStringAttr(name));
    if (size.hasValue()) {
        state.addAttribute("size", builder.getI32IntegerAttr(size.getValue()));
    }
    state.addTypes(type);
}

static void print(OpAsmPrinter &printer, Cminusf_VarDeclarationOp op) {
    printer << "cminusf.vardecl " << op.name();
    if (op.size().hasValue()) {
        printer << "[" << op.size().getValue() << "]";
    }
    printer << " : " << op.getType();
}

static ParseResult parseCminusfVarDeclarationOp(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    Optional<int32_t> sizeAttr;
    Type type;
    if (parser.parseAttribute(nameAttr, "name", result.attributes) ||
        parser.parseOptionalAttribute(sizeAttr, "size", result.attributes) ||
        parser.parseColonType(type))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_VarDeclarationOp op) { return success(); }

//===----------------------------------------------------------------------===//
// FunDeclarationOp
//===----------------------------------------------------------------------===//

void Cminusf_FunDeclarationOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                                     FunctionType type) {
    state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    state.addAttribute("type", TypeAttr::get(type));
    state.addRegion();
}

static void print(OpAsmPrinter &printer, Cminusf_FunDeclarationOp op) {
    printer << "cminusf.fundecl @" << op.getName() << "(";
    if (!op.getBody().empty()) {
        interleaveComma(op.getBody().front().getArguments(), printer);
    }
    printer << ")";
    printer.printOptionalAttrDict(op->getAttrs(), {SymbolTable::getSymbolAttrName(), "type"});
    printer << " : " << op.getType();
    printer.printRegion(op.getBody(), false, true);
}

static ParseResult parseCminusfFunDeclarationOp(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    Type type;
    if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes) ||
        parser.parseColonType(type))
        return failure();
    result.addAttribute("type", TypeAttr::get(type.cast<FunctionType>()));
    auto *body = result.addRegion();
    if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{}))
        return failure();
    return success();
}

static LogicalResult verify(Cminusf_FunDeclarationOp op) {
    // 校验函数类型是否正确
    if (!op.getType().isa<FunctionType>())
        return op.emitOpError("requires a 'FunctionType' attribute");
    return success();
}

//===----------------------------------------------------------------------===//
// ParamOp
//===----------------------------------------------------------------------===//

void Cminusf_ParamOp::build(OpBuilder &builder, OperationState &state, Type type, StringRef name,
                            bool isArray) {
    state.addAttribute("name", builder.getStringAttr(name));
    state.addAttribute("isArray", builder.getBoolAttr(isArray));
    state.addTypes(type);
}

static void print(OpAsmPrinter &printer, Cminusf_ParamOp op) {
    printer << "cminusf.param " << op.name();
    if (op.isArray())
        printer << "[]";
    printer << " : " << op.getType();
}

static ParseResult parseCminusfParamOp(OpAsmParser &parser, OperationState &result) {
    StringAttr nameAttr;
    BoolAttr isArrayAttr;
    Type type;
    if (parser.parseAttribute(nameAttr, "name", result.attributes) ||
        parser.parseAttribute(isArrayAttr, "isArray", result.attributes) ||
        parser.parseColonType(type))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_ParamOp op) { return success(); }

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

void Cminusf_AssignOp::build(OpBuilder &builder, OperationState &state, Value lhs, Value rhs) {
    state.addOperands(lhs);
    state.addOperands(rhs);
    state.addTypes(lhs.getType());
}

static void print(OpAsmPrinter &printer, Cminusf_AssignOp op) {
    printer << "cminusf.assign ";
    printer.printOperand(op.getOperands().front());
    printer << " = ";
    printer.printOperand(op.getOperands()[1]);
}

static ParseResult parseCminusfAssignOp(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::OperandType lhs, rhs;
    Type type;
    if (parser.parseOperand(lhs) || parser.parseEqual() || parser.parseOperand(rhs) ||
        parser.parseColonType(type) || parser.resolveOperands({lhs, rhs}, type, result.operands))
        return failure();
    result.addTypes(type);
    return success();
}

static LogicalResult verify(Cminusf_AssignOp op) { return success(); }

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