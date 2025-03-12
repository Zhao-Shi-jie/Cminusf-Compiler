#include "Dialect.h"
#include "ast.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "syntax_analyzer.h"
#include "llvm/Support/Debug.h"
#include <cstdint>
#include <string>

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
// Cminusf Utils
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
// // 支持 int32_t 和 float 的构建函数
// void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, int value) {
//     ConstantOp::build(builder, state, builder.getI32Type(), builder.getI32IntegerAttr(value));
// }
// void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, float value) {
//     ConstantOp::build(builder, state, builder.getF32Type(), builder.getF32FloatAttr(value));
// }

void ConstantOp::print(OpAsmPrinter &p) {
    // 打印结果类型和操作名称（这部分由 MLIR 框架处理）
    // 格式会是 "%0 = cminusf.constant"

    Attribute valueAttr = getValue();
    p << " ";
    if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>()) {
        p << intAttr.getInt();
    } else if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>()) {
        p << floatAttr.getValue().convertToFloat();
    } else {
        p << valueAttr;
    }
    p << " : ";
    p << getType();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
    // 解析结果类型
    Type type;
    if (parser.parseType(type))
        return failure();

    // 将类型添加到结果中
    result.addTypes(type);

    // 解析常量值
    if (type.isa<IntegerType>()) {
        // 整数常量
        APInt intValue;
        if (parser.parseInteger(intValue))
            return failure();

        // 创建整数属性
        auto valueAttr = IntegerAttr::get(type, intValue);
        result.addAttribute("value", valueAttr);
    } else if (type.isa<FloatType>()) {
        // 浮点数常量
        double floatValue;
        if (parser.parseFloat(floatValue))
            return failure();

        // 创建浮点属性
        auto valueAttr = FloatAttr::get(type, floatValue);
        result.addAttribute("value", valueAttr);
    } else {
        // 其他类型，尝试直接解析属性
        Attribute valueAttr;
        if (parser.parseAttribute(valueAttr, "value", result.attributes))
            return failure();
    }

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

// void VarDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, StringAttr var_name, Type
// var_type) {
//     state.addAttribute("var_name", var_name);
//     state.addAttribute("var_type_attr", TypeAttr::get(var_type));
//     // 结果类型与参数类型一致
//     state.addTypes(var_type);
// }

// void VarDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::StringAttr var_name,
//                       mlir::Type var_type,
//                       /*optional*/ mlir::IntegerAttr size) {
//     state.addAttribute("var_name", var_name);
//     state.addAttribute("var_type_attr", TypeAttr::get(var_type));
//     state.addAttribute("size", size);
//     // 结果类型是 MemRef
//     state.addTypes(MemRefType::get({size.getInt()}, var_type));
// }

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
            return emitOpError("memref维度大小 ") << memRefType.getDimSize(0) << " 与size属性不匹配 " << size;
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
                            llvm::ArrayRef<mlir::Type> results, mlir::function_interface_impl::VariadicFlag,
                            std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FunDeclOp::print(mlir::OpAsmPrinter &p) {
    // 调用 FunctionOpInterface 提供的工具方法来打印函数操作
    mlir::function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false,
                                                   getFunctionTypeAttrName().getValue(),
                                                   getArgAttrsAttrName(), getResAttrsAttrName());
}

mlir::Region *FunDeclOp::getCallableRegion() { return &getBody(); }

llvm::ArrayRef<mlir::Type> FunDeclOp::getCallableResults() { return getFunctionType().getResults(); }

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(OpBuilder &builder, OperationState &result, StringRef callee, ArrayRef<Value> arguments) {
    result.addAttribute("callee", FlatSymbolRefAttr::get(builder.getContext(), callee));
    result.addOperands(arguments);

    Operation *parentOp = builder.getInsertionBlock()->getParentOp();
    ModuleOp module = parentOp->getParentOfType<ModuleOp>();

    if (module) {
        if (auto callableOp = module.lookupSymbol<mlir::CallableOpInterface>(callee)) {
            auto resultType = callableOp.getCallableResults();
            if (!resultType.empty()) {
                result.addTypes(resultType);
            }
        } else if (auto funcOp = module.lookupSymbol<FunDeclOp>(callee)) {
            auto resultType = funcOp->getResultTypes();
            if (!resultType.empty()) {
                result.addTypes(resultType);
            }
        } else {
            // todo : 修正无法找到被调用函数时的处理
            result.addTypes(builder.getIntegerType(32));
        }
    }
}

LogicalResult CallOp::verify() {
    Operation *op = getOperation();
    StringRef callee = getCallee();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) {
        return emitError() << "cannot find parent module";
    }

    // 1. 验证被调用函数是否存在
    auto funcOp = module.lookupSymbol<FunDeclOp>(callee);
    if (!funcOp) {
        return emitError() << "cannot find function '" << callee << "'";
    }

    // 2. 验证参数数量是否一致
    unsigned numArguments = getNumOperands();
    unsigned numParameters = funcOp.getNumArguments();

    if (numArguments != numParameters) {
        return emitError() << "incorrect number of operands for callee '" << callee << "': expected "
                           << numParameters << ", but got " << numArguments;
    }

    // 3. 验证参数类型是否一致
    auto argumentTypes = getOperandTypes();
    auto parameterTypes = funcOp.getArgumentTypes();

    for (unsigned i = 0; i < numArguments; ++i) {
        if (argumentTypes[i] != parameterTypes[i]) {
            return emitError() << "operand type mismatch for operand " << i << " of callee '" << callee
                               << "': "
                               << "expected " << parameterTypes[i] << ", but got " << argumentTypes[i];
        }
    }

    // 4. 验证返回类型是否一致（如果函数有返回值）
    // 注意：您可能需要根据您的CallOp定义来调整这部分
    auto resultTypes = getResultTypes();
    auto funcResultTypes = funcOp.getResultTypes();

    if (resultTypes.size() != funcResultTypes.size()) {
        return emitError() << "incorrect number of results for callee '" << callee << "': expected "
                           << funcResultTypes.size() << ", but got " << resultTypes.size();
    }

    for (unsigned i = 0; i < resultTypes.size(); ++i) {
        if (resultTypes[i] != funcResultTypes[i]) {
            return emitError() << "result type mismatch for result " << i << " of callee '" << callee << "': "
                               << "expected " << funcResultTypes[i] << ", but got " << resultTypes[i];
        }
    }

    return success();
}

// CallOpInterface implementation
CallInterfaceCallable CallOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

Operation::operand_range CallOp::getArgOperands() { return getArgs(); }

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
                                         ValueRange operands, DictionaryAttr attributes, RegionRange regions,
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
// CmpOp
//===----------------------------------------------------------------------===//

/// 从字符串获取谓词枚举值
CmpPredicate CmpOp::stringToCmpPredicate(StringRef str) {
    return llvm::StringSwitch<CmpPredicate>(str)
        .Case("eq", CmpPredicate::eq)
        .Case("ne", CmpPredicate::ne)
        .Case("lt", CmpPredicate::lt)
        .Case("le", CmpPredicate::le)
        .Case("gt", CmpPredicate::gt)
        .Case("ge", CmpPredicate::ge)
        .Default(CmpPredicate::eq);
}

/// 获取谓词的反向谓词
CmpPredicate CmpOp::getInversePredicate() {
    switch (getPredicate()) {
    case CmpPredicate::eq:
        return CmpPredicate::ne;
    case CmpPredicate::ne:
        return CmpPredicate::eq;
    case CmpPredicate::lt:
        return CmpPredicate::ge;
    case CmpPredicate::le:
        return CmpPredicate::gt;
    case CmpPredicate::gt:
        return CmpPredicate::le;
    case CmpPredicate::ge:
        return CmpPredicate::lt;
    }
    llvm_unreachable("未知的比较谓词");
}

/// 检查两个操作数是否为同一类型
bool CmpOp::isSameOperandsType() { return getLhs().getType() == getRhs().getType(); }

void CmpOp::print(OpAsmPrinter &printer) {
    std::string logicalOp = "";
    switch (getPredicateAttr().getValue()) {
    case CmpPredicate::eq:
        logicalOp = "eq";
        break;
    case CmpPredicate::ne:
        logicalOp = "ne";
        break;
    case CmpPredicate::lt:
        logicalOp = "lt";
        break;
    case CmpPredicate::le:
        logicalOp = "le";
        break;
    case CmpPredicate::gt:
        logicalOp = "gt";
        break;
    case CmpPredicate::ge:
        logicalOp = "ge";
        break;
    default:
        logicalOp = "unkonwn";
    }
    // 打印谓词和操作数
    printer << " " << logicalOp;
    printer << " " << getLhs() << ", " << getRhs();

    // 打印类型
    printer << " : " << getLhs().getType();
}

ParseResult CmpOp::parse(OpAsmParser &parser, OperationState &result) {
    // 解析谓词关键字
    StringRef predicateStr;
    if (parser.parseKeyword(&predicateStr))
        return failure();

    // 解析操作数
    OpAsmParser::UnresolvedOperand lhs, rhs;
    if (parser.parseOperand(lhs) || parser.parseComma() || parser.parseOperand(rhs))
        return failure();

    // 解析类型
    Type operandType;
    if (parser.parseColonType(operandType))
        return failure();

    // 验证类型是否为 f32 或 i32
    if (!operandType.isF32() && !operandType.isInteger(32))
        return parser.emitError(parser.getNameLoc(), "操作数类型必须是 f32 或 i32");

    // 将谓词字符串转换为枚举并添加为属性
    CmpPredicate predicate = stringToCmpPredicate(predicateStr);
    result.addAttribute("predicate", CmpPredicateAttr::get(parser.getContext(), predicate));

    // 解析操作数，确保它们具有相同的类型
    if (parser.resolveOperand(lhs, operandType, result.operands) ||
        parser.resolveOperand(rhs, operandType, result.operands))
        return failure();

    // 添加结果类型（始终为 i32）
    result.addTypes(parser.getBuilder().getI32Type());

    return success();
}

LogicalResult CmpOp::verify() {
    // 验证操作数类型相同
    if (!isSameOperandsType())
        return emitOpError() << "操作数必须具有相同类型，但得到 " << getLhs().getType() << " 和 "
                             << getRhs().getType();

    // 验证操作数类型为 f32 或 i32
    Type operandType = getLhs().getType();
    if (!operandType.isF32() && !operandType.isInteger(32))
        return emitOpError() << "操作数类型必须是 f32 或 i32，但得到 " << operandType;

    return success();
}

// LogicalResult CmpOp::inferReturnTypes(MLIRContext *context, Optional<Location> location, ValueRange
// operands,
//                                       DictionaryAttr attributes, RegionRange regions,
//                                       SmallVectorImpl<Type> &inferredReturnTypes) {

//     // 比较操作始终返回 i32 类型表示布尔结果
//     inferredReturnTypes.push_back(IntegerType::get(context, 32));
//     return success();
// }

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

/// 检查结果类型是否有效
bool BinaryOp::hasValidResultType() {
    Type lhsType = getLhs().getType();
    Type rhsType = getRhs().getType();
    Type resultType = getResult().getType();

    // 加减法，操作数必须相同，结果必须与操作数相同
    if (isAddOrSubOp()) {
        if (lhsType != rhsType) {
            return false;
        }
        return resultType == lhsType;
    }
    // 乘除法，如果操作数类型相同，结果应与操作数相同
    // 如果操作数类型不同，结果必须是 f32
    else if (isMulOrDivOp()) {
        if (lhsType == rhsType) {
            return resultType == lhsType;
        } else {
            return resultType.isF32();
        }
    }

    return false;
}

/// 将字符串转换为二元操作类型枚举
BinaryOpType BinaryOp::stringToBinaryOpType(StringRef str) {
    return llvm::StringSwitch<BinaryOpType>(str)
        .Case("add", BinaryOpType::add)
        .Case("sub", BinaryOpType::sub)
        .Case("mul", BinaryOpType::mul)
        .Case("div", BinaryOpType::div)
        .Default(BinaryOpType::add); // 默认为加法
}

/// 获取操作类型的字符串表示
StringRef BinaryOp::getOperationSymbol() {
    switch (getOpType()) {
    case BinaryOpType::add:
        return "+";
    case BinaryOpType::sub:
        return "-";
    case BinaryOpType::mul:
        return "*";
    case BinaryOpType::div:
        return "/";
    }
    llvm_unreachable("未知的二元操作类型");
}

/// 自定义打印方法
void BinaryOp::print(OpAsmPrinter &printer) {
    std::string binaryOp = "";
    switch (getOpTypeAttr().getValue()) {
    case BinaryOpType::add:
        binaryOp = "add";
        break;
    case BinaryOpType::sub:
        binaryOp = "sub";
        break;
    case BinaryOpType::mul:
        binaryOp = "mul";
        break;
    case BinaryOpType::div:
        binaryOp = "div";
        break;
    default:
        binaryOp = "unkonwn";
    }
    printer << " ";
    printer << binaryOp << ", ";
    printer << getLhs() << ", " << getRhs();
    printer << " : " << getResult().getType();
}

/// 自定义解析方法
ParseResult BinaryOp::parse(OpAsmParser &parser, OperationState &result) {
    // 解析操作类型枚举字符串
    StringRef opTypeStr;
    if (parser.parseKeyword(&opTypeStr) || parser.parseComma())
        return failure();

    // 将操作类型转换为枚举并添加到属性
    auto opType = BinaryOp::stringToBinaryOpType(opTypeStr);
    auto attr = BinaryOpTypeAttr::get(parser.getContext(), opType);
    result.addAttribute("op_type", attr);

    // 解析操作数
    OpAsmParser::UnresolvedOperand lhs, rhs;
    if (parser.parseOperand(lhs) || parser.parseComma() || parser.parseOperand(rhs))
        return failure();

    // 解析结果类型
    Type resultType;
    if (parser.parseColonType(resultType))
        return failure();

    // 确保结果类型是 f32 或 i32
    if (!resultType.isF32() && !resultType.isInteger(32))
        return parser.emitError(parser.getNameLoc(), "结果类型必须是 f32 或 i32");

    bool isAddSubOp = (opType == BinaryOpType::add || opType == BinaryOpType::sub);

    // 对于加减法，所有操作数必须同类型
    if (isAddSubOp) {
        if (parser.resolveOperand(lhs, resultType, result.operands) ||
            parser.resolveOperand(rhs, resultType, result.operands))
            return failure();
    }
    // 对于乘除法，如果结果是 f32，操作数可以不同类型
    else {
        // 对于操作数，我们需要接受 i32 或 f32
        if (resultType.isF32()) {
            // 如果结果是 f32，则操作数可以是 i32 或 f32
            Type lhsType, rhsType;
            if (parser.parseOptionalColon()) {
                // 没有指定操作数类型，默认与结果相同
                lhsType = rhsType = resultType;
            } else {
                // 解析操作数类型
                if (parser.parseType(lhsType))
                    return failure();

                if (parser.parseOptionalComma()) {
                    // 只有一个类型，两个操作数使用相同类型
                    rhsType = lhsType;
                } else {
                    if (parser.parseType(rhsType))
                        return failure();
                }
            }

            // 验证操作数类型是否合法
            if ((!lhsType.isF32() && !lhsType.isInteger(32)) || (!rhsType.isF32() && !rhsType.isInteger(32)))
                return parser.emitError(parser.getNameLoc(), "操作数类型必须是 f32 或 i32");

            // 解析操作数
            if (parser.resolveOperand(lhs, lhsType, result.operands) ||
                parser.resolveOperand(rhs, rhsType, result.operands))
                return failure();
        } else {
            // 如果结果是 i32，则操作数必须是 i32
            if (parser.resolveOperand(lhs, resultType, result.operands) ||
                parser.resolveOperand(rhs, resultType, result.operands))
                return failure();
        }
    }

    // 设置结果类型
    result.addTypes(resultType);

    return success();
}

/// 自定义验证方法
LogicalResult BinaryOp::verify() {
    // 获取操作数和结果类型
    Type lhsType = getLhs().getType();
    Type rhsType = getRhs().getType();
    Type resultType = getResult().getType();

    // 验证操作数类型为 f32 或 i32
    if (!lhsType.isF32() && !lhsType.isInteger(32))
        return emitOpError() << "左操作数类型必须是 f32 或 i32，但得到 " << lhsType;

    if (!rhsType.isF32() && !rhsType.isInteger(32))
        return emitOpError() << "右操作数类型必须是 f32 或 i32，但得到 " << rhsType;

    std::string binaryOp = "";
    switch (getOpTypeAttr().getValue()) {
    case BinaryOpType::add:
        binaryOp = "add";
        break;
    case BinaryOpType::sub:
        binaryOp = "sub";
        break;
    case BinaryOpType::mul:
        binaryOp = "mul";
        break;
    case BinaryOpType::div:
        binaryOp = "div";
        break;
    default:
        binaryOp = "unkonwn";
    }
    // 加减法专用验证
    if (isAddOrSubOp()) {
        if (lhsType != rhsType)
            return emitOpError() << binaryOp << " 操作要求操作数类型相同，但得到 " << lhsType << " 和 "
                                 << rhsType;

        if (resultType != lhsType)
            return emitOpError() << binaryOp << " 操作的结果类型必须与操作数类型相同，但得到 " << resultType
                                 << " 而不是 " << lhsType;
    }

    // 乘除法专用验证
    if (isMulOrDivOp()) {
        if (lhsType == rhsType) {
            // 相同类型操作数，结果必须相同
            if (resultType != lhsType)
                return emitOpError() << binaryOp << " 操作中，相同类型操作数的结果类型必须相同，但得到 "
                                     << resultType << " 而不是 " << lhsType;
        } else {
            // 不同类型操作数，结果必须是 f32
            if (!resultType.isF32())
                return emitOpError() << binaryOp << " 操作中，不同类型操作数的结果类型必须是 f32，但得到 "
                                     << resultType;
        }
    }

    // 如果是整数除法，可以添加额外警告或检查
    if (getOpType() == BinaryOpType::div) {
        if (auto rhsOp = getRhs().getDefiningOp<ConstantOp>()) {
            if (auto intValue = rhsOp.getValue().dyn_cast<IntegerAttr>()) {
                if (intValue.getInt() == 0) {
                    return emitOpError() << "除法操作的右操作数不能是整数0";
                }
            } else if (auto fpValue = rhsOp.getValue().dyn_cast<FloatAttr>()) {
                if (fpValue.getValue().isZero()) {
                    return emitOpError() << "除法操作的右操作数不能是浮点数0.0";
                }
            }
        }
    }

    return success();
}

/// 类型推断方法
LogicalResult BinaryOp::inferReturnTypes(MLIRContext *context, Optional<Location> location,
                                         ValueRange operands, DictionaryAttr attributes, RegionRange regions,
                                         SmallVectorImpl<Type> &inferredReturnTypes) {

    // 确保有两个操作数
    if (operands.size() != 2)
        return emitOptionalError(location, "期望两个操作数");

    // 获取操作数类型
    Type lhsType = operands[0].getType();
    Type rhsType = operands[1].getType();

    // 验证操作数类型为 f32 或 i32
    if (!lhsType.isF32() && !lhsType.isInteger(32))
        return emitOptionalError(location, "左操作数类型必须是 f32 或 i32");

    if (!rhsType.isF32() && !rhsType.isInteger(32))
        return emitOptionalError(location, "右操作数类型必须是 f32 或 i32");

    // 获取操作类型属性
    auto opTypeAttr = attributes.get("op_type");
    if (!opTypeAttr)
        return emitOptionalError(location, "缺少必需的'op_type'属性");

    auto intAttr = opTypeAttr.dyn_cast<mlir::IntegerAttr>();
    if (!intAttr)
        return emitOptionalError(location, "'op_type'属性必须是整数");

    BinaryOpType opType = static_cast<BinaryOpType>(intAttr.getInt());
    bool isAddSubOp = (opType == BinaryOpType::add || opType == BinaryOpType::sub);

    // 应用类型推断规则
    if (isAddSubOp) {
        // 加减法：操作数必须类型相同，结果与操作数相同
        if (lhsType != rhsType)
            return emitOptionalError(location, "加减操作的操作数类型必须相同");

        inferredReturnTypes.push_back(lhsType);
    } else {
        // 乘除法：如果操作数类型相同，结果与操作数相同
        // 如果操作数类型不同，结果为 f32
        if (lhsType == rhsType) {
            inferredReturnTypes.push_back(lhsType);
        } else {
            inferredReturnTypes.push_back(FloatType::getF32(context));
        }
    }

    return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

// 实现构建器方法
void IfOp::build(OpBuilder &builder, OperationState &result, Value condition,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
    result.addOperands(condition);

    // 创建 then 区域
    Region *thenRegion = result.addRegion();
    Block *thenBlock = new Block();
    thenRegion->push_back(thenBlock);

    // 在 then 区域内使用给定的构建器函数
    if (thenBuilder) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(thenBlock);
        thenBuilder(builder, result.location);
    }

    // 创建 else 区域
    Region *elseRegion = result.addRegion();
    Block *elseBlock = new Block();
    elseRegion->push_back(elseBlock);

    // 在 else 区域内使用给定的构建器函数
    if (elseBuilder) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(elseBlock);
        elseBuilder(builder, result.location);
    }
}

// YieldOp 构建器实现
void YieldOp::build(OpBuilder &builder, OperationState &result) {
    // 空实现，不需要添加任何内容
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

void WhileOp::build(OpBuilder &builder, OperationState &result, Value condition,
                    function_ref<void(OpBuilder &, Location)> bodyBuilder) {
    result.addOperands(condition);

    // 创建循环体区域
    Region *bodyRegion = result.addRegion();
    Block *bodyBlock = new Block();
    bodyRegion->push_back(bodyBlock);

    // 生成循环体内容
    if (bodyBuilder) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(bodyBlock);
        bodyBuilder(builder, result.location);
    }
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
    auto function = cast<FunDeclOp>((*this)->getParentOp());

    if (getNumOperands() > 1)
        return emitOpError() << "expects at most 1 return operand";

    // The operand number and types must match the function signature.
    const auto &results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size())
        return emitOpError() << "does not return the same number of values (" << getNumOperands()
                             << ") as the enclosing function (" << results.size() << ")";

    // If the operation does not have an input, we are done.
    if (!hasOperand())
        return mlir::success();

    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    // Check that the result type of the function matches the operand type.
    if (inputType == resultType)
        return mlir::success();

    return emitError() << "type of return operand (" << inputType << ") doesn't match function result type ("
                       << resultType << ")";
}

#define GET_OP_CLASSES
#include "mlir/CminusfOps.cpp.inc"

// 为了引入辅助类的声明
#define GET_ENUM_CLASSES
#include "mlir/CminusfEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/CminusfAttributes.cpp.inc"