//===- Cminusf.td - TableGen definitions for Cminusf Dialect and Ops -----------------*- tablegen -*-===//
//
// This file defines the Cminusf dialect and operations in MLIR.
// It combines the dialect definition and the operation definitions
// into a single TableGen file.
//
//===-----------------------------------------------------------------------------------------------===//

#ifndef CMINUSF_OPS
#define CMINUSF_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// 定义一个名为 "cminusf" 的 Dialect，前缀同为 "cminusf"
def Cminusf_Dialect : Dialect {
  let name = "cminusf";
  let cppNamespace = "mlir::cminusf";
}

// 定义一个 Cminusf_Op 基类，用于简化操作定义
class Cminusf_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cminusf_Dialect, mnemonic, traits>;

///===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Cminusf_ConstantOp : Cminusf_Op<"constant", [Pure]> {
  let summary = "constant integer operation";
  let description = [{
    生成一个常量整数值。
  }];

  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$res);

  let assemblyFormat = "`" value `:` type($res);
}

//===----------------------------------------------------------------------===//
// VarDeclarationOp
//===----------------------------------------------------------------------===//

def Cminusf_VarDeclarationOp : Cminusf_Op<"vardecl", [Pure]> {
  let summary = "variable declaration";
  let description = [{
    声明一个变量。
  }];

  let arguments = (ins I32:$type, StrAttr:$name, Optional<I32Attr>:$size);
  let results = (outs );
  let assemblyFormat = "`" type ` ` name (`[` size `]`)? ;
}

//===----------------------------------------------------------------------===//
// FunDeclarationOp
//===----------------------------------------------------------------------===//

def Cminusf_FunDeclarationOp : Cminusf_Op<"fundecl", [Symbol]> {
  let summary = "function declaration";
  let description = [{
    声明一个函数。
  }];

  let regions = (region any "function body");
  let traits = [IsolatedFromAbove];
  let assemblyFormat = "`" symbol-name($name) `(` `)` attr-dict
                       ":" type($function_type);

  let arguments = (ins StrAttr:$name, TypeAttr:$function_type);
  let results = (outs );
}

//===----------------------------------------------------------------------===//
// ParamOp
//===----------------------------------------------------------------------===//

def Cminusf_ParamOp : Cminusf_Op<"param", [Pure]> {
  let summary = "function parameter";
  let description = [{
    声明一个函数参数。
  }];

  let arguments = (ins I32:$type, StrAttr:$name, BoolAttr:$isarray);
  let results = (outs );
  let assemblyFormat = "`" type ` ` name (`[` `]`)? ;
}

//===----------------------------------------------------------------------===//
// CompoundStmtOp
//===----------------------------------------------------------------------===//

def Cminusf_CompoundStmtOp : Cminusf_Op<"compound", [Pure]> {
  let summary = "compound statement";
  let description = [{
    表示一个复合语句块。
  }];

  let regions = (region any "body");
  let arguments = (ins );
  let results = (outs );
  let assemblyFormat = "`" attr-dict ;
}

//===----------------------------------------------------------------------===//
// ExpressionStmtOp
//===----------------------------------------------------------------------===//

def Cminusf_ExpressionStmtOp : Cminusf_Op<"exprstmt", [Pure]> {
  let summary = "expression statement";
  let description = [{
    表示一个表达式语句。
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs );
  let assemblyFormat = "`" expr ;
}

//===----------------------------------------------------------------------===//
// SelectionStmtOp
//===----------------------------------------------------------------------===//

def Cminusf_SelectionStmtOp : Cminusf_Op<"if", [Pure]> {
  let summary = "if statement";
  let description = [{
    表示一个选择语句（if-else）。
  }];

  let arguments = (ins AnyType:$cond, AnyType:$then, Optional<AnyType>:$else);
  let results = (outs );
  let assemblyFormat = "`" cond `:` then (`else` else)?;
}

//===----------------------------------------------------------------------===//
// IterationStmtOp
//===----------------------------------------------------------------------===//

def Cminusf_IterationStmtOp : Cminusf_Op<"while", [Pure]> {
  let summary = "while statement";
  let description = [{
    表示一个迭代语句（while）。
  }];

  let arguments = (ins AnyType:$cond, AnyType:$body);
  let results = (outs );
  let assemblyFormat = "`" cond `:` body ;
}

//===----------------------------------------------------------------------===//
// ReturnStmtOp
//===----------------------------------------------------------------------===//

def Cminusf_ReturnOp : Cminusf_Op<"return", [Terminator]> {
  let summary = "return statement";
  let description = [{
    返回一个可选值，如果是函数无返回则没有操作数。
  }];

  let arguments = (ins Optional<AnyType>:$value);
  let results = (outs );
  let assemblyFormat = "`" attr-dict ( $value^ `:` type($value))?
                      ;
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

def Cminusf_AssignOp : Cminusf_Op<"assign", [Pure]> {
  let summary = "assignment operation";
  let description = [{
    表示一个赋值操作。
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$res);
  let assemblyFormat = "`" lhs `=` rhs ;
}

//===----------------------------------------------------------------------===//
// SimpleOp
//===----------------------------------------------------------------------===//

def Cminusf_SimpleOp : Cminusf_Op<"simple", [Pure]> {
  let summary = "simple expression";
  let description = [{
    表示一个简单表达式，带有关系运算符。
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I32Attr:$op);
  let results = (outs AnyType:$res);
  let assemblyFormat = "`" lhs ` op ` rhs ;
}

//===----------------------------------------------------------------------===//
// AdditiveOp
//===----------------------------------------------------------------------===//

def Cminusf_AdditiveOp : Cminusf_Op<"additive", [Pure]> {
  let summary = "additive expression";
  let description = [{
    表示一个加法表达式，带有加法运算符。
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I32Attr:$op);
  let results = (outs AnyType:$res);
  let assemblyFormat = "`" lhs ` op ` rhs ;
}

//===----------------------------------------------------------------------===//
// TermOp
//===----------------------------------------------------------------------===//

def Cminusf_TermOp : Cminusf_Op<"term", [Pure]> {
  let summary = "term expression";
  let description = [{
    表示一个项，带有乘法运算符。
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I32Attr:$op);
  let results = (outs AnyType:$res);
  let assemblyFormat = "`" lhs ` op ` rhs ;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def Cminusf_CallOp : Cminusf_Op<"call", [Pure]> {
  let summary = "function call";
  let description = [{
    表示一个函数调用。
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$args);
  let results = (outs AnyType:$res);
  let assemblyFormat = "`" name `(` args `)` ;
}