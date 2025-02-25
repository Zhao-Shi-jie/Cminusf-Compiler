include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Cminusf_Dialect : Dialect {
  let name = "cminusf";
  let cppNamespace = "mlir::cminusf";
}

class Cminusf_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cminusf_Dialect, mnemonic, traits>;

// 统一的常量操作，支持整数和浮点类型
def Cminusf_ConstantOp : Cminusf_Op<"constant", [Pure]> {
  let summary = "Constant value operation";
  let description = [{
    Represents a constant value that can be either integer or floating-point.
    
    该操作通过类型属性自动推导结果类型，支持：
    - 整数常量 (i32)
    - 浮点常量 (f32)
  }];

  // 使用单一属性存储常量值
  let arguments = (ins
    AnyAttrOf<[F32Attr, I32Attr]>:$value  // 携带类型和值的属性
  );

  let results = (outs
    AnyTypeOf<[F32, I32]>:$result   // 自动推导的结果类型
  );

  // 自动类型推导：结果类型取自属性类型
  let builders = [
    OpBuilder<(ins "Attribute":$value), [{
      auto type = value.getType();
      $_state.addTypes(type);
    }]>
  ];

  let hasVerifier = 1;
}


/// 变量声明
def Cminusf_VarDeclOp : Cminusf_Op<"var_decl", [Pure]> {
  let summary = "Variable declaration";
  let description = [{
    Declares a variable with a given name and type.
  }];
  
  let arguments = (ins StrAttr:$var_name, AnyTypeOf<[F32, I32]>:$var_type);
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
}

/// 函数声明
def Cminusf_FunDeclOp : Cminusf_Op<"fun_decl", [Symbol]> {
  let summary = "Function declaration";
  let description = [{
    Declares a function with a name, return type, and parameters.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
}

/// 赋值表达式 `var = expr`
def Cminusf_AssignOp : Cminusf_Op<"assign", [Pure]> {
  let summary = "Variable assignment operation";
  let description = [{
    将表达式的值赋给左值变量，要求变量和表达式类型严格匹配。
    
    示例：
      %0 = "cminusf.assign"(%var, %expr) : (i32, i32) -> i32
  }];

let arguments = (ins 
    // `var` 是变量（左值），可以是 `i32` 或 `f32`
    AnyTypeOf<[F32, I32]>:$var,
    // `expr` 必须是 `var` 的相同类型
    AnyTypeOf<[F32, I32]>:$expr
  );

  let results = (outs 
    // 赋值操作的结果类型与变量类型一致
    AnyTypeOf<[F32, I32]>:$result
  );

  // 自动类型推导构建器
  let builders = [
    OpBuilder<(ins "Value":$var, "Value":$expr), [{
      auto varType = var.getType();
      auto exprType = expr.getType();

      // 确保变量类型和表达式类型匹配
      if (varType != exprType) {
        emitError("变量类型 ") << varType 
             << " 与表达式类型 " << exprType << " 不匹配";
      }

      // 结果类型与变量类型一致
      $_state.addTypes(varType);
      $_state.addOperands({var, expr});
    }]>
  ];
}

/// 比较运算
def Cminusf_CmpOp : Cminusf_Op<"cmp", [Pure]> {
  let summary = "Comparison operation";
  let description = [{
    Performs a comparison operation like ==, !=, <, >, <=, >=.
  }];

  let arguments = (ins AnyTypeOf<[F32, I32]>:$lhs, AnyTypeOf<[F32, I32]>:$rhs, 
                   I64Attr:$cmp_predicate);
  let results = (outs I32:$result);
}

/// 算术运算 (+, -, *, /)
def Cminusf_AddOp : Cminusf_Op<"add", [Pure]> {
  let summary = "Addition operation";
  let description = [{ Performs addition of two values. }];
  let arguments = (ins AnyTypeOf<[F32, I32]>:$lhs, AnyTypeOf<[F32, I32]>:$rhs);
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
}

def Cminusf_SubOp : Cminusf_Op<"sub", [Pure]> {
  let summary = "Subtraction operation";
  let description = [{ Performs subtraction of two values. }];
  let arguments = (ins AnyTypeOf<[F32, I32]>:$lhs, AnyTypeOf<[F32, I32]>:$rhs);
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
}

def Cminusf_MulOp : Cminusf_Op<"mul", [Pure]> {
  let summary = "Multiplication operation";
  let description = [{ Performs multiplication of two values. }];
  let arguments = (ins AnyTypeOf<[F32, I32]>:$lhs, AnyTypeOf<[F32, I32]>:$rhs);
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
}

def Cminusf_DivOp : Cminusf_Op<"div", [Pure]> {
  let summary = "Division operation";
  let description = [{ Performs division of two values. }];
  let arguments = (ins AnyTypeOf<[F32, I32]>:$lhs, AnyTypeOf<[F32, I32]>:$rhs);
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
}

/// 选择语句 `if (cond) then else`
def Cminusf_IfOp : Cminusf_Op<"if", [Pure]> {
  let summary = "Conditional branch operation";
  let description = [{
    A conditional if-else statement.
  }];
  
  let arguments = (ins I32:$condition);
  let regions = (region AnyRegion:$then_branch, AnyRegion:$else_branch);
}

/// 循环语句 `while (cond) { body }`
def Cminusf_WhileOp : Cminusf_Op<"while", [Pure]> {
  let summary = "While loop operation";
  let description = [{
    A loop that executes while the condition holds true.
  }];
  
  let arguments = (ins I32:$condition);
  let regions = (region AnyRegion:$body);
}

/// 函数调用 `call foo(args)`
def Cminusf_CallOp : Cminusf_Op<"call", [Pure]> {
  let summary = "Function call operation";
  let description = [{
    Calls a function with given arguments.
  }];
  
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyTypeOf<[F32,I32]>>:$args);
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
}

/// 返回语句 `return expr`
def Cminusf_ReturnOp : Cminusf_Op<"return", [Pure]> {
  let summary = "Return statement";
  let description = [{
    Returns a value from a function.
  }];
  
  let arguments = (ins Optional<AnyTypeOf<[F32, I32]>>:$value);
}