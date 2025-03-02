#ifndef CMINUSF_OPS
#define CMINUSF_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"

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
    AnyAttrOf<[F32Attr, I32Attr]>:$value
  );

  let results = (outs
    AnyTypeOf<[F32, I32]>:$result
  );

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";

  // 自动类型推导：结果类型取自属性类型
  let builders = [
    OpBuilder<(ins "int":$value), [{
        build($_builder, $_state, $_builder.getIntegerAttr($_builder.getI32Type(), value));
    }]>,
    OpBuilder<(ins "float":$value), [{
        build($_builder, $_state, $_builder.getF32FloatAttr(value));
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
  
  let arguments = (ins
    /*属性的类型:属性的名称（标识符）
      名称将用于生成访问方法的基础，ex:
      getVarNameAttr();getVarTypeAttrAttr();getSizeAttr();
      getVarName();getSize();没有直接var访问方法，因为是混合类型
    */
    StrAttr:$var_name, 
    AnyAttrOf<[F32Attr, I32Attr]>:$var_type_attr,
    OptionalAttr<I32Attr>:$size
  );

  let results = (outs AnyTypeOf<[F32, I32, F32MemRef, I32MemRef]>:$result);

  let builders = [
    /*
      参数中的字符串指定build的参数类型
    */
    // 标量变量
    OpBuilder<(ins "StringAttr":$var_name, "Type":$var_type), [{
      $_state.addAttribute("var_name", var_name);
      $_state.addAttribute("var_type", TypeAttr::get(var_type));

      // 结果类型是标量类型
      $_state.addTypes(var_type);
    }]>,
    
    // 数组变量: 额外接受 `size`
    OpBuilder<(ins "StringAttr":$var_name, "Type":$var_type, "IntegerAttr":$size), [{
      $_state.addAttribute("var_name", var_name);
      $_state.addAttribute("var_type", TypeAttr::get(var_type));
      $_state.addAttribute("size", size);

      // 结果类型是 MemRef
      $_state.addTypes(MemRefType::get({size.getInt()}, var_type));
    }]>
  ];

  // let assemblyFormat = [{
  //   `(` $var_name `:` $var_type_attr (`,` `size` `=` $size^)? `)` attr-dict `:` type($result)
  // }];
  // let assemblyFormat = [{
  //   `(` $var_name (`,` `size` `=` $size^)? `)` attr-dict `:` type($result)
  // }];
  let assemblyFormat = [{
    $var_name (`[` $size^ `]`)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

/// 函数声明
def Cminusf_FunDeclOp : Cminusf_Op<"fun_decl", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "Function declaration";
  let description = [{
    Declares a function with a name, return type, and parameters.

    Example:
    ```mlir
    cminusf.fun_decl @factorial(%n: i32) -> i32 {
      // function body
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
  
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  // let assemblyFormat = [{
  //   $sym_name `(` `:` $function_type `)` (`arg_attrs` $arg_attrs^)? (`res_attrs` $res_attrs^)? attr-dict-with-keyword $body
  // }];
}

/// 赋值表达式 `var = expr`
def Cminusf_AssignOp : Cminusf_Op<"assign", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Variable assignment operation";
  let description = [{
    将表达式的值赋给左值变量，要求变量和表达式类型严格匹配。
    
    示例：
      %result = cminusf.assign %var = %expr : i32
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
      assert(varType == exprType && "变量类型与表达式类型不匹配");

      // 结果类型与变量类型一致
      $_state.addTypes(varType);
      $_state.addOperands({var, expr});
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    LogicalResult inferReturnTypes(MLIRContext *context,
                                  std::optional<Location> location,
                                  ValueRange operands,
                                  DictionaryAttr attributes,
                                  RegionRange regions,
                                  SmallVectorImpl<Type> &inferredReturnTypes);
  }];
}

// 首先，在文件开头定义比较谓词枚举
def Cminusf_CmpPredicateEQ : I32EnumAttrCase<"eq", 0>;
def Cminusf_CmpPredicateNE : I32EnumAttrCase<"ne", 1>;
def Cminusf_CmpPredicateLT : I32EnumAttrCase<"lt", 2>;
def Cminusf_CmpPredicateLE : I32EnumAttrCase<"le", 3>;
def Cminusf_CmpPredicateGT : I32EnumAttrCase<"gt", 4>;
def Cminusf_CmpPredicateGE : I32EnumAttrCase<"ge", 5>;

def Cminusf_CmpPredicate : I32EnumAttr<"CmpPredicate", "comparison predicate",
  [Cminusf_CmpPredicateEQ, Cminusf_CmpPredicateNE, Cminusf_CmpPredicateLT, 
   Cminusf_CmpPredicateLE, Cminusf_CmpPredicateGT, Cminusf_CmpPredicateGE]> {
  let cppNamespace = "::cminusf::mlir";
}

/// 比较运算
def Cminusf_CmpOp : Cminusf_Op<"cmp", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Comparison operation";
  let description = [{
    Performs a comparison operation like ==, !=, <, >, <=, >=.
    
    The operation takes two operands of the same type (either f32 or i32)
    and returns an i32 result (1 for true, 0 for false).
    
    Example:
    ```mlir
    %result = cminusf.cmp eq %lhs, %rhs : i32
    %result = cminusf.cmp lt %lhs, %rhs : f32
    ```
    
    Available predicates:
    - eq: equal (==)
    - ne: not equal (!=)
    - lt: less than (<)
    - le: less than or equal (<=)
    - gt: greater than (>)
    - ge: greater than or equal (>=)
  }];

  let arguments = (ins AnyTypeOf<[F32, I32]>:$lhs,
                       AnyTypeOf<[F32, I32]>:$rhs,
                       Cminusf_CmpPredicate:$predicate);
  
  let results = (outs I32:$result);
  
  // 添加自定义验证
  let hasVerifier = 1;
  
  // 添加自定义汇编格式
  let hasCustomAssemblyFormat = 1;
  
  // 为构建器添加方便的方法
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "CmpPredicate":$predicate), [{
      build($_builder, $_state, $_builder.getI32Type(), lhs, rhs, 
            CmpPredicateAttr::get($_builder.getContext(), predicate));
    }]>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "StringRef":$predicate), [{
      build($_builder, $_state, $_builder.getI32Type(), lhs, rhs, 
            CmpPredicateAttr::get($_builder.getContext(), 
              stringToCmpPredicate(predicate)));
    }]>
  ];
  
  // 添加额外方法
  let extraClassDeclaration = [{
    /// 从字符串获取谓词
    static CmpPredicate stringToCmpPredicate(StringRef str);
    
    /// 获取谓词的反向谓词: lt 的反向是 ge
    CmpPredicate getInversePredicate();
    
    /// 检查两个操作数是否为同一类型
    bool isSameOperandsType();
  }];
}

/// 算术运算 (+, -, *, /)
// 定义二元操作类型枚举
def Cminusf_BinaryOpTypeAdd : I32EnumAttrCase<"add", 0>;
def Cminusf_BinaryOpTypeSub : I32EnumAttrCase<"sub", 1>;
def Cminusf_BinaryOpTypeMul : I32EnumAttrCase<"mul", 2>;
def Cminusf_BinaryOpTypeDiv : I32EnumAttrCase<"div", 3>;

def Cminusf_BinaryOpType : I32EnumAttr<"BinaryOpType", "Binary operation type",
  [Cminusf_BinaryOpTypeAdd, Cminusf_BinaryOpTypeSub, Cminusf_BinaryOpTypeMul, Cminusf_BinaryOpTypeDiv]> {
  let cppNamespace = "::cminusf::mlir";
}

def Cminusf_BinaryOp : Cminusf_Op<"binary", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  
  let summary = "Binary operation";
  let description = [{
    Performs binary operations (add, sub, mul, div) on two values.
    
    The operation type is specified as the first argument.
    
    Type rules:
    - For add and sub: Both operands must have the same type (either f32 or i32).
      The result will have that same type.
    - For mul and div: 
      * If operands have the same type, the result will have that same type.
      * If operands have different types, the result will be f32.
    
    Example:
    ```mlir
    // Addition (same type operands)
    %result = cminusf.binary add, %a, %b : i32
    
    // Subtraction (same type operands)
    %result = cminusf.binary sub, %a, %b : f32
    
    // Multiplication (same or different type operands)
    %result = cminusf.binary mul, %a, %b : i32
    
    // Division (using mixed types will produce f32 internally)
    %result = cminusf.binary div, %a, %f : f32
    ```
  }];
  
  let arguments = (ins 
      Cminusf_BinaryOpType:$op_type,
      AnyTypeOf<[F32, I32]>:$lhs,
      AnyTypeOf<[F32, I32]>:$rhs
  );
  
  let results = (outs AnyTypeOf<[F32, I32]>:$result);
  
  // 添加自定义验证
  let hasVerifier = 1;
  
  // 添加自定义汇编格式 
  let hasCustomAssemblyFormat = 1;
  
  // 为构建器添加方便的方法
  let builders = [
    // 基本构建器
    OpBuilder<(ins "BinaryOpType":$op_type, "Value":$lhs, "Value":$rhs), [{
      Type lhsType = lhs.getType();
      Type rhsType = rhs.getType();
      Type resultType;
      
      // 判断操作类型和应用相应的类型规则
      if (op_type == BinaryOpType::add || op_type == BinaryOpType::sub) {
        // 加减法: 操作数必须相同类型，结果与操作数相同
        assert(lhsType == rhsType && "加减操作的操作数类型必须相同");
        resultType = lhsType;
      } else {
        // 乘除法: 同类型时结果相同，不同类型时结果为f32
        if (lhsType == rhsType) {
          resultType = lhsType;
        } else {
          resultType = $_builder.getF32Type();
        }
      }
      
      build($_builder, $_state, resultType, op_type, lhs, rhs);
    }]>,
    
    // 字符串构建器 - 从操作符字符串创建
    OpBuilder<(ins "StringRef":$op_type_str, "Value":$lhs, "Value":$rhs), [{
      // 从字符串构建 - 转换操作类型字符串为枚举
      BinaryOpType op_type = stringToBinaryOpType(op_type_str);
      
      // 调用上面定义的构建器
      build($_builder, $_state, op_type, lhs, rhs);
    }]>
  ];
  
  let extraClassDeclaration = [{
    /// 检查两个操作数是否为同一类型
    bool isSameOperandsType() {
      return getLhs().getType() == getRhs().getType();
    }
    
    /// 检查是否为加法或减法操作
    bool isAddOrSubOp() {
      return getOpType() == BinaryOpType::add || getOpType() == BinaryOpType::sub;
    }
    
    /// 检查是否为乘法或除法操作
    bool isMulOrDivOp() {
      return getOpType() == BinaryOpType::mul || getOpType() == BinaryOpType::div;
    }
    
    /// 检查结果类型是否有效
    bool hasValidResultType();
    
    /// 将字符串转换为二元操作类型枚举
    static BinaryOpType stringToBinaryOpType(StringRef str);
    
    /// 获取操作类型的字符串表示
    StringRef getOperationSymbol();
  }];
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

#endif // CMINUSF_OPS