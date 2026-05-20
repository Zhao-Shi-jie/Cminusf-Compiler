---
layout: post
title: "从零给 Cminusf 编译器添加常量折叠 Pass"
date: 2026-05-20
categories: [MLIR, Compiler]
tags: [MLIR, Pass, ConstantFolding, PatternRewrite, Pipeline]
---

## 背景

[Cminusf-Compiler](https://github.com/Zhao-Shi-jie/Cminusf-Compiler) 是一个基于 MLIR 的编译器，支持将 Cminusf 语言编译为可执行文件。当前的 pipeline 是：

```
.cminus → AST → cminusf dialect → standard MLIR → LLVM dialect → LLVM IR → binary
```

在实现常量折叠之前，pipeline 中**没有任何优化 pass**。编译器忠实地将前端生成的每一条 IR 指令都生成机器码，包括大量编译时就能求值的冗余运算。

本文记录添加第一个优化 Pass —— **常量折叠（Constant Folding）** 的完整过程。

---

## 1. 常量折叠是什么

常量折叠是编译优化中最基础的 pass。它的思想很简单：**如果操作数都是编译时已知的常量，就在编译时求值，用结果替换原来的运算**。

例如编译器前端生成的 IR：

```mlir
%c2 = cminusf.constant 2 : i32
%c3 = cminusf.constant 3 : i32
%0 = cminusf.binary add, %c2, %c3 : i32
```

这里的 `2 + 3` 完全可以在编译时算出 `5`，不需要生成运行时的加法指令。常量折叠的目标就是把它变成：

```mlir
%0 = cminusf.constant 5 : i32
```

---

## 2. MLIR Pass 是什么

在 MLIR 中，一个 Pass 是对 IR 的一次**变换（transformation）**或**分析（analysis）**。每个 pass 需要提供：

| 要素 | 说明 |
|------|------|
| **argument** | 命令行中标识此 pass 的字符串，如 `"cminusf-const-fold"` |
| **description** | 人类可读的说明 |
| **作用域** | 在什么级别的 IR 上运行，如 `OperationPass<ModuleOp>` |
| **核心逻辑** | `runOnOperation()` 方法 |

### PassWrapper & OperationPass

```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyPass)

  StringRef getArgument() const final { return "my-pass"; }
  StringRef getDescription() const final { return "这是我的第一个 Pass"; }

  void runOnOperation() final {
    // 在这里实现 Pass 逻辑
  }
};
```

- **`PassWrapper`** 是 CRTP 包装器，自动生成样板代码
- **`OperationPass<ModuleOp>`** 表示这个 pass 作用在整个 `ModuleOp` 上。也可以写成 `OperationPass<FuncOp>` 来按函数作用
- **`MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID`** 为 pass 生成唯一类型 ID，MLIR 用它来标识 pass

### 两种编写方式

| 方式 | 适用场景 | 核心技术 |
|------|---------|---------|
| **Pattern-based** | 局部变换（常量折叠、代数化简） | `OpRewritePattern` + `GreedyPatternRewriteDriver` |
| **Manual traversal** | 全局分析、跨操作变换 | `module.walk([&](Operation *op) {...})` |

本文的常量折叠使用**第一种方式**。

---

## 3. OpRewritePattern 原理

`OpRewritePattern` 是 MLIR 的模式匹配引擎核心：

```
继承 OpRewritePattern<MyOp>
    │
    └── matchAndRewrite(MyOp op, PatternRewriter &rewriter)
            │
            ├── ① 检查匹配条件（match）
            │     不匹配 → return failure()  // 跳过
            │
            ├── ② 执行替换操作（rewrite）
            │     rewriter.create<NewOp>(...)  // 创建新操作
            │     rewriter.replaceOp(op, val)  // 替换旧操作
            │
            └── ③ return success()  // 告知框架 IR 已修改
```

每个 pattern 关注一种特定的 IR 模式。当框架发现匹配时，替换操作会自动触发 use-def 链的更新。

---

## 4. 实现四个 Pattern

### 4.1 Pattern 1：折叠常量二元运算

**触发模式**：`cminusf.binary {add|sub|mul|div}, constant(a), constant(b)`

```cpp
struct FoldConstantBinaryOp : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    // 检查两个操作数是否都是 constant
    auto lhsConst = op.getLhs().getDefiningOp<ConstantOp>();
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (!lhsConst || !rhsConst)
      return failure();          // 不匹配：至少一个操作数不是常量

    // 提取常量值
    Attribute lhsVal = lhsConst.getValue();
    Attribute rhsVal = rhsConst.getValue();

    // 整数运算：提取值 → 编译时求值 → 生成新常量
    if (lhsVal.isa<IntegerAttr>() && rhsVal.isa<IntegerAttr>()) {
      int64_t l = lhsVal.cast<IntegerAttr>().getInt();
      int64_t r = rhsVal.cast<IntegerAttr>().getInt();
      int64_t v;
      switch (op.getOpType()) {
      case BinaryOpType::add: v = l + r; break;
      case BinaryOpType::sub: v = l - r; break;
      case BinaryOpType::mul: v = l * r; break;
      case BinaryOpType::div:
        if (r == 0) return failure();  // 除零，拒绝折叠
        v = l / r; break;
      }
      Value result = rewriter.create<ConstantOp>(op.getLoc(), static_cast<int>(v));
      rewriter.replaceOp(op, result);
      return success();
    }

    // 浮点运算同理（代码略，详见源码）
    ...
  }
};
```

**关键 API 解读**：

| API | 含义 |
|-----|------|
| `op.getLhs().getDefiningOp<ConstantOp>()` | 沿 use-def 链回溯，检查操作数是否由 `ConstantOp` 定义 |
| `rewriter.create<OpType>(loc, args...)` | 在 IR 中创建新操作 |
| `rewriter.replaceOp(oldOp, newValue)` | 用新值替换旧操作的所有使用，然后删除旧操作 |
| `return failure()` | 告知框架：当前 pattern 不匹配，无需修改 IR |
| `return success()` | 告知框架：IR 已被修改，需要重新运行 pattern 集 |

### 4.2 Pattern 2：代数恒等化简

**触发模式**：

- `add(x, 0)` 或 `sub(x, 0)` → `x`
- `mul(x, 1)` 或 `div(x, 1)` → `x`

```cpp
struct FoldBinaryIdentityOp : public OpRewritePattern<BinaryOp> {
  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const override {
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (!rhsConst) return failure();

    auto attr = rhsConst.getValue();
    bool isZero = /* int 0 或 float 0.0 */;
    bool isOne  = /* int 1 或 float 1.0 */;

    // add(x, 0) 或 sub(x, 0) → x
    if ((isAddOrSub) && isZero) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    // mul(x, 1) 或 div(x, 1) → x
    if ((isMulOrDiv) && isOne) {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};
```

**设计选择**：为什么只检查右操作数？因为 `add(constant(0), x)` 这种左操作数是常量的情况，会在贪婪引擎的另一次迭代中被 `FoldConstantBinaryOp` 先折叠为 `x`，然后再被本 pattern 消除。

### 4.3 Pattern 3：相同操作数的比较

**触发模式**：`cminusf.cmp {eq|ne|lt|le|gt|ge} %x, %x`

```cpp
struct FoldCmpSameOperand : public OpRewritePattern<CmpOp> {
  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getLhs() != op.getRhs())
      return failure();       // 不是同一个 SSA 值，不匹配

    Value result;
    switch (op.getPredicate()) {
    case CmpPredicate::eq: case CmpPredicate::le: case CmpPredicate::ge:
      result = rewriter.create<ConstantOp>(op.getLoc(), 1); break;  // 恒为真
    case CmpPredicate::ne: case CmpPredicate::lt: case CmpPredicate::gt:
      result = rewriter.create<ConstantOp>(op.getLoc(), 0); break;  // 恒为假
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
```

这个优化在一些场景下很重要——例如函数内联或循环展开之后，经常出现 `a[i] == a[i]` 这样的比较，可以全部消除。

### 4.4 Pattern 4：两个常量的比较

**触发模式**：`cminusf.cmp {eq|ne|lt|le|gt|ge}, constant(a), constant(b)`

```cpp
struct FoldConstantCmpOp : public OpRewritePattern<CmpOp> {
  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsConst = op.getLhs().getDefiningOp<ConstantOp>();
    auto rhsConst = op.getRhs().getDefiningOp<ConstantOp>();
    if (!lhsConst || !rhsConst) return failure();

    // 编译时计算比较结果
    bool ret = evalCmp(op.getPredicate(), lhsVal, rhsVal);

    Value folded = rewriter.create<ConstantOp>(op.getLoc(), ret ? 1 : 0);
    rewriter.replaceOp(op, folded);
    return success();
  }
};
```

---

## 5. 组装成 Pass

将四个 pattern 注册到 `RewritePatternSet`，然后交给 `GreedyPatternRewriteDriver`：

```cpp
struct CminusfConstantFoldingPass
    : public PassWrapper<CminusfConstantFoldingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CminusfConstantFoldingPass)

  StringRef getArgument() const final { return "cminusf-const-fold"; }
  StringRef getDescription() const final {
    return "Apply constant folding and algebraic simplifications on cminusf ops";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<FoldConstantBinaryOp, FoldBinaryIdentityOp,
                 FoldCmpSameOperand, FoldConstantCmpOp>(&getContext());

    GreedyRewriteConfig config;
    config.maxIterations = GreedyRewriteConfig::kNoLimit;  // 迭代到收敛

    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
      signalPassFailure();
  }
};
```

**`GreedyRewriteDriver` 的工作方式**：

```
① 遍历 IR 中的所有操作（post-order）
② 对每个操作尝试所有 pattern
③ 一旦有 pattern 匹配成功 → IR 被修改 → 标记受影响的操作重新处理
④ 重复直到没有任何 pattern 能匹配（收敛）
```

之所以叫"贪心"（Greedy），因为它会反复应用 pattern 直到没有新的匹配，实现多个 pattern 的**级联效应**——前一个 pattern 的结果可能触发后一个 pattern 的新匹配。

---

## 6. 注册 Pass & 加入 Pipeline

### 6.1 头文件声明

```cpp
// include/mlir/Passes.h
std::unique_ptr<Pass> createCminusfConstantFoldingPass();
```

### 6.2 Pass 注册

```cpp
// src/mlir/LoweringPasses.cpp
void registerCminusfPasses() {
    // ...
    PassRegistration<CminusfConstantFoldingPass>();
}
```

`PassRegistration<T>` 是**静态注册**——程序启动时将 pass 工厂函数注册到全局 pass 表，使得可以通过命令行 `--pass-pipeline` 等方式按名称调用 pass。

### 6.3 加入优化 Pipeline

```cpp
// src/cminusfc/main.cpp
if (optLevel >= 1) {
    pm.addPass(createCminusfConstantFoldingPass());      // 常量折叠
    pm.addPass(createCminusfConstantPropagationPass());   // 常量传播
}
pm.addPass(createLowerCminusfToStandardPass());           // 降级
if (optLevel >= 1) {
    pm.addPass(mlir::createCanonicalizerPass());          // 标准规范化
    pm.addPass(mlir::createCSEPass());                    // 公共子表达式消除
}
```

`PassManager::addPass()` 按顺序添加 pass。各个 pass 依次执行，前一个的输出是后一个的输入。

---

## 7. 优化效果

### 测试程序

```c
int main(void) {
    int a, b, c;
    a = 2 + 3;
    b = a - 0;
    c = 6 * 2;
    if (1 == 1)
        output(a);
    else
        output(0);
    return 0;
}
```

### 优化前 vs 优化后

**优化前**（`--emit-mlir`）：

```mlir
%3 = cminusf.constant 3 : i32
%4 = cminusf.constant 2 : i32
%5 = cminusf.binary add, %4, %3 : i32      // 运行时做 2+3
cminusf.store %5, %0 : i32, memref<1xi32>
...
%12 = cminusf.constant 1 : i32
%13 = cminusf.constant 1 : i32
%14 = cminusf.cmp eq %12, %13 : i32        // 运行时做 1==1
cminusf.if %14{ ... }                       // 动态条件分支
```

**优化后**（`--emit-mlir -O1`）：

```mlir
%3 = cminusf.constant 5 : i32              // 2+3 → 5 ✓
cminusf.store %3, %0 : i32, memref<1xi32>
%4 = cminusf.constant 5 : i32              // 常量传播：a → 5 ✓
cminusf.store %4, %1 : i32, memref<1xi32>  // 恒等化简：5-0 → 5 ✓
%5 = cminusf.constant 12 : i32             // 6*2 → 12 ✓
cminusf.store %5, %2 : i32, memref<1xi32>
%6 = cminusf.constant 1 : i32              // 1==1 → 1 ✓
cminusf.if %6{ ... }                       // if 恒为真，else 分支变死代码
```

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| binary 操作 | 3 | 0 |
| cmp 操作 | 1 | 0 |
| 运行时算术运算 | 4 | 0 |

### 运行验证

```bash
$ ./cminusfc -O1 test.cminus | clang -x ir - src/io/io.c -o test && ./test
5      # 输出正确
```

---

## 8. 文件修改清单

| 文件 | 修改内容 |
|------|---------|
| `include/mlir/Passes.h` | 声明 `createCminusfConstantFoldingPass()` |
| `src/mlir/LoweringPasses.cpp` | 4 个 pattern 类 + pass 类 + `PassRegistration` |
| `src/cminusfc/main.cpp` | 将 pass 加入 `-O1` 优化 pipeline |
| `docs/constant-folding-pass.md` | 本文档 |

完整源码：[GitHub](https://github.com/Zhao-Shi-jie/Cminusf-Compiler)

---

## 9. 踩坑记录

### 坑 1：`IntegerAttr::getInt()` 返回 `int64_t`

`ConstantOp::build(builder, state, int)` 接受 `int`（32 位），而 `IntegerAttr::getInt()` 返回 `int64_t`。直接传入会导致重载歧义（在 `int` 和 `float` 之间无法选择）。需要显式转换：

```cpp
builder.create<ConstantOp>(loc, static_cast<int>(intAttr.getInt()));
```

### 坑 2：`replaceOp` 在 MLIR 17 中的 API 变化

MLIR 17 的 `replaceOp` 接受 `ValueRange` 而非 `Value`。虽然 `Value` 可隐式转换为 `ValueRange`，但 `Operation`（如 `ConstantOp`）不可直接用于 `replaceOp`：

```cpp
// 错误：ConstantOp → ValueRange 没有隐式转换
rewriter.replaceOp(op, rewriter.create<ConstantOp>(loc, 5));

// 正确：先保存为 Value，再传入
Value folded = rewriter.create<ConstantOp>(loc, 5);
rewriter.replaceOp(op, folded);
```

### 坑 3：除零检查

折叠除法时，必须检查除数是否为零。如果是零则 `return failure()` 跳过折叠——运行时除以零是未定义行为，编译时拒绝折叠更安全。

---

## 10. 下一步

这个 pass 只是在 cminusf 级别做常量折叠。后续可以继续添加：

- **Canonicalization patterns**：将 pattern 注册为 dialect 的 canonicalization patterns，复用 MLIR 内置的 `canonicalize` pass
- **常量传播**：将常量值传播到所有使用点
- **死代码消除（DCE）**：删除无用的操作和变量
- **公共子表达式消除（CSE）**：消除重复计算

---

## 参考

- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [MLIR Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)
