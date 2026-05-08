# Cminusf 编译器原型

这个项目是一个基于 MLIR 的 Cminusf 编译器原型，用来学习编译基础和MLIR框架
## 主要流程

Cminusf 编译器的主要流程如下：

```
Cminusf 源码
    -> 解析器 / AST
    -> 自定义 Cminusf Dialect MLIR
    -> 标准 MLIR Dialects (func / arith / memref / scf / cf)
    -> LLVM Dialect
    -> LLVM IR
    -> 使用 Clang 生成 x86 可执行文件
```

## 构建和运行

```bash
# 构建项目
mkdir build && cd build
cmake ..
make -j

# 编译并运行示例程序
./build/cminusfc --emit-exe ./tests/<program_name>.cminus
./tests/<program_name>
```

Cminusf 源自 [USTC 编译器课程实验](https://github.com/USTC-Compiler-2025/homepage)，本项目复用了该实验的词法分析、语法分析和 AST 部分。

## 依赖
- LLVM (commit: **21f4b84c456b471cc52016cf360e14d45f7f2960**)

## TODO
- 在多个层级添加优化 Pass