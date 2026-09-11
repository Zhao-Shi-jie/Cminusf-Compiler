# Cminusf 编译器原型

这个项目是一个基于 MLIR 的 Cminusf 编译器原型，用来学习编译基础和 MLIR 框架。

## 主要流程

Cminusf 编译器的主要流程如下：

```
Cminusf 源码
    -> 解析器 / AST
    -> 自定义 Cminusf Dialect MLIR
    -> 标准 MLIR Dialects (func / arith / memref / scf / cf)
    -> LLVM Dialect
    -> LLVM IR
    -> 使用系统 Clang 生成 x86 可执行文件
```

## 依赖

- CMake >= 3.16
- Flex
- Bison
- Ninja（用于构建 LLVM）
- 系统 Clang

## 获取源码

LLVM 源码以 Git 子模块的形式固定在项目所需的提交。首次克隆时需要同时初始化子模块：

```bash
git clone --recurse-submodules https://github.com/Zhao-Shi-jie/Cminusf-Compiler.git
cd Cminusf-Compiler
```

如果已经克隆了主仓库，可补充初始化子模块：

```bash
git submodule update --init --recursive
```

## 构建 LLVM 和 MLIR

构建本项目前，先单独构建并安装子模块中的 LLVM 和 MLIR：

```bash
./scripts/build-llvm.sh
```

该脚本不会构建 Clang。LLVM 的构建树位于
`3rdparty/llvm-project/build/`，安装产物位于
`3rdparty/llvm-project/build/install/`。

## 构建和运行

```bash
cmake -S . -B build
cmake --build build --parallel

# 可选：安装到 build/install
cmake --install build

# 编译并运行示例程序
./build/cminusfc --emit-exe tests/<program_name>.cminus
./tests/<program_name>
```

CMake 只会使用子模块 `build/install` 中的 LLVM 和 MLIR，不会回退到系统中
的其他 LLVM 安装。生成汇编、目标文件和可执行文件时使用配置阶段在 `PATH`
中找到的系统 Clang。

Cminusf 源自 [USTC 编译器课程实验](https://github.com/USTC-Compiler-2025/homepage)，本项目复用了该实验的词法分析、语法分析和 AST 部分。

## TODO

- 在多个层级添加优化 Pass
