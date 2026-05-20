---
layout: page
title: About
permalink: /about/
---

## Cminusf 编译器

基于 MLIR (Multi-Level Intermediate Representation) 实现的教学编译器。

- **源语言**：Cminusf（C 语言子集 + 浮点支持）
- **技术栈**：Flex/Bison → MLIR → LLVM
- **目标**：通过逐步添加优化 Pass 来学习 MLIR 框架

### 当前功能

| 组件 | 状态 |
|------|------|
| 词法分析 & 语法分析 | ✅ |
| AST 构建 | ✅ |
| cminusf dialect (MLIR) | ✅ |
| 降级到 standard MLIR | ✅ |
| 降级到 LLVM dialect | ✅ |
| LLVM IR 输出 | ✅ |
| 可执行文件生成 | ✅ |
| 常量折叠 Pass | ✅ |
| 常量传播 Pass | ✅ |
| 标准优化 Pass 集成 | ✅ |

### 博客

这里记录实现过程中学到的技术，包括：

- 如何添加 MLIR Pass
- OpRewritePattern 模式匹配
- Pipeline 设计
- 优化级别控制

项目地址：[github.com/Zhao-Shi-jie/Cminusf-Compiler](https://github.com/Zhao-Shi-jie/Cminusf-Compiler)
