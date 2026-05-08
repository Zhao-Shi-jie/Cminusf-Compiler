# Cminusf MLIR Pipeline

The intended compiler pipeline is:

```text
Cminusf source
  -> cminusf dialect IR
  -> standard MLIR dialects
  -> LLVM dialect
  -> LLVM IR
  -> x86 assembly/object/executable via clang
```

Current implementation status:

- `cminusfc -emit-mlir` now uses the real MLIR builder and emits a
  verifier-checked cminusf dialect module.
- `cminusfc -emit-standard-mlir` runs the project pass manager with
  `lower-cminusf-to-standard` and prints the standard MLIR module.
- `cminusfc -emit-llvm-dialect` runs `lower-cminusf-to-standard` followed by
  `lower-standard-to-llvm-dialect`, then prints an LLVM dialect module.
- `cminusfc -emit-llvm` now uses the MLIR pipeline, translates the LLVM dialect
  module through MLIR's LLVM IR exporter, and prints textual LLVM IR.
- `cminusfc -S`, `cminusfc -c`, `cminusfc --emit-obj`, and
  `cminusfc --emit-exe` also use the MLIR-backed LLVM IR path before invoking
  clang.
- `src/mlir/MLIRGen.cpp` now contains the MLIR-builder path that constructs a
  verifier-checked `mlir::ModuleOp` from the Cminusf AST.
- `src/mlir/LoweringPasses.cpp` registers the MLIR pass entry points.  The
  `lower-cminusf-to-standard` pass lowers cminusf ops into
  standard MLIR dialects: `fun_decl`, `call`, `constant`, `binary`, `cmp`,
  `global`, `var`, `load`, `store`, `subscript`, `if`, `while`, and `return`.
  The `lower-standard-to-llvm-dialect` pass lowers those standard dialects to
  LLVM dialect.
- `cminusf.while` now has two regions: a condition region that yields an `i32`
  condition and a body region that yields no value.  This is intentional: a
  condition operand would evaluate before the loop once, while a condition
  region preserves Cminusf `while` semantics by recomputing the condition on
  every iteration.  The lowering pass maps safe loops to `scf.while`.
- Conditionals with early returns are lowered to standard CFG using `cf.cond_br`
  blocks and `func.return`.  Loops that contain nested conditionals or early
  returns are also lowered to CFG so a return can exit the enclosing function,
  not merely the loop body.
- All autogen samples currently pass `cminusfc -emit-standard-mlir`, and the
  lowered output no longer contains `cminusf.` operations for those samples.
- All autogen samples also pass `cminusfc -emit-llvm-dialect`; the checked
  output contains only LLVM dialect operations plus the builtin module wrapper.
- All autogen samples pass MLIR-backed `cminusfc -emit-llvm` and
  `cminusfc --emit-exe`.
- `include/mlir/Ops.td` contains the initial operation set needed for memory
  and CFG lowering: `global`, `var`, `load`, `store`, `subscript`, `br`, and
  `cond_br`.
- Runtime declarations such as `input`, `output`, `outputFloat`, and
  `neg_idx_except` are kept as external LLVM declarations so they can be linked
  from `src/io/io.c`.
- MLIR/LLVM is expected to come from
  `/data/home/zsj/MLIR-Related/llvm-project/build/cminusf-dev-install`, which
  was rebuilt with consistent RTTI settings for this out-of-tree compiler.

Next required step:

- Add focused regression tests for the MLIR-backed pipeline commands.
