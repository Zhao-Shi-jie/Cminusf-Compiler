//===- Cminusf.h - Cminusf dialect and ops definition -----------*- C++ -*-===//
//
// This file declares the dialect and operations for a simplified Cminusf grammar example.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CMINUSF_H_
#define MLIR_DIALECT_CMINUSF_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "CminusfDialect.h.inc"

#define GET_OP_CLASSES
#include "CminusfOps.h.inc"

#endif // MLIR_DIALECT_CMINUSF_CMINUSF_H_