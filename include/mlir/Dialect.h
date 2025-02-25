#ifndef MLIR_DIALECT_CMINUSF_H_
#define MLIR_DIALECT_CMINUSF_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/CminusfDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/CminusfOps.h.inc"

#endif // MLIR_DIALECT_CMINUSF_CMINUSF_H_