#include "cminusf_builder.hpp"
#include "Constant.hpp"
#include "GlobalVariable.hpp"
#include "Type.hpp"
#include "Value.hpp"
#include "ast.hpp"
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#define CONST_FP(num) ConstantFP::get((float)num, module.get())
#define CONST_INT(num) ConstantInt::get(num, module.get())

// types
Type *VOID_T;
Type *INT1_T;
Type *INT32_T;
Type *INT32PTR_T;
Type *FLOAT_T;
Type *FLOATPTR_T;

/*
 * use CMinusfBuilder::Scope to construct scopes
 * scope.enter: enter a new scope
 * scope.exit: exit current scope
 * scope.push: add a new binding to current scope
 * scope.find: find and return the value bound to the name
 */

Value* CminusfBuilder::visit(ASTProgram &node) {
    VOID_T = module->get_void_type();
    INT1_T = module->get_int1_type();
    INT32_T = module->get_int32_type();
    INT32PTR_T = module->get_int32_ptr_type();
    FLOAT_T = module->get_float_type();
    FLOATPTR_T = module->get_float_ptr_type();

    Value *ret_val = nullptr;
    for (auto &decl : node.declarations) {
        ret_val = decl->accept(*this);
    }
    return ret_val;
}

Value* CminusfBuilder::visit(ASTNum &node) {
    Value *num_val;
    context.numType = nullptr;
    if (node.type == TYPE_INT) {
        context.numType = INT32_T;
        context.Integer = node.i_val;
        num_val = CONST_INT(node.i_val);
    } else if (node.type == TYPE_FLOAT) {
        context.numType = FLOAT_T;
        num_val = CONST_FP(node.f_val);
    } else {
        assert(false && "Unknown type in ASTNum");
        return nullptr;
    }
    return num_val;
}

Value* CminusfBuilder::visit(ASTVarDeclaration &node) {
    Type *var_type = nullptr;
    if (node.type == TYPE_INT) {
        var_type = INT32_T;
    } else if (node.type == TYPE_FLOAT) {
        var_type = FLOAT_T;
    }

    if (scope.in_global()) {
        auto init = ConstantZero::get(INT32_T, builder->get_module());
        if (node.num == nullptr) {
            auto *varAlloca = (var_type != nullptr) ? GlobalVariable::create(node.id, 
                                builder->get_module(), var_type, false, init)
                                : nullptr;
            scope.push(node.id, varAlloca);
        } else {
            node.num->accept(*this);
            if (context.numType == FLOAT_T || context.Integer <= 0) {
                builder->create_call(scope.find("neg_idx_except"), std::vector<Value *> {});  
            }
            auto arrayType = ArrayType::get(var_type, context.Integer);
            auto arrayAlloca = GlobalVariable::create(node.id, builder->get_module(), 
                                                                            arrayType, false, init);
            scope.push(node.id, arrayAlloca);
        }
    } else {
        if (node.num == nullptr) {
            auto *varAlloca = (var_type != nullptr) ? builder->create_alloca(var_type) : nullptr;
            scope.push(node.id, varAlloca);
        } else {
            node.num->accept(*this);
            if (context.numType == FLOAT_T || context.Integer <= 0) {
                builder->create_call(scope.find("neg_idx_except"), std::vector<Value *> {});  
            }
            auto arrayType = ArrayType::get(var_type, context.Integer);
            auto arrayAlloca = builder->create_alloca(arrayType);
            scope.push(node.id, arrayAlloca);
        }
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTFunDeclaration &node) {
    FunctionType *fun_type;
    Type *ret_type;
    std::vector<Type *> param_types;
    std::vector<Value *> params;
    if (node.type == TYPE_INT)
        ret_type = INT32_T;
    else if (node.type == TYPE_FLOAT)
        ret_type = FLOAT_T;
    else
        ret_type = VOID_T;

    for (auto &param : node.params) {
        // TODO: Please accomplish param_types.
        Value *tmp = param->accept(*this);
        params.push_back(tmp);
        param_types.push_back(tmp->get_type());
    }

    fun_type = FunctionType::get(ret_type, param_types);
    auto func = Function::create(fun_type, node.id, module.get());
    scope.push(node.id, func);
    context.func = func;
    auto funBB = BasicBlock::create(module.get(), "entry", func);
    builder->set_insert_point(funBB);
    scope.enter();
    std::vector<Value *> args;
    for (auto &arg : func->get_args()) {
        args.push_back(&arg);
    }
    for (unsigned int i = 0; i < node.params.size(); ++i) {
        // TODO: You need to deal with params and store them in the scope.
        builder->create_store(args[i], params[i]);
        scope.push(node.params[i]->id, params[i]);
    }
    node.compound_stmt->accept(*this);
    if (not builder->get_insert_block()->is_terminated()) 
    {
        if (context.func->get_return_type()->is_void_type())
            builder->create_void_ret();
        else if (context.func->get_return_type()->is_float_type())
            builder->create_ret(CONST_FP(0.));
        else
            builder->create_ret(CONST_INT(0));
    }
    scope.exit();
    return nullptr;
}

Value* CminusfBuilder::visit(ASTParam &node) {
    std::string param_name = node.id;
    Type *param_type = (node.type == TYPE_INT) ? INT32_T : FLOAT_T;
    if (node.isarray) {
        param_type = PointerType::get(param_type);
    }
    Value *paramAlloca = builder->create_alloca(param_type);
    //scope.push(param_name, paramAlloca);
    return paramAlloca;
}

Value* CminusfBuilder::visit(ASTCompoundStmt &node) {
    // TODO: This function is not complete.
    // You may need to add some code here
    // to deal with complex statements. 
    scope.enter();
    for (auto &decl : node.local_declarations) {
        decl->accept(*this);
    }

    for (auto &stmt : node.statement_list) {
        stmt->accept(*this);
        if (builder->get_insert_block()->is_terminated())
            break;
    }
    scope.exit();
    return nullptr;
}

Value* CminusfBuilder::visit(ASTExpressionStmt &node) {
    scope.enter();
    if (node.expression) {
        node.expression->accept(*this);
    }
    scope.exit();
    return nullptr;
}

Value* CminusfBuilder::visit(ASTSelectionStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTIterationStmt &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTReturnStmt &node) {
    if (node.expression == nullptr) {
        builder->create_void_ret();
        return nullptr;
    } else {
        // TODO: The given code is incomplete.
        // You need to solve other return cases (e.g. return an integer).
    }
    return nullptr;
}

Value* CminusfBuilder::visit(ASTVar &node) {
    node.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTAssignExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    node.var->accept(*this);

    return nullptr;
}

Value* CminusfBuilder::visit(ASTSimpleExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTAdditiveExpression &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTTerm &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}

Value* CminusfBuilder::visit(ASTCall &node) {
    // TODO: This function is empty now.
    // Add some code here.
    return nullptr;
}
