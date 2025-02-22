#include "ast.hpp"
#include "syntax_tree.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <stack>
#include <string>

#define _AST_NODE_ERROR_                                                                           \
    std::cerr << "Abort due to node cast error."                                                   \
                 "Contact with TAs to solve your problem."                                         \
              << std::endl;                                                                        \
    std::abort();
// NOTE: use std::string instead of strcmp to compare strings,
// 注意使用这里的接口
#define _STR_EQ(a, b) (strcmp((a), (b)) == 0)

AST::AST(syntax_tree *s) {
    if (s == nullptr) {
        std::cerr << "empty input tree!" << std::endl;
        std::abort();
    }
    auto node = transform_node_iter(s->root);
    del_syntax_tree(s);
    root = std::shared_ptr<ASTProgram>(static_cast<ASTProgram *>(node));
}

ASTNode *AST::transform_node_iter(syntax_tree_node *n) {
    /*
    example:
    program -> declaration-list
    declaration-list -> declaration-list declaration | declaration
    将其转化成AST中ASTProgram的结构
    */
    if (_STR_EQ(n->name, "program")) {
        auto node = new ASTProgram();
        // flatten declaration list
        std::stack<syntax_tree_node *> s; // 为什么这里要用stack呢？如果用其他数据结构应该如何实现
        auto list_ptr = n->children[0];
        while (list_ptr->children_num == 2) {
            s.push(list_ptr->children[1]);
            list_ptr = list_ptr->children[0];
        }
        s.push(list_ptr->children[0]);

        while (!s.empty()) {
            auto child_node = static_cast<ASTDeclaration *>(transform_node_iter(s.top()));

            auto child_node_shared = std::shared_ptr<ASTDeclaration>(child_node);
            node->declarations.push_back(child_node_shared);
            s.pop();
        }
        return node;
    } else if (_STR_EQ(n->name, "declaration")) {
        return transform_node_iter(n->children[0]);
    } else if (_STR_EQ(n->name, "var-declaration")) {
        auto node = new ASTVarDeclaration();
        // NOTE: 思考 ASTVarDeclaration的结构，需要填充的字段有哪些
        // type
        // 为什么不会有 TYPE_VOID?
        if (_STR_EQ(n->children[0]->children[0]->name, "int"))
            node->type = TYPE_INT;
        else
            node->type = TYPE_FLOAT;
        // id & num
        // 由不同的表达式填充
        if (n->children_num == 3) {
            node->id = n->children[1]->name;
        } else if (n->children_num == 6) {
            node->id = n->children[1]->name;
            int num = std::stoi(n->children[3]->name);
            auto num_node = std::make_shared<ASTNum>();
            num_node->i_val = num;
            num_node->type = TYPE_INT;
            node->num = num_node;
        } else {
            std::cerr << "[ast]: var-declaration transform failure!" << std::endl;
            std::abort();
        }
        return node;
    } else if (_STR_EQ(n->name, "fun-declaration")) {
        // fun-declaration -> type-specifier ID ( params ) compound-stmt
        // 由表达式和 ASTFunDeclaration的结构，我们需要填充
        // type, id, params, compound_stmt 这四个字段
        auto node = new ASTFunDeclaration();
        // type 字段填充
        if (_STR_EQ(n->children[0]->children[0]->name, "int")) {
            node->type = TYPE_INT;
        } else if (_STR_EQ(n->children[0]->children[0]->name, "float")) {
            node->type = TYPE_FLOAT;
        } else {
            node->type = TYPE_VOID;
        }
        // id 字段填充
        node->id = n->children[1]->name;

        /*
            params 字段填充
            注意这里的params是一个列表，每个元素都是一个ASTParam，需要flatten
            params -> param-list | void
            param-list -> param-list , param | param
        */
        // TODO: 1.fill in the fields of ASTFunDeclaration
        // 1.1 flatten params
        auto params_ptr = n->children[3]->children[0];
        if (_STR_EQ(params_ptr->name, "void")) {
            node->params.clear();
        } else if (_STR_EQ(params_ptr->name, "param-list")) {
            std::stack<syntax_tree_node *> s;
            while (params_ptr->children_num == 3) {
                // std::cout<<"param: "<<params_ptr->children[2]->name<<std::endl;
                s.push(params_ptr->children[2]);
                params_ptr = params_ptr->children[0];
            }
            // std::cout<<"param: "<<params_ptr->children[0]->name<<std::endl;
            s.push(params_ptr->children[0]);

            while (!s.empty()) {
                // std::cout<<"param-out: "<<s.top()->name<<std::endl;
                auto child_node = static_cast<ASTParam *>(transform_node_iter(s.top()));
                auto child_node_shared = std::shared_ptr<ASTParam>(child_node);
                node->params.push_back(child_node_shared);
                s.pop();
            }
        } else {
            std::cerr << "[ast]: params transform failure!" << std::endl;
            std::abort();
        }

        // 1.2 compound_stmt 字段填充
        auto compound_stmt_ptr = n->children[5];
        auto child_node = static_cast<ASTCompoundStmt *>(transform_node_iter(compound_stmt_ptr));
        auto child_node_shared = std::shared_ptr<ASTCompoundStmt>(child_node);
        node->compound_stmt = child_node_shared;

        return node;
    } else if (_STR_EQ(n->name, "param")) {
        // param -> type-specifier ID | type-specifier ID [ ]
        // ASTParam的结构 主要需要填充的属性有 type, id, isarray
        auto node = new ASTParam();
        // type, id, isarray
        if (_STR_EQ(n->children[0]->children[0]->name, "int"))
            node->type = TYPE_INT;
        else
            node->type = TYPE_FLOAT;
        node->id = n->children[1]->name;
        if (n->children_num > 2)
            node->isarray = true;
        return node;
    } else if (_STR_EQ(n->name, "compound-stmt")) {
        auto node = new ASTCompoundStmt();
        // TODO: 2.fill in the fields of ASTCompoundStmt
        /*
          文法表达式如下
          compound-stmt -> { local-declarations statement-list }
          local-declarations -> local-declarations var-declaration | empty
          statement-list -> statement-list statement | empty
        */
        // local declarations
        // 2.1 flatten local declarations
        auto local_declaration_ptr = n->children[1];
        if (local_declaration_ptr->children_num == 2) {
            std::stack<syntax_tree_node *> s;
            while (local_declaration_ptr->children_num == 2) {
                // std::cout<<": "<<params_ptr->children[1]->name;
                s.push(local_declaration_ptr->children[1]);
                local_declaration_ptr = local_declaration_ptr->children[0];
            }

            while (!s.empty()) {
                auto child_node = static_cast<ASTVarDeclaration *>(transform_node_iter(s.top()));
                auto child_node_shared = std::shared_ptr<ASTVarDeclaration>(child_node);
                node->local_declarations.push_back(child_node_shared);
                s.pop();
            }
        }

        // statement list
        // 2.2 flatten statement-list
        auto statement_list_ptr = n->children[2];
        if (statement_list_ptr->children_num == 2) {
            std::stack<syntax_tree_node *> s;
            while (statement_list_ptr->children_num == 2) {
                s.push(statement_list_ptr->children[1]);
                statement_list_ptr = statement_list_ptr->children[0];
            }

            while (!s.empty()) {
                auto child_node = static_cast<ASTStatement *>(transform_node_iter(s.top()));
                auto child_node_shared = std::shared_ptr<ASTStatement>(child_node);
                node->statement_list.push_back(child_node_shared);
                s.pop();
            }
        }

        return node;
    } else if (_STR_EQ(n->name, "statement")) {
        return transform_node_iter(n->children[0]);
    } else if (_STR_EQ(n->name, "expression-stmt")) {
        auto node = new ASTExpressionStmt();
        if (n->children_num == 2) {
            auto expr_node = static_cast<ASTExpression *>(transform_node_iter(n->children[0]));

            auto expr_node_ptr = std::shared_ptr<ASTExpression>(expr_node);
            node->expression = expr_node_ptr;
        }
        return node;
    } else if (_STR_EQ(n->name, "selection-stmt")) {
        auto node = new ASTSelectionStmt();
        // TODO: 5.fill in the fields of ASTSelectionStmt
        /*
          selection-stmt -> if ( expression ) statement | if ( expression )
          statement else statement ASTSelectionStmt的结构，需要填充的字段有
          expression, if_statement, else_statement
        */
        // 5.1 expresstion
        auto expression_node = static_cast<ASTExpression *>(transform_node_iter(n->children[2]));
        auto expression_node_ptr = std::shared_ptr<ASTExpression>(expression_node);
        node->expression = expression_node_ptr;
        // 5.2 if statement
        auto ifstatement_node = static_cast<ASTStatement *>(transform_node_iter(n->children[4]));
        auto ifstatement_node_ptr = std::shared_ptr<ASTStatement>(ifstatement_node);
        node->if_statement = ifstatement_node_ptr;
        // check whether this selection statement contains
        // 5.3 else structure
        if (n->children_num > 5) {
            auto elsestatement_node =
                static_cast<ASTStatement *>(transform_node_iter(n->children[6]));
            auto elsestatement_node_ptr = std::shared_ptr<ASTStatement>(elsestatement_node);
            node->else_statement = elsestatement_node_ptr;
        }
        return node;
    } else if (_STR_EQ(n->name, "iteration-stmt")) {
        auto node = new ASTIterationStmt();

        auto expr_node = static_cast<ASTExpression *>(transform_node_iter(n->children[2]));
        auto expr_node_ptr = std::shared_ptr<ASTExpression>(expr_node);
        node->expression = expr_node_ptr;

        auto stmt_node = static_cast<ASTStatement *>(transform_node_iter(n->children[4]));
        auto stmt_node_ptr = std::shared_ptr<ASTStatement>(stmt_node);
        node->statement = stmt_node_ptr;

        return node;
    } else if (_STR_EQ(n->name, "return-stmt")) {
        auto node = new ASTReturnStmt();
        if (n->children_num == 3) {
            auto expr_node = static_cast<ASTExpression *>(transform_node_iter(n->children[1]));
            node->expression = std::shared_ptr<ASTExpression>(expr_node);
        }
        return node;
    } else if (_STR_EQ(n->name, "expression")) {
        // simple-expression
        if (n->children_num == 1) {
            return transform_node_iter(n->children[0]);
        }
        auto node = new ASTAssignExpression();

        auto var_node = static_cast<ASTVar *>(transform_node_iter(n->children[0]));
        node->var = std::shared_ptr<ASTVar>(var_node);

        auto expr_node = static_cast<ASTExpression *>(transform_node_iter(n->children[2]));
        node->expression = std::shared_ptr<ASTExpression>(expr_node);

        return node;
    } else if (_STR_EQ(n->name, "var")) {
        auto node = new ASTVar();
        node->id = n->children[0]->name;
        if (n->children_num == 4) {
            auto expr_node = static_cast<ASTExpression *>(transform_node_iter(n->children[2]));
            node->expression = std::shared_ptr<ASTExpression>(expr_node);
        }
        return node;
    } else if (_STR_EQ(n->name, "simple-expression")) {
        auto node = new ASTSimpleExpression();
        auto expr_node_1 =
            static_cast<ASTAdditiveExpression *>(transform_node_iter(n->children[0]));
        node->additive_expression_l = std::shared_ptr<ASTAdditiveExpression>(expr_node_1);

        if (n->children_num == 3) {
            auto op_name = n->children[1]->children[0]->name;
            if (_STR_EQ(op_name, "<="))
                node->op = OP_LE;
            else if (_STR_EQ(op_name, "<"))
                node->op = OP_LT;
            else if (_STR_EQ(op_name, ">"))
                node->op = OP_GT;
            else if (_STR_EQ(op_name, ">="))
                node->op = OP_GE;
            else if (_STR_EQ(op_name, "=="))
                node->op = OP_EQ;
            else if (_STR_EQ(op_name, "!="))
                node->op = OP_NEQ;

            auto expr_node_2 =
                static_cast<ASTAdditiveExpression *>(transform_node_iter(n->children[2]));
            node->additive_expression_r = std::shared_ptr<ASTAdditiveExpression>(expr_node_2);
        }
        return node;
    } else if (_STR_EQ(n->name, "additive-expression")) {
        auto node = new ASTAdditiveExpression();
        if (n->children_num == 3) {
            // TODO: 4.fill in the fields of ASTAdditiveExpression
            /*
              文法表达式如下
              additive-expression -> additive-expression addop term | term
            */
            // additive_expression, term, op
            auto additive_expression_node =
                static_cast<ASTAdditiveExpression *>(transform_node_iter(n->children[0]));
            node->additive_expression =
                std::shared_ptr<ASTAdditiveExpression>(additive_expression_node);
            auto addop_name = n->children[1]->children[0]->name;
            if (_STR_EQ(addop_name, "+")) {
                node->op = OP_PLUS;
            } else if (_STR_EQ(addop_name, "-")) {
                node->op = OP_MINUS;
            }
            auto term_node = static_cast<ASTTerm *>(transform_node_iter(n->children[2]));
            node->term = std::shared_ptr<ASTTerm>(term_node);
        } else {
            auto term_node = static_cast<ASTTerm *>(transform_node_iter(n->children[0]));
            node->term = std::shared_ptr<ASTTerm>(term_node);
        }
        return node;
    } else if (_STR_EQ(n->name, "term")) {
        auto node = new ASTTerm();
        if (n->children_num == 3) {
            auto term_node = static_cast<ASTTerm *>(transform_node_iter(n->children[0]));
            node->term = std::shared_ptr<ASTTerm>(term_node);

            auto op_name = n->children[1]->children[0]->name;
            if (_STR_EQ(op_name, "*"))
                node->op = OP_MUL;
            else if (_STR_EQ(op_name, "/"))
                node->op = OP_DIV;

            auto factor_node = static_cast<ASTFactor *>(transform_node_iter(n->children[2]));
            node->factor = std::shared_ptr<ASTFactor>(factor_node);
        } else {
            auto factor_node = static_cast<ASTFactor *>(transform_node_iter(n->children[0]));
            node->factor = std::shared_ptr<ASTFactor>(factor_node);
        }
        return node;
    } else if (_STR_EQ(n->name, "factor")) {
        int i = 0;
        if (n->children_num == 3)
            i = 1;
        auto name = n->children[i]->name;
        if (_STR_EQ(name, "expression") || _STR_EQ(name, "var") || _STR_EQ(name, "call"))
            return transform_node_iter(n->children[i]);
        else {
            auto num_node = new ASTNum();
            // TODO: 3.fill in the fields of ASTNum
            /*
              文法表达式如下
              factor -> ( expression ) | var | call | integer | float
              float -> FLOATPOINT
              integer -> INTEGER
            */
            if (_STR_EQ(name, "integer")) {
                // 3.1
                num_node->type = TYPE_INT;
                num_node->i_val = std::stoi(n->children[i]->children[0]->name);
            } else if (_STR_EQ(name, "float")) {
                // 3.2
                num_node->type = TYPE_FLOAT;
                num_node->f_val = std::stof(n->children[i]->children[0]->name);
            } else {
                _AST_NODE_ERROR_
            }
            return num_node;
        }
    } else if (_STR_EQ(n->name, "call")) {
        auto node = new ASTCall();
        node->id = n->children[0]->name;
        // flatten args
        if (_STR_EQ(n->children[2]->children[0]->name, "arg-list")) {
            auto list_ptr = n->children[2]->children[0];
            auto s = std::stack<syntax_tree_node *>();
            while (list_ptr->children_num == 3) {
                s.push(list_ptr->children[2]);
                list_ptr = list_ptr->children[0];
            }
            s.push(list_ptr->children[0]);

            while (!s.empty()) {
                auto expr_node = static_cast<ASTExpression *>(transform_node_iter(s.top()));
                auto expr_node_ptr = std::shared_ptr<ASTExpression>(expr_node);
                node->args.push_back(expr_node_ptr);
                s.pop();
            }
        }
        return node;
    } else {
        std::cerr << "[ast]: transform failure!" << std::endl;
        std::abort();
    }
}

#define _DEBUG_PRINT_N_(N)                                                                         \
    {                                                                                              \
        std::cout << std::string(N, '-');                                                          \
    }
