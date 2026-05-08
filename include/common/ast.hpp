#pragma once

#include <cstdarg>
extern "C" {
#include "syntax_tree.h"
extern syntax_tree *parse(const char *input);
}
#include <memory>
#include <string>
#include <vector>

enum CminusType { TYPE_INT, TYPE_FLOAT, TYPE_VOID };

enum RelOp {
    // <=
    OP_LE,
    // <
    OP_LT,
    // >
    OP_GT,
    // >=
    OP_GE,
    // ==
    OP_EQ,
    // !=
    OP_NEQ
};

enum AddOp {
    // +
    OP_PLUS,
    // -
    OP_MINUS
};

enum MulOp {
    // *
    OP_MUL,
    // /
    OP_DIV
};

enum class ASTKind {
    Program,
    VarDeclaration,
    FunDeclaration,
    Param,
    CompoundStmt,
    ExpressionStmt,
    SelectionStmt,
    IterationStmt,
    ReturnStmt,
    Num,
    Var,
    AssignExpression,
    SimpleExpression,
    AdditiveExpression,
    Term,
    Call
};

class AST;

struct ASTNode;
struct ASTProgram;
struct ASTDeclaration;
struct ASTNum;
struct ASTVarDeclaration;
struct ASTFunDeclaration;
struct ASTParam;

struct ASTCompoundStmt;
struct ASTStatement;
struct ASTExpressionStmt;
struct ASTSelectionStmt;
struct ASTIterationStmt;
struct ASTReturnStmt;
struct ASTFactor;
struct ASTExpression;
struct ASTVar;
struct ASTAssignExpression;
struct ASTSimpleExpression;
struct ASTAdditiveExpression;
struct ASTTerm;
struct ASTCall;

class AST {
  public:
    AST() = delete;
    AST(syntax_tree *);
    AST(AST &&tree) {
        root = tree.root;
        tree.root = nullptr;
    };
    ASTProgram *get_root() { return root.get(); }

  private:
    ASTNode *transform_node_iter(syntax_tree_node *);
    std::shared_ptr<ASTProgram> root = nullptr;
};

struct ASTNode {
    virtual ~ASTNode() = default;
    virtual ASTKind getKind() const = 0;
};

struct ASTProgram : ASTNode {
    virtual ~ASTProgram() = default;
    ASTKind getKind() const override { return ASTKind::Program; }
    std::vector<std::shared_ptr<ASTDeclaration>> declarations;
};

struct ASTDeclaration : ASTNode {
    virtual ~ASTDeclaration() = default;
    CminusType type;
    std::string id;
};

struct ASTFactor : ASTNode {
    virtual ~ASTFactor() = default;
};

struct ASTNum : ASTFactor {
    ASTKind getKind() const override { return ASTKind::Num; }
    CminusType type;
    union {
        int i_val;
        float f_val;
    };
};

struct ASTVarDeclaration : ASTDeclaration {
    ASTKind getKind() const override { return ASTKind::VarDeclaration; }
    std::shared_ptr<ASTNum> num;
};

struct ASTFunDeclaration : ASTDeclaration {
    ASTKind getKind() const override { return ASTKind::FunDeclaration; }
    std::vector<std::shared_ptr<ASTParam>> params;
    std::shared_ptr<ASTCompoundStmt> compound_stmt;
};

struct ASTParam : ASTNode {
    ASTKind getKind() const override { return ASTKind::Param; }
    CminusType type;
    std::string id;
    // true if it is array param
    bool isarray;
};

struct ASTStatement : ASTNode {
    virtual ~ASTStatement() = default;
};

struct ASTCompoundStmt : ASTStatement {
    ASTKind getKind() const override { return ASTKind::CompoundStmt; }
    std::vector<std::shared_ptr<ASTVarDeclaration>> local_declarations;
    std::vector<std::shared_ptr<ASTStatement>> statement_list;
};

struct ASTExpressionStmt : ASTStatement {
    ASTKind getKind() const override { return ASTKind::ExpressionStmt; }
    std::shared_ptr<ASTExpression> expression;
};

struct ASTSelectionStmt : ASTStatement {
    ASTKind getKind() const override { return ASTKind::SelectionStmt; }
    std::shared_ptr<ASTExpression> expression;
    std::shared_ptr<ASTStatement> if_statement;
    // should be nullptr if no else structure exists
    std::shared_ptr<ASTStatement> else_statement;
};

struct ASTIterationStmt : ASTStatement {
    ASTKind getKind() const override { return ASTKind::IterationStmt; }
    std::shared_ptr<ASTExpression> expression;
    std::shared_ptr<ASTStatement> statement;
};

struct ASTReturnStmt : ASTStatement {
    ASTKind getKind() const override { return ASTKind::ReturnStmt; }
    // should be nullptr if return void
    std::shared_ptr<ASTExpression> expression;
};

struct ASTExpression : ASTFactor {};

struct ASTAssignExpression : ASTExpression {
    ASTKind getKind() const override { return ASTKind::AssignExpression; }
    std::shared_ptr<ASTVar> var;
    std::shared_ptr<ASTExpression> expression;
};

struct ASTSimpleExpression : ASTExpression {
    ASTKind getKind() const override { return ASTKind::SimpleExpression; }
    std::shared_ptr<ASTAdditiveExpression> additive_expression_l;
    std::shared_ptr<ASTAdditiveExpression> additive_expression_r;
    RelOp op;
};

struct ASTVar : ASTFactor {
    ASTKind getKind() const override { return ASTKind::Var; }
    std::string id;
    // nullptr if var is of int type
    std::shared_ptr<ASTExpression> expression;
};

struct ASTAdditiveExpression : ASTNode {
    ASTKind getKind() const override { return ASTKind::AdditiveExpression; }
    std::shared_ptr<ASTAdditiveExpression> additive_expression;
    AddOp op;
    std::shared_ptr<ASTTerm> term;
};

struct ASTTerm : ASTNode {
    ASTKind getKind() const override { return ASTKind::Term; }
    std::shared_ptr<ASTTerm> term;
    MulOp op;
    std::shared_ptr<ASTFactor> factor;
};

struct ASTCall : ASTFactor {
    ASTKind getKind() const override { return ASTKind::Call; }
    std::string id;
    std::vector<std::shared_ptr<ASTExpression>> args;
};
