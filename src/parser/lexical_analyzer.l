%option noyywrap
%{
/*****************声明和选项设置  begin*****************/
#include <stdio.h>
#include <stdlib.h>

#include "syntax_tree.h"
#include "syntax_analyzer.h"

int lines=1;
int pos_start=1;
int pos_end=1;

int comment_level = 0; // 嵌套层级计数

void pass_node(char *text){
     yylval.node = new_syntax_tree_node(text);
}

/*****************声明和选项设置  end*****************/

%}

%x COMMENT

%%
 /* to do for students */
 /* two cases for you, pass_node will send flex's token to bison */
\n+ {
    int count = yyleng;     // 连续换行符的数量
    lines += count;         // 更新行号
    pos_start = 1;          // 起始列重置为 1
    pos_end = 1;            // 结束列重置为 1
}

[ \t]+ {
    pos_start = pos_end;
    pos_end += yyleng;
}
\+ 	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return ADD;}
\-	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return SUB;}
\*	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return MUL;}
\/	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return DIV;}
\<	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return LT;}
\<=	{pos_start = pos_end; pos_end += 2; pass_node(yytext); return LTE;}
\>	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return GT;}
\>=	{pos_start = pos_end; pos_end += 2; pass_node(yytext); return GTE;}
==	{pos_start = pos_end; pos_end += 2; pass_node(yytext); return EQ;}
!=	{pos_start = pos_end; pos_end += 2; pass_node(yytext); return NEQ;}
=	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return ASSIGN;}
;	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return SEMICOLON;}
,	{pos_start = pos_end; pos_end += 1; pass_node(yytext); return COMMA;}
 /* TODO: phase1. 请在这里补充其他的词法规则 */
 /*关键字*/
"int"    { pos_start = pos_end; pos_end += yyleng; pass_node(yytext); return INT; }
"void"   { pos_start = pos_end; pos_end += yyleng; pass_node(yytext); return VOID; }
"float"  { pos_start = pos_end; pos_end += yyleng; pass_node(yytext); return FLOAT; } 
"if" {
    pos_start = pos_end;
    pos_end += 2;
    pass_node(yytext);
    return IF;
}
"else" {
    pos_start = pos_end;
    pos_end += 4;
    pass_node(yytext);
    return ELSE;
}
"while" {
    pos_start = pos_end;
    pos_end += 5;
    pass_node(yytext);
    return WHILE;
}
"return" {
    pos_start = pos_end;
    pos_end += 6;
    pass_node(yytext);
    return RETURN;
}
 /*标识符*/
[a-zA-Z_](?:[a-zA-Z]*(?:_[a-zA-Z0-9]+)?)* {
    pos_start = pos_end;
    pos_end += yyleng;
    pass_node(yytext);
    return IDENTIFIER;
}
 /*括号*/
\( {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LPARENTHESE;}
\) {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RPARENTHESE;}
\[ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACKET;}
\] {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACKET;}
\{ {pos_start = pos_end; pos_end += 1; pass_node(yytext); return LBRACE;}
\} {pos_start = pos_end; pos_end += 1; pass_node(yytext); return RBRACE;}
 /*字符串*/
(["'])([^\\\1]|\\.)*\1 {
    pos_start = pos_end;
    pos_end += yyleng;
    pass_node(yytext);
    return STRING;
}
 /*整数和浮点数*/
[0-9]+          { pos_start = pos_end; pos_end += yyleng; pass_node(yytext); return INTEGER; }
[0-9]*\.[0-9]+|[0-9]+\.[0-9]*  { pos_start = pos_end; pos_end += yyleng; pass_node(yytext); return FLOATPOINT; }

"/*" {
    if (comment_level == 0) BEGIN(COMMENT); // 进入注释状态
    comment_level++;  // 增加嵌套层级
}

<COMMENT>"/*" {
    comment_level++;  // 遇到嵌套注释，增加层级
}

<COMMENT>"*/" {
    comment_level--;  // 关闭一层嵌套
    if (comment_level == 0) BEGIN(INITIAL); // 全部嵌套结束，返回初始状态
}

<COMMENT>.|\n {
    // 处理注释中的其他内容，更新行列号
    if (yytext[0] == '\n') {
        lines++;
        pos_end = 1;
    } else {
        pos_end++;
    }
}

<COMMENT><<EOF>> {
    // 文件结束但注释未闭合，报错
    fprintf(stderr, "Error: Unterminated comment\n");
    exit(1);
}

. { pos_start = pos_end; pos_end++; return ERROR; }

 /****请在此补全所有flex的模式与动作  end******/
%%

/* 以下为 C 代码部分 */
