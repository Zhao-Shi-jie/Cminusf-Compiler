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

void pass_node(char *text){
     yylval.node = new_syntax_tree_node(text);
}

/*****************声明和选项设置  end*****************/

%}

%x COMMENT

%%
 /* to do for students */
 /* two cases for you, pass_node will send flex's token to bison */
\n {return EOL;}
[ \t\r]+ {return BLANK;}
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
[a-zA-Z_][a-zA-Z0-9_]* {
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
[0-9]+\.[0-9]+  { pos_start = pos_end; pos_end += yyleng; pass_node(yytext); return FLOATPOINT; }
 /*空白忽略*/
[ \t\n]+ {
    pos_end += yyleng;
}
\/\*\/*([^\/]*[^\/\*]\/[\/]*)*[^\/]*\*\/ {return COMMENT;}
. { pos_start = pos_end; pos_end++; return ERROR; }

 /****请在此补全所有flex的模式与动作  end******/
%%

/* 以下为 C 代码部分 */
