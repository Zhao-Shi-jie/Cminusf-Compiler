#include "BasicBlock.hpp"
#include "Constant.hpp"
#include "Function.hpp"
#include "IRBuilder.hpp"
#include "Module.hpp"
#include "Type.hpp"

#include <iostream>
#include <memory>

// 定义一个从常数值获取/创建 ConstantInt 类实例化的宏，方便多次调用
#define CONST_INT(num) ConstantInt::get(num, module)

int main() {
    // 创建一个 Module 实例
    auto module = new Module();
    // 创建一个 IRBuilder 实例（后续创建指令均使用此实例操作）
    auto builder = new IRBuilder(nullptr, module);

    // 从 Module 处取出 32 位整形 type 的实例
    Type *Int32Type = module->get_int32_type();

    // 创建 main 函数
    auto mainFun = Function::create(FunctionType::get(Int32Type, {}), "main", module);

    // 创建 entry 基本块并设置插入点
    auto entryBB = BasicBlock::create(module, "entry", mainFun);
    builder->set_insert_point(entryBB);

    // 创建变量 a 和 i 并分配空间
    auto aAlloca = builder->create_alloca(Int32Type);
    auto iAlloca = builder->create_alloca(Int32Type);

    // 将 a 初始化为 10
    builder->create_store(CONST_INT(10), aAlloca);
    // 将 i 初始化为 0
    builder->create_store(CONST_INT(0), iAlloca);

    // 创建循环的基本块
    auto loopCondBB = BasicBlock::create(module, "loopCond", mainFun);
    auto loopBodyBB = BasicBlock::create(module, "loopBody", mainFun);
    auto afterLoopBB = BasicBlock::create(module, "afterLoop", mainFun);

    // 在 entryBB 中跳转到条件检查基本块
    builder->create_br(loopCondBB);

    // 设置插入点到循环条件检查基本块
    builder->set_insert_point(loopCondBB);

    // 加载 i 的值并与 10 比较
    auto iLoad = builder->create_load(iAlloca);
    auto icmp = builder->create_icmp_lt(iLoad, CONST_INT(10));

    // 创建条件跳转指令
    builder->create_cond_br(icmp, loopBodyBB, afterLoopBB);

    // 在循环体内创建指令
    builder->set_insert_point(loopBodyBB);

    // i = i + 1;
    auto iIncrement = builder->create_iadd(iLoad, CONST_INT(1));
    builder->create_store(iIncrement, iAlloca);

    // a = a + i;
    auto aLoad = builder->create_load(aAlloca);
    auto aIncrement = builder->create_iadd(aLoad, iIncrement);
    builder->create_store(aIncrement, aAlloca);

    // 跳转回循环条件基本块
    builder->create_br(loopCondBB);

    // 设置插入点到循环后的基本块
    builder->set_insert_point(afterLoopBB);

    // 返回 a 的值
    auto retLoad = builder->create_load(aAlloca);
    builder->create_ret(retLoad);

    // 输出 module 中的所有 IR 指令
    std::cout << module->print();

    delete module;
    return 0;
}