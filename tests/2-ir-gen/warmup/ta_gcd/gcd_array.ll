; ModuleID = 'gcd_array.c'
source_filename = "gcd_array.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = dso_local global [1 x i32] zeroinitializer, align 4
@y = dso_local global [1 x i32] zeroinitializer, align 4

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @gcd(i32 noundef %u, i32 noundef %v) #0 {
entry:
  %retval = alloca i32, align 4
  %u.addr = alloca i32, align 4
  %v.addr = alloca i32, align 4
  store i32 %u, ptr %u.addr, align 4
  store i32 %v, ptr %v.addr, align 4
  %0 = load i32, ptr %v.addr, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %u.addr, align 4
  store i32 %1, ptr %retval, align 4
  br label %return

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %v.addr, align 4
  %3 = load i32, ptr %u.addr, align 4
  %4 = load i32, ptr %u.addr, align 4
  %5 = load i32, ptr %v.addr, align 4
  %div = sdiv i32 %4, %5
  %6 = load i32, ptr %v.addr, align 4
  %mul = mul nsw i32 %div, %6
  %sub = sub nsw i32 %3, %mul
  %call = call i32 @gcd(i32 noundef %2, i32 noundef %sub)
  store i32 %call, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.else, %if.then
  %7 = load i32, ptr %retval, align 4
  ret i32 %7
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @funArray(ptr noundef %u, ptr noundef %v) #0 {
entry:
  %u.addr = alloca ptr, align 8
  %v.addr = alloca ptr, align 8
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %temp = alloca i32, align 4
  store ptr %u, ptr %u.addr, align 8
  store ptr %v, ptr %v.addr, align 8
  %0 = load ptr, ptr %u.addr, align 8
  %arrayidx = getelementptr inbounds i32, ptr %0, i64 0
  %1 = load i32, ptr %arrayidx, align 4
  store i32 %1, ptr %a, align 4
  %2 = load ptr, ptr %v.addr, align 8
  %arrayidx1 = getelementptr inbounds i32, ptr %2, i64 0
  %3 = load i32, ptr %arrayidx1, align 4
  store i32 %3, ptr %b, align 4
  %4 = load i32, ptr %a, align 4
  %5 = load i32, ptr %b, align 4
  %cmp = icmp slt i32 %4, %5
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %6 = load i32, ptr %a, align 4
  store i32 %6, ptr %temp, align 4
  %7 = load i32, ptr %b, align 4
  store i32 %7, ptr %a, align 4
  %8 = load i32, ptr %temp, align 4
  store i32 %8, ptr %b, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %9 = load i32, ptr %a, align 4
  %10 = load i32, ptr %b, align 4
  %call = call i32 @gcd(i32 noundef %9, i32 noundef %10)
  ret i32 %call
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  store i32 90, ptr @x, align 4
  store i32 18, ptr @y, align 4
  %call = call i32 @funArray(ptr noundef @x, ptr noundef @y)
  ret i32 %call
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git e477989a055f92f6ca63fc8f76929cde81d33e44)"}
