add_executable(
    cminusfc
    main.cpp
)

add_dependencies(cminusfc mlir)

target_link_libraries(
    cminusfc
    mlir              # 使用新的 MLIR 库
    common
    syntax
    stdc++fs
)

install(
    TARGETS cminusfc
    RUNTIME DESTINATION bin
)
