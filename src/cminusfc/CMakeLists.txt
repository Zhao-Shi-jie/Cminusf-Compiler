add_executable(
    cminusfc
    main.cpp
)

target_link_libraries(
    cminusfc
    mlir
    cminusf_passes
    common
    syntax
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    stdc++fs
    ${llvm_libs}
)

target_compile_definitions(cminusfc PRIVATE
    CMINUSF_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)


install(
    TARGETS cminusfc
    RUNTIME DESTINATION bin
)
