function(add_app TARGET_NAME)

    message("samples project dir " ${MY_BUILDCONFIG_DIR})

    set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
    target_include_directories(${TARGET_NAME} PUBLIC ".")
    target_link_libraries(${TARGET_NAME} ${LIBS} hyperacorn)
    target_include_directories(${TARGET_NAME} PUBLIC "." ${MY_BUILDCONFIG_DIR})

    if (PR_NO_WINDOW)
        target_compile_definitions(${TARGET_NAME} PUBLIC PR_NO_WINDOW)
    endif ()


    message(STATUS "App enabled:       ${TARGET_NAME}")
endfunction()



add_executable(hyper_train hyper_train.cpp hyper_viewer.h)
add_app(hyper_train)
