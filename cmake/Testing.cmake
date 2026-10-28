# Prefer a system installation; otherwise use the pinned GoogleTest 1.14.0 release.
find_package(GTest 1.14 CONFIG QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()
include(GoogleTest)

add_executable(test_lr_scheduler tests/kernels/test_lr_scheduler.cpp)
target_include_directories(test_lr_scheduler PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_compile_definitions(test_lr_scheduler PRIVATE _USE_MATH_DEFINES)
target_link_libraries(test_lr_scheduler PRIVATE GTest::gtest_main)
gtest_discover_tests(test_lr_scheduler PROPERTIES LABELS cpu)
