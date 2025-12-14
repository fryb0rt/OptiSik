cmake_minimum_required(VERSION 3.10.0)
project(OptiSik VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(TestOptiSik tests/vector_gtest.cpp tests/matrix_gtest.cpp tests/simplexSolver_gtest.cpp tests/lumatrix_gtest.cpp)

# Make project root available so we can include headers without relative paths
target_include_directories(TestOptiSik PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(
  TestOptiSik
    PRIVATE
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(TestOptiSik)