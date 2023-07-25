cmake_minimum_required(VERSION 3.17)


project(cuDnn LANGUAGES CXX)
find_package(CUDAToolkit QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cuDnn main.cpp)

install(TARGETS cuDnn
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)
