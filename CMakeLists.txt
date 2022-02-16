cmake_minimum_required(VERSION 2.8)
project(AA277_PROJ)

include(CTest)

add_subdirectory(src)
add_subdirectory(RVO2)

include(CPack)
