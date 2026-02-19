find_library(HAILORT_LIBRARY hailort)
find_path(HAILORT_INCLUDE_DIR hailo/hailort.h)

if (NOT HAILORT_LIBRARY OR NOT HAILORT_INCLUDE_DIR)
    message(FATAL_ERROR "HailoRT not found. Please install the HailoRT SDK.")
endif()

message(STATUS "HailoRT library: ${HAILORT_LIBRARY}")
message(STATUS "HailoRT include: ${HAILORT_INCLUDE_DIR}")

find_package(Threads REQUIRED)

# hailo-common static library

set(TARGET hailo-common)

add_library(${TARGET} STATIC
    hailo-common.cpp
    hailo-common.h
    hailo-engine.cpp
    hailo-engine.h
)

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HAILORT_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann
)

target_link_libraries(${TARGET} PUBLIC
    ${HAILORT_LIBRARY}
    Threads::Threads
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# llama-hailo-server executable

set(TARGET llama-hailo-server)

add_executable(${TARGET} hailo-server.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor
)

target_link_libraries(${TARGET} PRIVATE
    hailo-common
    cpp-httplib
    Threads::Threads
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# llama-hailo-cli executable

set(TARGET llama-hailo-cli)

add_executable(${TARGET} hailo-cli.cpp)
install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE
    hailo-common
    Threads::Threads
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)
