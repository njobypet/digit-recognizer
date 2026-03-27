cmake_minimum_required(VERSION 3.14)
project(DigitRecognizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_HIP "Enable AMD ROCm/HIP GPU acceleration" OFF)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O2)
endif()

# ---- Source files (always compiled with the C++ compiler) ----
set(SOURCES
    src/main.cpp
    src/neural_network.cpp
    src/image_processor.cpp
    src/digit_recognizer.cpp
    src/mnist_loader.cpp
)

# ---- Try to find ROCm/HIP when requested ----
if(ENABLE_HIP)
    # ROCm typically installs to /opt/rocm on Linux
    if(NOT DEFINED HIP_PATH)
        if(DEFINED ENV{HIP_PATH})
            set(HIP_PATH $ENV{HIP_PATH})
        elseif(EXISTS "/opt/rocm/hip")
            set(HIP_PATH "/opt/rocm/hip")
        elseif(EXISTS "/opt/rocm")
            set(HIP_PATH "/opt/rocm")
        endif()
    endif()

    list(APPEND CMAKE_PREFIX_PATH
        "${HIP_PATH}"
        "/opt/rocm"
        "/opt/rocm/lib/cmake"
        "/opt/rocm/lib/cmake/hip"
    )

    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "HIP found: version ${hip_VERSION}")
        message(STATUS "HIP path: ${HIP_PATH}")
        enable_language(HIP)
        set(USE_HIP ON)
    else()
        message(WARNING "ENABLE_HIP=ON but HIP not found. Building CPU-only version.")
        set(USE_HIP OFF)
    endif()
else()
    set(USE_HIP OFF)
endif()

# ---- Build the executable ----
if(USE_HIP)
    # HIP kernel sources (compiled with hipcc)
    set(HIP_SOURCES
        src/gpu_kernels.hip
    )
    set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

    add_executable(digit_recognizer ${SOURCES} ${HIP_SOURCES})

    target_compile_definitions(digit_recognizer PRIVATE USE_HIP)
    target_link_libraries(digit_recognizer PRIVATE hip::device)

    message(STATUS "==> Building WITH GPU acceleration (AMD ROCm/HIP)")
else()
    add_executable(digit_recognizer ${SOURCES})
    message(STATUS "==> Building WITHOUT GPU acceleration (CPU only)")
endif()

target_include_directories(digit_recognizer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/thirdparty
)

if(UNIX)
    target_link_libraries(digit_recognizer PRIVATE pthread m)
endif()

install(TARGETS digit_recognizer DESTINATION bin)
