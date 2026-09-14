cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

include(FetchContent)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(OpenGL REQUIRED)

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(GL_LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(GL_LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

file(GLOB_RECURSE sources CONFIGURE_DEPENDS
    "src/*.cpp" 
    "src/*.cu"
)
list(FILTER sources EXCLUDE REGEX "src/optixshaders/.*")

set(imgui_headers
    src/ImGui/imconfig.h
    src/ImGui/imgui.h
    src/ImGui/imgui_impl_glfw.h
    src/ImGui/imgui_impl_opengl3.h
    src/ImGui/imgui_impl_opengl3_loader.h
    src/ImGui/imgui_internal.h
    src/ImGui/imstb_rectpack.h
    src/ImGui/imstb_textedit.h
    src/ImGui/imstb_truetype.h
)

set(imgui_sources
    src/ImGui/imgui.cpp
    src/ImGui/imgui_demo.cpp
    src/ImGui/imgui_draw.cpp
    src/ImGui/imgui_impl_glfw.cpp
    src/ImGui/imgui_impl_opengl3.cpp
    src/ImGui/imgui_tables.cpp
    src/ImGui/imgui_widgets.cpp
)


list(SORT imgui_headers)
list(SORT imgui_sources)

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${imgui_sources} ${imgui_headers})

if(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/W2>
    )
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W2>
    )
else()
    # For Linux/macOS (GCC/Clang)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endif()

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281) # 10.2.1
FetchContent_MakeAvailable(fmt)

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
# set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=27>
    )

    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>
    )

    if(MSVC)
        target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        )
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

find_path(OptiX_INCLUDE
    NAMES optix.h
    PATHS "$ENV{OPTIX_SDK}/include"
    NO_DEFAULT_PATH
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE src)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    ${OptiX_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
    ${GL_LIBRARIES}
    fmt::fmt
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::cudart_static
)

include(CheckIPOSupported)
check_ipo_supported(RESULT LUXION_IPO_SUPPORTED OUTPUT LUXION_IPO_MESSAGE LANGUAGES CUDA)
if(NOT LUXION_IPO_SUPPORTED)
    message(STATUS "CUDA device LTO unavailable: ${LUXION_IPO_MESSAGE}")
endif()

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
    INTERPROCEDURAL_OPTIMIZATION ${LUXION_IPO_SUPPORTED}
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})