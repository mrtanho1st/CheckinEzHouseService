
cmake_minimum_required(VERSION 3.16)

project(yunet_service LANGUAGES CXX)

# ==================================================
# 1. C++ Configuration
# ==================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release optimization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -O3
        -pipe
    )
endif()

# ==================================================
# 2. OpenCV 4.10+
# ==================================================
find_package(OpenCV 4.10 REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    dnn
    objdetect
)

message(STATUS "=====================================")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV Includes: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "=====================================")

# ==================================================
# 3. Dependencies
# ==================================================
include(FetchContent)

# cpp-httplib
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
    GIT_SHALLOW TRUE
)

# nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(
    httplib
    nlohmann_json
)

find_package(Threads REQUIRED)

# ==================================================
# 4. Build YuNet Service
# ==================================================
add_executable(yunet_service
    src/main.cpp
)

target_include_directories(yunet_service PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yunet_service PRIVATE
    ${OpenCV_LIBS}
    httplib::httplib
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# ==================================================
# 5. Runtime library lookup
# ==================================================
set_target_properties(yunet_service PROPERTIES
    BUILD_RPATH "/opt/opencv/lib"
    INSTALL_RPATH "/opt/opencv/lib"
)

message(STATUS "YuNet C++ Service configured successfully")
