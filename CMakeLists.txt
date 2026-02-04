cmake_minimum_required(VERSION 3.14)

project(qt_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

option(ENABLE_DXRT "Enable DXRT support" ON)
option(ENABLE_OPENCV "Enable OpenCV support" ON)

find_package(Qt5 COMPONENTS Core Gui Quick Qml REQUIRED)

# Include directories
include_directories(include)
include_directories(include/utils)
include_directories(include/common)

# Base sources
set(SOURCES
    src/main.cpp
    src/videostreamer.cpp
    src/videostreamitem.cpp
    resources/qml.qrc
)

set(HEADERS
    src/videostreamer.h
    src/videostreamitem.h
)

# OpenCV Configuration
if(ENABLE_OPENCV)
    find_package(OpenCV REQUIRED)
    add_definitions(-DUSE_OPENCV)
else()
    message(STATUS "OpenCV support disabled")
endif()

# DXRT Configuration
if(ENABLE_DXRT)
    if(NOT DEFINED DXRT_DIR)
        if(DEFINED ENV{DXRT_DIR})
           set(DXRT_DIR $ENV{DXRT_DIR})
        endif()
    endif()

    if(DEFINED DXRT_DIR)
        include_directories(${DXRT_DIR}/include)
        link_directories(${DXRT_DIR}/lib)
        add_definitions(-DUSE_DXRT)
    else()
        message(WARNING "DXRT enabled but DXRT_DIR not found. Disabling DXRT.")
        set(ENABLE_DXRT OFF)
    endif()
else()
    message(STATUS "DXRT support disabled")
endif()

# Specialized Source Files
if(ENABLE_DXRT)
    list(APPEND SOURCES
        src/bbox.cpp
        src/nms.cpp
        src/yolo.cpp
        src/yolo_cfg.cpp
    )
endif()

if(ENABLE_OPENCV AND ENABLE_DXRT)
    list(APPEND SOURCES
        src/display.cpp
        src/image.cpp
    )
endif()

# Note: if OpenCV is ON but DXRT is OFF, we don't need image.cpp (PreProc) 
# because it's only used for model input preparation.
# We also don't need display.cpp (DisplayBoundingBox).

add_executable(qt_example ${SOURCES} ${HEADERS})

target_link_libraries(qt_example
    Qt5::Core
    Qt5::Gui
    Qt5::Quick
    Qt5::Qml
)

if(ENABLE_OPENCV)
    target_link_libraries(qt_example ${OpenCV_LIBS})
endif()

if(ENABLE_DXRT)
    target_link_libraries(qt_example dxrt)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(qt_example pthread)
endif()
