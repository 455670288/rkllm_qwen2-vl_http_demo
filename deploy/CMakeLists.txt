cmake_minimum_required(VERSION 3.15.1)

project(demo)


find_package(nlohmann_json CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)




set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

if(${CMAKE_VERSION} VERSION_GREATER "3.15.0" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_link_options("-Wl,-Bsymbolic")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set (TARGET_LIB_ARCH ${CMAKE_ANDROID_ARCH_ABI})
else()
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set (TARGET_LIB_ARCH aarch64)
    else()
        set (TARGET_LIB_ARCH armhf)
    endif()
    if (CMAKE_C_COMPILER MATCHES "uclibc")
        set (TARGET_LIB_ARCH ${TARGET_LIB_ARCH}_uclibc)
    endif()
endif()

# opencv
# if (CMAKE_SYSTEM_NAME MATCHES "Linux")
# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv-linux-aarch64/share/OpenCV)
# elseif(CMAKE_SYSTEM_NAME MATCHES "Android")
  # set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv-mobile-3.4.20-android/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
# endif()

# set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv-linux-aarch64/share/OpenCV)
set(OpenCV_DIR /home/firefly/ljh/opencv4.5-dynamic/lib/cmake/opencv4/)
find_package(OpenCV REQUIRED)

# rknn runtime
set(RKNN_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librknnrt)
set(LIBRKNNRT ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/librknn_api/${TARGET_LIB_ARCH}/librknnrt.so)
set(LIBRKNNRT_INCLUDES ${RKNN_PATH}/${CMAKE_SYSTEM_NAME}/librknn_api/include)

# imgenc
include_directories(src/image_enc.h ${LIBRKNNRT_INCLUDES})
add_executable(imgenc src/image_enc.cc src/img_encoder.cpp)
target_link_libraries(imgenc PRIVATE ${OpenCV_LIBS} ${LIBRKNNRT})

# # rkllm runtime
set(RKLLM_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../rkllm-runtime)
set(LIBRKLLMRT ${RKLLM_PATH}/${CMAKE_SYSTEM_NAME}/librkllm_api/${TARGET_LIB_ARCH}/librkllmrt.so)
set(LIBRKLLMRT_INCLUDES ${RKLLM_PATH}/${CMAKE_SYSTEM_NAME}/librkllm_api/include)

# llm
include_directories(${LIBRKLLMRT_INCLUDES})
add_executable(llm src/llm.cpp)
target_link_libraries(llm PRIVATE ${LIBRKLLMRT})

# llm_test
include_directories(${LIBRKLLMRT_INCLUDES})
add_executable(llm_test src/llm_test.cpp)
target_link_libraries(llm_test PRIVATE ${LIBRKLLMRT})

# demo
add_executable(${PROJECT_NAME} src/image_enc.cc src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${LIBRKNNRT} ${LIBRKLLMRT})

#demo_api
include_directories(${CMAKE_SOURCE_DIR}/http_include)
add_executable(demo_api src/image_enc.cc src/main_openai.cpp)
target_link_libraries(demo_api PRIVATE ${OpenCV_LIBS}  ${LIBRKNNRT} ${LIBRKLLMRT} ${CURL_LIBRARIES} pthread OpenSSL::SSL OpenSSL::Crypto)
set_target_properties(demo_api PROPERTIES LINK_FLAGS  -Wl,--no-undefined,--version-script,${CMAKE_SOURCE_DIR}/c_export.map)


#demo_stream
include_directories(${CMAKE_SOURCE_DIR}/http_include)
add_executable(demo_stream src/image_enc.cc src/main_openai_stream.cpp)
target_link_libraries(demo_stream PRIVATE ${OpenCV_LIBS}  ${LIBRKNNRT} ${LIBRKLLMRT} ${CURL_LIBRARIES} pthread OpenSSL::SSL OpenSSL::Crypto)
set_target_properties(demo_stream PROPERTIES LINK_FLAGS  -Wl,--no-undefined,--version-script,${CMAKE_SOURCE_DIR}/c_export.map)


# Hide symbols to avoid possible symbol conflicts with RKLLM
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS  -Wl,--no-undefined,--version-script,${CMAKE_SOURCE_DIR}/c_export.map)

# Install the executable file to the specified directory
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/demo_${CMAKE_SYSTEM_NAME}_${TARGET_LIB_ARCH})
install(TARGETS imgenc llm llm_test ${PROJECT_NAME} demo_api demo_stream DESTINATION ./)
install(PROGRAMS ${LIBRKNNRT} ${LIBRKLLMRT} DESTINATION lib)
install(PROGRAMS ../data/demo.jpg DESTINATION ./)

