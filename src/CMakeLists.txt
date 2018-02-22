include_directories(${nnet_SOURCE_DIR}/src)
include_directories(${nnet_SOURCE_DIR}/include)
link_directories(${Tutorial_BINARY_DIR}/src)


find_package(Boost 1.58.0 REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})




add_executable(nnet nnet.cpp)
target_link_libraries(nnet ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS})
