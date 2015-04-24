INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()


IF(CUDA_FOUND)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-I${PROJECT_SOURCE_DIR}/src)


CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp image_stack_utils.cpp)

IF(WIN32)
MESSAGE("[Win] compiler is ${CMAKE_CXX_COMPILER_ID}")
MESSAGE("[Win] using Boost libraries ${Boost_LIBRARIES}")
IF("${CMAKE_CXX_COMPILER_ID}" MATCHES MSVC)
target_link_libraries(test_gpu_convolve ${BOOST_LIRBARIES} ${PROJECT_NAME}_static)
set_target_properties(test_gpu_convolve PROPERTIES COMPILE_FLAGS "/I${PROJECT_SOURCE_DIR}/src")
ELSE()
target_link_libraries(test_gpu_convolve ${BOOST_LIRBARIES} ${PROJECT_NAME})
ENDIF()
ELSE()
target_link_libraries(test_gpu_convolve ${BOOST_LIRBARIES} ${PROJECT_NAME})
set_target_properties(test_gpu_convolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/src")
ENDIF()

ELSE(CUDA_FOUND)
MESSAGE(WARNING "Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)


