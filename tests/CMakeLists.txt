INCLUDE_DIRECTORIES(.)

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS system filesystem unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

#FIXME: OS dependent
IF(DEFINED ${BOOST_ROOT} AND ${BOOST_ROOT} MATCHES ".*/.*")
  SET(BOOST_LIB_DIR ${BOOST_ROOT}/lib)
ELSE()
  SET(BOOST_LIB_DIR ${Boost_LIBRARY_DIRS})
ENDIF()

add_library( ext_boost_system  SHARED IMPORTED )
set_target_properties( ext_boost_system PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_system.so )

add_library( ext_boost_filesystem  SHARED IMPORTED )
set_target_properties( ext_boost_filesystem PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_filesystem.so )

add_library( ext_boost_unit_test_framework  SHARED IMPORTED )
set_target_properties( ext_boost_unit_test_framework PROPERTIES IMPORTED_LOCATION ${BOOST_LIB_DIR}/libboost_unit_test_framework.so )


IF(CUDA_FOUND)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-I${PROJECT_SOURCE_DIR}/src)


CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp image_stack_utils.cpp)
target_link_libraries(test_gpu_convolve ext_boost_system ext_boost_filesystem ext_boost_unit_test_framework ${PROJECT_NAME})
set_target_properties(test_gpu_convolve PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/src")

ELSE(CUDA_FOUND)
MESSAGE(WARNING "Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)


