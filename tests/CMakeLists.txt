INCLUDE_DIRECTORIES(.)

FIND_PACKAGE (Boost 1.42 QUIET COMPONENTS # system filesystem
  unit_test_framework REQUIRED)
IF(Boost_FOUND)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS}) 
ENDIF()

FIND_PACKAGE(CUDA)

IF(CUDA_FOUND)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src)
LINK_DIRECTORIES(${PROJECT_BINARY_DIR}/src) 

CUDA_ADD_EXECUTABLE(test_gpu_convolve test_gpu_convolve.cpp image_stack_utils.cpp)
CUDA_ADD_EXECUTABLE(test_gpu_numerical_stability test_gpu_numerical_stability.cpp image_stack_utils.cpp)
CUDA_ADD_EXECUTABLE(test_how_cufft_works test_how_cufft_works.cu image_stack_utils.cpp)

IF(Boost_FOUND)

  MESSAGE(">> Boost UTF: ${Boost_LIBRARIES} ")
  target_link_libraries(test_gpu_convolve ${Boost_LIBRARIES} ${PROJECT_NAME})
  set_target_properties(test_gpu_convolve PROPERTIES COMPILE_FLAGS "-DBOOST_TEST_DYN_LINK")

  target_link_libraries(test_gpu_numerical_stability ${Boost_LIBRARIES} ${PROJECT_NAME})
  set_target_properties(test_gpu_numerical_stability PROPERTIES COMPILE_FLAGS "-DBOOST_TEST_DYN_LINK")

  target_link_libraries(test_how_cufft_works ${Boost_LIBRARIES} )
  CUDA_ADD_CUFFT_TO_TARGET( test_how_cufft_works )


ENDIF(Boost_FOUND)

ELSE(CUDA_FOUND)
MESSAGE(WARNING "Skipping GPU based tests, CUDA not found\!")
ENDIF(CUDA_FOUND)


