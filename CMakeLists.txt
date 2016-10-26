# cmake compatibility issues
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# project name
PROJECT(FourierConvolutionCUDALib CXX C)
if(NOT(${CMAKE_VERSION} VERSION_LESS "3.0.0"))
cmake_policy(SET CMP0042 NEW)
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.1")
cmake_policy(SET CMP0054 NEW)
endif()

option(ENABLE_TESTING "enable the test suite (requires boost to be installed" ON) 
# version number
SET (FOURIERCONVOLUTIONCUDALIB_NAME "CUDA FOURIER CONVOLUTION LIBRARY")
SET (FOURIERCONVOLUTIONCUDALIB_CODENAME "${PROJECT_NAME}")
SET (FOURIERCONVOLUTIONCUDALIB_COPYRIGHT_YEARS "2013")
SET (FOURIERCONVOLUTIONCUDALIB_VERSION_MAJOR 2)
SET (FOURIERCONVOLUTIONCUDALIB_VERSION_MINOR 5)
SET (FOURIERCONVOLUTIONCUDALIB_VERSION_PATCH 0)
SET (FOURIERCONVOLUTIONCUDALIB_VERSION_TYPE SNAPSHOT)
SET (FOURIERCONVOLUTIONCUDALIB_VERSION_STRING "${FOURIERCONVOLUTIONCUDALIB_VERSION_MAJOR}.${FOURIERCONVOLUTIONCUDALIB_VERSION_MINOR}.${FOURIERCONVOLUTIONCUDALIB_VERSION_PATCH}-${CMAKE_BUILD_TYPE}")
SET (FOURIERCONVOLUTIONCUDALIB_VERSION "${FOURIERCONVOLUTIONCUDALIB_VERSION_MAJOR}.${FOURIERCONVOLUTIONCUDALIB_VERSION_MINOR}.${FOURIERCONVOLUTIONCUDALIB_VERSION_PATCH}")
SET (FOURIERCONVOLUTIONCUDALIB_VENDOR_ID "mpi cbg")
SET (FOURIERCONVOLUTIONCUDALIB_VENDOR_NAME "Max Planck Institute of Molecular Cell Biology and Genetics ")
SET (FOURIERCONVOLUTIONCUDALIB_VENDOR_URL "www.mpi-cbg.de")
SET (FOURIERCONVOLUTIONCUDALIB_ID "${FOURIERCONVOLUTIONCUDALIB_VENDOR_ID}.${PROJECT_NAME}")

# trying to setup paths so this package can be picked up
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)
MESSAGE(">> Setting up ${CMAKE_BUILD_TYPE} build")

# shared path is architecture independent for now, TODO extend this to lib/bin/include
IF(UNIX)
  IF(APPLE)
    set(INSTALL_SHARE_DIR ${PROJECT_NAME}.app/Contents/Resources/ CACHE PATH "Installation directory for shared files")
    #the following was tested with OSX 10.8.5 and Xcode 5.0.2
    #seems to me that under apple the rpath is not stripped automatically when doing the install
    #under linux it is
    SET(CMAKE_SKIP_RPATH ON)
  ELSE(APPLE)
    set(INSTALL_SHARE_DIR lib/CMake/${PROJECT_NAME} CACHE PATH "Installation directory for shared files")
  ENDIF(APPLE)
ELSE(UNIX)
  IF(WIN32 AND NOT CYGWIN)
    set(INSTALL_SHARE_DIR CMake CACHE PATH "Installation directory for shared files")
  ELSE()
    MESSAGE(FATAL_ERROR ">> UNKNOWN ARCHITECTURE .. unable to set share dir")
  ENDIF()
ENDIF(UNIX)
 
# Make relative paths absolute (needed later on)
foreach(p LIB INCLUDE SHARE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

#########################################################################################################
## CUDA related
# project options
OPTION(INCLUDE_CUDA "Set to OFF to not search for CUDA" ON)
# find project dependencies
# find cuda
IF(INCLUDE_CUDA)
    FIND_PACKAGE(CUDA)
    IF(CUDA_FOUND)
	SET(CUDA_VERBOSE_BUILD ON)
	set(CUDA_ARCHS 10;20;30;35;37;50;52;60;61)

	SET(CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
	IF(APPLE)
	  IF(${CUDA_HOST_COMPILER} MATCHES "/usr/bin/.*cc" OR EXISTS "/usr/bin/llvm-g++")
	    MESSAGE(">> adapting CUDA_HOST_COMPILER (${CUDA_HOST_COMPILER}) to match a CUDA supported compiler (/usr/bin/llvm-g++-4.2)")
	    SET(CUDA_HOST_COMPILER "/usr/bin/llvm-g++")
	    SET(CMAKE_CXX_COMPILER ${CUDA_HOST_COMPILER})
	    SET(CMAKE_C_COMPILER "/usr/bin/llvm-gcc")
	    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libstdc++")

	    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
	  ELSE()
	    MESSAGE(WARNING ">> unknown CUDA_HOST_COMPILER (${CUDA_HOST_COMPILER}) or /usr/bin/llvm-g++-4.2 does not exist, cuda host compiler remains set to default")
	  ENDIF()
	ENDIF(APPLE)

	set(CUDA_ARCHS 10;20;21)

	IF("${CUDA_VERSION}" VERSION_GREATER "4.5")
	  list(APPEND CUDA_ARCHS 30 32 35)#5.0+
	ENDIF()

	IF("${CUDA_VERSION}" VERSION_GREATER "5.5")
	  list(APPEND CUDA_ARCHS 50)#6.0+
	ENDIF()

	IF("${CUDA_VERSION}" VERSION_GREATER "6.0")
	  list(APPEND CUDA_ARCHS 37)#6.5+
	ENDIF()
	
	IF("${CUDA_VERSION}" VERSION_GREATER "6.5")
	  list(APPEND CUDA_ARCHS 52)#7.0+
	  list(REMOVE_ITEM CUDA_ARCHS 10 20 21)	  
	ENDIF()

	#IF("${CUDA_VERSION}" VERSION_GREATER "7.0")
	#  list(REMOVE_ITEM CUDA_ARCHS 20 21)#8.0+
	#ENDIF()

	IF("${CUDA_VERSION}" VERSION_GREATER "7.5")
	  list(APPEND CUDA_ARCHS 60 61)#8.0+
	ENDIF()

	list(SORT CUDA_ARCHS)

	list(LENGTH CUDA_ARCHS CUDA_ARCHS_SIZE)
	
	foreach(_SM IN LISTS CUDA_ARCHS)
	  list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${_SM},code=sm_${_SM}")
	endforeach()

	math(EXPR CUDA_ARCHS_LAST_INDEX "${CUDA_ARCHS_SIZE}-1")
	list(GET CUDA_ARCHS ${CUDA_ARCHS_LAST_INDEX} CUDA_HIGHEST_SM)

        list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${CUDA_HIGHEST_SM},code=compute_${CUDA_HIGHEST_SM}")

	MESSAGE(">> CUDA version ${CUDA_VERSION} detected, compiling for Compute Capability/ies ${CUDA_ARCHS} (highest SM: ${CUDA_HIGHEST_SM})")
	
	set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE};-O2;--use_fast_math)
	set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};-g;-G)
    ELSE(CUDA_FOUND)
        MESSAGE(FATAL_ERROR ">> CUDA not found. Exiting ...")
    ENDIF(CUDA_FOUND)
	
	IF(WIN32)
		IF(${CUDA_VERSION} VERSION_GREATER "7" OR ${CUDA_VERSION} VERSION_EQUAL "7")
			IF(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
				message(FATAL_ERROR ">> CUDA version 7 or higher does not support 32bit builds on Windows (found CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})")
			ELSE()
				message(">> [Windows] 64bit build detected, size of void pointer ${CMAKE_SIZEOF_VOID_P}")
			ENDIF()
		
		ENDIF()
		SET(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

	ENDIF(WIN32)
ENDIF(INCLUDE_CUDA)
 

# add subdirectories
ADD_SUBDIRECTORY(src)

FIND_PACKAGE (Boost 1.42 COMPONENTS system filesystem unit_test_framework thread QUIET)
IF(Boost_FOUND AND ENABLE_TESTING)
  unset(Boost_FOUND)
  ADD_SUBDIRECTORY(tests)
  enable_testing()
  include("CTestLists.txt")
ELSE()
  if(NOT Boost_FOUND)
    MESSAGE(">> Boost libraries not found")
  endif()
  MESSAGE(">> skipping test suite")
ENDIF()

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
#              v = binary    v = library

export(PACKAGE ${PROJECT_NAME})

# Create the fourierconvolutioncudalib-config.cmake and fourierconvolutioncudalib-config-version files
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_SHARE_DIR}"
   "${INSTALL_INCLUDE_DIR}")
# ... for the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
configure_file(fourierconvolutioncudalib-config.cmake.in
  "${PROJECT_BINARY_DIR}/fourierconvolutioncudalib-config.cmake" @ONLY)
# ... for the install tree
set(CONF_INCLUDE_DIRS "\${FOURIERCONVOLUTIONCUDALIB}/${REL_INCLUDE_DIR}")
configure_file(fourierconvolutioncudalib-config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/fourierconvolutioncudalib-config.cmake" @ONLY)
# ... for both
configure_file(fourierconvolutioncudalib-config-version.cmake.in
  "${PROJECT_BINARY_DIR}/fourierconvolutioncudalib-config-version.cmake" @ONLY)

# Install the fourierconvolutioncudalib-config.cmake and fourierconvolutioncudalib-config-version.cmake
install(FILES
  "${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/fourierconvolutioncudalib-config.cmake"
  "${PROJECT_BINARY_DIR}/fourierconvolutioncudalib-config-version.cmake"
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)

# Install the export set for use with the install-tree 
install(EXPORT fourierconvolutioncudalib-targets 
  DESTINATION "${INSTALL_SHARE_DIR}" COMPONENT dev)



