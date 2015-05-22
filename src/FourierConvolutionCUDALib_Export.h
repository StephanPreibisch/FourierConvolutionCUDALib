
#ifndef FourierConvolutionCUDALib_EXPORT_H
#define FourierConvolutionCUDALib_EXPORT_H

#ifdef FourierConvolutionCUDALib_BUILT_AS_STATIC
#  define FourierConvolutionCUDALib_EXPORT
#  define FOURIERCONVOLUTIONCUDALIB_NO_EXPORT
#else
#  ifndef FourierConvolutionCUDALib_EXPORT
#    ifdef FourierConvolutionCUDALib_EXPORTS
        /* We are building this library */
#      define FourierConvolutionCUDALib_EXPORT __declspec(dllexport)
#    else
        /* We are using this library */
#      define FourierConvolutionCUDALib_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef FOURIERCONVOLUTIONCUDALIB_NO_EXPORT
#    define FOURIERCONVOLUTIONCUDALIB_NO_EXPORT 
#  endif
#endif

#ifndef FOURIERCONVOLUTIONCUDALIB_DEPRECATED
#  define FOURIERCONVOLUTIONCUDALIB_DEPRECATED __declspec(deprecated)
#endif

#ifndef FOURIERCONVOLUTIONCUDALIB_DEPRECATED_EXPORT
#  define FOURIERCONVOLUTIONCUDALIB_DEPRECATED_EXPORT FourierConvolutionCUDALib_EXPORT FOURIERCONVOLUTIONCUDALIB_DEPRECATED
#endif

#ifndef FOURIERCONVOLUTIONCUDALIB_DEPRECATED_NO_EXPORT
#  define FOURIERCONVOLUTIONCUDALIB_DEPRECATED_NO_EXPORT FOURIERCONVOLUTIONCUDALIB_NO_EXPORT FOURIERCONVOLUTIONCUDALIB_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define FOURIERCONVOLUTIONCUDALIB_NO_DEPRECATED
#endif

#endif
