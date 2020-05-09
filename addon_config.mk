=======
# All variables and this file are optional, if they are not present the PG and the
# makefiles will try to parse the correct values from the file system.
#
# Variables that specify exclusions can use % as a wildcard to specify that anything in
# that position will match. A partial path can also be specified to, for example, exclude
# a whole folder from the parsed paths from the file system
#
# Variables can be specified using = or +=
# = will clear the contents of that variable both specified from the file or the ones parsed
# from the file system
# += will add the values to the previous ones in the file or the ones parsed from the file 
# system
# 
# The PG can be used to detect errors in this file, just create a new project with this addon 
# and the PG will write to the console the kind of error and in which line it is

meta:
    ADDON_NAME = ofxLibTorch
    ADDON_DESCRIPTION = Addon for LibTorch
    ADDON_AUTHOR = @rystylee
    ADDON_TAGS = "machine learning" "deep learning"
    ADDON_URL = https://github.com/rystylee/ofxLibTorch

common:
    # dependencies with other addons, a list of them separated by spaces 
    # or use += in several lines
    ADDON_DEPENDENCIES = ofxOpenCv
    
    # include search paths, this will be usually parsed from the file system
    # but if the addon or addon libraries need special search paths they can be
    # specified here separated by spaces or one per line using +=
    ADDON_INCLUDES =
    ADDON_INCLUDES += "src"
    
    # any special flag that should be passed to the compiler when using this
    # addon
    # ADDON_CFLAGS =
    
    # any special flag that should be passed to the compiler for c++ files when 
    # using this addon
    ADDON_CPPFLAGS = "-std=c++14"
    
    # any special flag that should be passed to the linker when using this
    # addon, also used for system libraries with -lname
    # ADDON_LDFLAGS =
    
    # source files, these will be usually parsed from the file system looking
    # in the src folders in libs and the root of the addon. if your addon needs
    # to include files in different places or a different set of files per platform
    # they can be specified here
    # ADDON_SOURCES =
    
    # source files that will be included as C files explicitly
    # ADDON_C_SOURCES = 
    
    # source files that will be included as header files explicitly
    # ADDON_HEADER_SOURCES = 
    
    # source files that will be included as c++ files explicitly
    # ADDON_CPP_SOURCES = 
    
    # source files that will be included as objective c files explicitly
    # ADDON_OBJC_SOURCES = 
    
    # derines that will be passed to the compiler when including this addon
    ADDON_DEFINES = "AT_PARALLEL_OPENMP=1"
    
    # some addons need resources to be copied to the bin/data folder of the project
    # specify here any files that need to be copied, you can use wildcards like * and ?
    # ADDON_DATA = 
    
    # when parsing the file system looking for libraries exclude this for all or
    # a specific platform
    # ADDON_LIBS_EXCLUDE =
    
    # binary libraries, these will be usually parsed from the file system but some 
    # libraries need to passed to the linker in a specific order/
    # 
    ADDON_LIBS =
    
    
linux64:
    # linux only, any library that should be included in the project using
    # pkg-config
    # ADDON_PKG_CONFIG_LIBRARIES =
vs:
    # After compiling copy the following dynamic libraries to the executable directory
    # only windows visual studio
    # ADDON_DLLS_TO_COPY = 

    # Release
    ADDON_INCLUDES += "libs/win/include/vs/x64/Release"
    ADDON_INCLUDES += "libs/win/include/vs/x64/Release/torch/csrc/api/include"
    # Debug
    # ADDON_INCLUDES += "libs/win/include/vs/x64/Debug"
    # ADDON_INCLUDES += "libs/win/include/vs/x64/Debug/torch/csrc/api/include"

    ADDON_INCLUDES += "C:/Program Files/NVIDIA Corporation/NvToolsExt/include"
    ADDON_INCLUDES += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/include"

    # Release
    ADDON_LIBS += "libs/win/lib/vs/x64/Release/c10.lib"
    ADDON_LIBS += "libs/win/lib/vs/x64/Release/caffe2_nvrtc.lib"
    ADDON_LIBS += "libs/win/lib/vs/x64/Release/c10_cuda.lib"
    ADDON_LIBS += "libs/win/lib/vs/x64/Release/torch.lib"
    # Debug
    # ADDON_LIBS += "libs/win/lib/vs/x64/Debug/c10.lib"
    # ADDON_LIBS += "libs/win/lib/vs/x64/Debug/caffe2_nvrtc.lib"
    # ADDON_LIBS += "libs/win/lib/vs/x64/Debug/c10_cuda.lib"
    # ADDON_LIBS += "libs/win/lib/vs/x64/Debug/torch.lib"

    ADDON_LIBS += "C:/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64/nvToolsExt64_1.lib"

    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cudart_static.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cufft.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/curand.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cublas.lib"
    ADDON_LIBS += "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/lib/x64/cudnn.lib"

linuxarmv6l:
linuxarmv7l:
android/armeabi:    
android/armeabi-v7a:    
osx:
    # osx/iOS only, any framework that should be included in the project
    # ADDON_FRAMEWORKS =
    
    ADDON_INCLUDES += "libs/osx/include"
    ADDON_INCLUDES += "libs/osx/include/torch/csrc/api/include"

    ADDON_LIBS += "libs/osx/lib/libc10.dylib"
    ADDON_LIBS += "libs/osx/lib/libtorch.dylib"
    ADDON_LIBS += "libs/osx/lib/libtorch_cpu.dylib"
    ADDON_LIBS += "libs/osx/lib/libiomp5.dylib"

ios:
tvos:
