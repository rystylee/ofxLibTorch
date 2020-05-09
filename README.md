# ofxLibTorch

## Description
an openFrameworks wrapper for [LibTorch](https://pytorch.org/cppdocs/)

## Requirements
- Windows10
- Visual Studio 2019
- CUDA v10.1
- openFramewrks v0.11.0
- LibTorch v1.4.0
- ofxOpenCV

## Install
1. Download LibTorch from https://pytorch.org/get-started/locally/
2. Unzip the downloaded file and copy the contents of the "include" and "lib" directories to the following location

### Windows
- libtorch-win-shared-with-deps-1.4.0/include >> libs/win/include/vs/x64/Release
- libtorch-win-shared-with-deps-1.4.0/lib >> libs/win/lib/vs/x64/Release

### macOS
- libtorch-macos-1.4.0/include >> libs/osx/include
- libtorch-macos-1.4.0/lib >> libs/osx/lib

3. In Xcode Build Settings:
 - Under linking -> runpath search paths, add @loader_path/../Frameworks
 - Under search paths -> use header maps, set NO


4. In Xcode Build Phases -> Run Script, add:

```
rsync -aved "$OF_PATH/addons/ofxLibTorch/libs/osx/lib/libc10.dylib" "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/Frameworks/";
rsync -aved "$OF_PATH/addons/ofxLibTorch/libs/osx/lib/libtorch.dylib" "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/Frameworks/";
rsync -aved "$OF_PATH/addons/ofxLibTorch/libs/osx/lib/libtorch_cpu.dylib" "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/Frameworks/";
rsync -aved "$OF_PATH/addons/ofxLibTorch/libs/osx/lib/libiomp5.dylib" "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/Frameworks/";
```

and

```
install_name_tool -change @rpath/libtorch.dylib @loader_path/../Frameworks/libtorch.dylib "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/MacOS/$PRODUCT_NAME";
install_name_tool -change @rpath/libtorch_cpu.dylib @loader_path/../Frameworks/libtorch_cpu.dylib "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/MacOS/$PRODUCT_NAME";
install_name_tool -change @rpath/libc10.dylib @loader_path/../Frameworks/libc10.dylib "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/MacOS/$PRODUCT_NAME";
install_name_tool -change /DLC/torch/libiomp5.dylib @loader_path/../Frameworks/libiomp5.dylib "$TARGET_BUILD_DIR/$PRODUCT_NAME.app/Contents/MacOS/$PRODUCT_NAME";
```

before

```
echo "$GCC_PREPROCESSOR_DEFINITIONS";
```

## Licence
[MIT](https://github.com/rystylee/ofxLibTorch/blob/master/LICENSE)
