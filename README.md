# Neural Style Transfer
This project demonstrates the integration of PyTorch and OpenCV to process images and videos. It uses a pre-trained models to transfer style to specified to an image or a video.

## Prerequisites
- C++ Compiler (e.g., GCC or Clang)
- CMake
- OpenCV 4.x
- LibTorch (PyTorch C++ API)

Ensure that you have installed the required libraries (OpenCV and LibTorch) on your system.

## Build project:
```bash
mkdir build
cd build
cmake ..
make
```

## Usage
```bash
./nst <type> <resolution_width> <resolution_height> <style> <src> <dst>
```

Command Parameters:
- `type:` Specifies the type of input, either `image` or `video`. This parameter tells the application how to handle the input file, either as a single image (.jpg) or as a video (.mp4).

- `resolution_width`: The width of the output file in pixels. This defines the horizontal size of the processed output.

- `resolution_height`: The height of the output file in pixels. This defines the vertical size of the processed output.

- `style`: The name of the style model to use for the transfer. This parameter should correspond to a pre-trained model file that implements a particular artistic style.

- `src`: The path to the source file. This is the input image or video file that you want to apply the style transfer to.

- `dst`: The destination path where the stylized output will be saved. This should be a valid path where the application can write the processed image or video.

## Example
Image style transfer:
```bash
./nst image 640 480 mosaic .../example.jpg .../processed_example.jpg
```

Video style transfer:
```bash
./nst video 640 480 udnie .../example.mp4 .../processed_example.mp4
```

Warning: The resolution width and height will be rounded to the smallest number divisible by 4 greater than the given number.

## Styles

The `./models` directory contains pre-trained models for four different styles. You can apply any of these styles to your images or videos by specifying the appropriate model name in the usage command. Below are the descriptions and model names for each style:

1. **candy** - Produces vibrant and colorful candy-like textures.
2. **mosaic** - Applies a classic mosaic texture, reminiscent of ancient tiled artwork.
3. **rain_princess** - Transfers the soft and dreamy aesthetics of the "Rain Princess" painting.
4. **udnie** - Infuses images with the swirling modern art style seen in Francis Picabia's "Udnie".

## Acknowledgements
This application utilizes style transfer models that were originally developed by contributors to the [Fast Neural Style](https://github.com/pytorch/examples/tree/main/fast_neural_style) project, hosted on PyTorch's official examples repository.

I have adapted these models by tracing them with PyTorch to create TorchScript versions suitable for deployment in c++ environment.


