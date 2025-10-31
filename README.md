# Trainable Logic Gate Networks (LGN)

## Prerequisites

This repo uses the *uv* package manager, see the [uv repo](https://github.com/astral-sh/uv/) for installation instructions or simply install *uv* via pip:

```bash
pip install uv
```

## How to run

Run the script to train LGN architecture with 2 layers of 8000 gates each and learnable connections between them.
```bash
uv run mnist.py
```
It will produce small 48K parameters 2-layer Logic Gate Network with an accuracy of **97.7%** on the MNIST dataset.

## Config

You can override almost any of the training parameters by supplying environment variables!

Put variables in the command line just before calling our script, for example:
```bash
GATE_ARCHITECTURE="[4000, 2550]" INTERCONNECT_ARCHITECTURE="[[],[-1]]" C_INIT="NORMAL" IMG_WIDTH=16 IMG_CROP=22 LEARNING_RATE=0.03 EPOCHS=100 SEED=230646   uv run mnist.py
```

It will produce ultra tiny 15K parameters 2-layer Logic Gate Network that is still capable of a remarkable **97.6%** accuracy on the MNIST dataset.

The following is the subset of the most interesting and useful configuration variables:
- `DATASET`, `LEARNING_RATE`, `EPOCHS`,
- `GATE_ARCHITECTURE`, `INTERCONNECT_ARCHITECTURE`,
- `SCALE_LOGITS` and `MANUAL_GAIN` see below for more defails.

### Dataset and Image resolution

- **`DATASET`** currently supports the following settings: `MNIST` (default), `FashionMNIST`, `KMNIST`, `QMNIST`, `EMNIST`, `CIFAR10`.

- **`BINARIZE_IMAGE_TRESHOLD`** supports an array of thresholds for thermometer encoding of the input pixels. For example `BINARIZE_IMAGE_TRESHOLD="[0.25, 0.5, 0.75]"` will produce 3 inputs for each input pixel. Respecitively pixel of value `0.3` will result in a binary vector of `1,0,0`, value of `0.6` will become `1,1,0` and so on. It is useful for the datasets like **FashionMNIST** that contain grayscale natural images.

- **`RGB_TO`** defines which color space to convert input images before passing into thermometer encoder controlled by `BINARIZE_IMAGE_TRESHOLD`. This setting is useful for the datasets that contain natural color images like **CIFAR10**.
  - `MONO` (default) will convert images to grayscale aka monochrome,
  - `RGB` will keep color images input in sRGB color space and **currently gives the best results**,
  - `YUV` will convert color images to Y'U'V', Luminocity (Y') channel encodes brightness and two Chrominocity channels (U'V') encode color information,
  - `LAB` will convert RGB pixels into a modified CIE Lab perceptual color space [CIE Lab](https://en.wikipedia.org/wiki/Oklab_color_space), theoreticallly perceptual color space should be more suitable for Machine Learning than RGB, but we are not there yet!
- **`IMG_WIDTH`** scales input image to a particular resolution, by default it takes the value of 28 for MNIST datasets and 32 for CIFAR.

- **`IMG_CROP`** crops input image, by default it is equal to `IMG_WIDTH`

To train on CIFAR10 dataset while converting every input pixel into 9 binary element vector, run:
```bash
DATASET="CIFAR10" BINARIZE_IMAGE_TRESHOLD="[0.25, 0.5, 0.75]" RGB_TO="RGB"   C_INIT="NORMAL" SCALE_LOGITS="AUTOTAU"   uv run mnist.py
```

### Network architecture
- **`GATE_ARCHITECTURE`** specifies a number of logic gates for each layer of the network. `GATE_ARCHITECTURE="[4000, 2550]"` creates a 2 layer network with 4000 gates for the first layer and 2550 for the last one,  `GATE_ARCHITECTURE="[8000, 8000, 8000, 8000]"` creates a 4 layer architecture with 8000 gates each.

- **`INTERCONNECT_ARCHITECTURE`** specifies a type (and parameters) for interconnect between the layers where the first parameter specifies interconnect between the input and the first layer:
  - `[]` trainable interconnect,
  - `[k, 5]` trainable "TopK" interconnect which will allow each gate to select one of the 5 random connections chosen at the beginning of the training. NOTE that using value higher than 8 can significantly slow down the training,
  - `[-1]` fixed (non trainable) random interconnect.
 
- **`C_INIT`** accepts either `DIRAC` (default) for sparse initialisation or `NORMAL` for gaussian initialisaion for the trainable interconnect. Natural images and tiny shallow networks might prefer `NORMAL` initialisation.

### Logits
- **`SCALE_LOGITS`** specifies how TAU value is calculated which is crucial for training networks that map to exact logical operations
  - `ADAVAR` (default) will adaptiveley choose TAU during the training,
  - `AUTOTAU` will use heuristic to pick TAU value depending on the size of the output layer, TAU value will be constant during the training,
  - `MANUAL` will use `MANUAL_GAIN` as TAU value.
- **`SCALE_TARGET`** specify modifier parameter for `ADAVAR` and `AUTOTAU` alogirthms
- **`MANUAL_GAIN`** specify TAU value or additional multiplier to TAU value, if `ADAVAR` or `AUTOTAU` are used.

It appears, that `ADOVAR` currently suits better MNIST and `AUTOTAU` is better for CIFAR10.

To quickly, _just in 10 epochs_ train small, but capable **48%** accuracy network on CIFAR10 dataset, run:
```bash
SCALE_LOGITS="AUTOTAU" SCALE_TARGET=0.5   DATASET="CIFAR10" IMG_WIDTH=22 BINARIZE_IMAGE_THRESHOLD="[0.25, 0.5, 0.75]" RGB_TO="RGB"   GATE_ARCHITECTURE="[8000,16000]" INTERCONNECT_ARCHITECTURE="[],[-1]" C_INIT="NORMAL"   LEARNING_RATE=0.075 EPOCHS=10  uv run  mnist.py
```

### Training parameters
- **`SEED`** if not specified, random seed is picked,
- **`BATCH_SIZE`** default is `256`,
- **`EPOCHS`** default is `30`,
- **`LEARNING_RATE`** default is `0.075`.

## Configuration file

To manage configuration settings, you can use optional `.env` file. A sample configuration is available in `.env.example`.  To set up, copy `.env.example` to `.env` and update it as needed.
For security reasons, the `.env` file is excluded from version control to prevent accidental exposure of sensitive data like API keys.
