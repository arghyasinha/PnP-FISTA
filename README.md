
# PnP-FISTA
This repository contains codes for experiments on PnP-FISTA with data-driven denoisers.
Currently the repository contains three experiments. More will get added soon.

## Experiments
1. ***Superresolution***: The [notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/superresolution_demo.ipynb) contains an implementation of superresolution with *DSG-NLM* denoiser. This includes an implementation of the forward operator $A$ , *PnP-FISTA* and the reconstructions.
2. ***Deblurring***: The [notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/deblurring_demo.ipynb) contains an implementation of deblurring with DSG-NLM denoiser.
3. ***Divergence of DRUNet***: The [notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/drunet_divergence_example.ipynb) contains an example of a practical case where PnP-ISTA with *DRUNet* denoiser diverges.



## Instructions to Activate Conda Environment and Install Required Packages

## Step 1: Activate Conda Environment

1. Open your terminal or command prompt.

2. Create a new conda environment with Python version 3.8.18 (if you haven't already):

    ```sh
    conda create --name myenv python=3.8.18
    ```

3. Activate the newly created conda environment:

    ```sh
    conda activate myenv
    ```

## Step 2: Install Required Packages

1. Ensure you are in the root directory of your project where `dependencies.sh` is located.

2. Run the `dependencies.sh` script to install the required packages:

    ```sh
    sh dependencies.sh
    ```



### Models
Two denoisers, DRUNet and DnCNN (Pesquet et. al.) are imported through [Deepinv](https://deepinv.github.io/deepinv/). The models are stored in `models` folder and also can be found [here](https://deepinv.github.io/deepinv/deepinv.denoisers.html).