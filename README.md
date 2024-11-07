
# PnP-FISTA Repository

This repository features code for conducting experiments with PnP-FISTA using data-driven denoisers.

## Experiments Overview

1. **Superresolution**: The [superresolution demo notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/superresolution_demo.ipynb) showcases an implementation of superresolution utilizing the *DSG-NLM* denoiser. It includes the forward operator $ A $, the *PnP-FISTA* algorithm, and the resulting reconstructions.

2. **Deblurring**: The [deblurring demo notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/deblurring_demo.ipynb) presents an implementation of deblurring using the *DSG-NLM* denoiser.

3. **Divergence of DRUNet**: The [DRUNet divergence example notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/drunet_divergence_example.ipynb) illustrates a practical scenario in which *PnP-ISTA* with the *DRUNet* denoiser diverges.



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



### Models for deep denoisers
Two denoisers, DRUNet and DnCNN (MMO, Pesquet et. al. 2020) are imported through [Deepinv](https://deepinv.github.io/deepinv/). The models can be downloaded through [dependencies.sh](https://github.com/arghyasinha/PnP-FISTA/blob/main/dependencies.sh) and also can be found [here](https://deepinv.github.io/deepinv/deepinv.denoisers.html).