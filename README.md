# 🚀 PnP-FISTA: Experiments & Code

This repository contains the code and experiments for  
**"FISTA Iterates Converge Linearly for Denoiser-Driven Regularization"**.

- 📰 Published Version: [SIIMS](https://epubs.siam.org/doi/10.1137/24M1656530)
- 📝 arXiv Version: [2411.10808](https://arxiv.org/abs/2411.10808)

---

## 👨‍💻 Authors

- [Arghya Sinha](https://arghyasinha.github.io)
- [Kunal N. Chaudhury](https://sites.google.com/site/kunalnchaudhury/home)

---

## 🧪 Experiments Overview

1. **Superresolution:**  
   The [superresolution demo notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/superresolution_demo.ipynb) demonstrates superresolution using the *DSG-NLM* denoiser, including the forward operator $A$, the *PnP-FISTA* algorithm, and the resulting reconstructions.

2. **Deblurring:**  
   The [deblurring demo notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/deblurring_demo.ipynb) showcases deblurring with the *DSG-NLM* denoiser.

3. **Divergence of DRUNet:**  
   The [DRUNet divergence example notebook](https://github.com/arghyasinha/PnP-FISTA/blob/main/drunet_divergence_example.ipynb) highlights a case where *PnP-ISTA* with the *DRUNet* denoiser diverges.

---

## ⚙️ Setup Instructions

### 1️⃣ Activate Conda Environment

```bash
# Create a new conda environment with Python 3.8.18
conda create --name myenv python=3.8.18

# Activate the environment
conda activate myenv
```

### 2️⃣ Install Required Packages

Make sure you are in the project's root directory (where `dependencies.sh` is located):

```bash
sh dependencies.sh
```

---

## 🧩 Models for Deep Denoisers

The repository uses two deep denoisers:

- **DRUNet**
- **DnCNN** ([MMO, Pesquet et al., 2020](https://deepinv.github.io/deepinv/deepinv.denoisers.html))

These are imported through [Deepinv](https://deepinv.github.io/deepinv/).  
You can download these models by running [dependencies.sh](https://github.com/arghyasinha/PnP-FISTA/blob/main/dependencies.sh) or directly from the [Deepinv denoisers page](https://deepinv.github.io/deepinv/deepinv.denoisers.html).

---

## 📚 Citation

If you use this code or find it helpful, please cite:

```bibtex
@article{doi:10.1137/24M1656530,
    author = {Sinha, Arghya and Chaudhury, Kunal N.},
    title = {Short Communication: FISTA Iterates Converge Linearly for Denoiser-Driven Regularization},
    journal = {SIAM Journal on Imaging Sciences},
    volume = {18},
    number = {1},
    pages = {SC1-SC15},
    year = {2025},
    doi = {10.1137/24M1656530},
}
```
