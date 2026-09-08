# EEG Seizure Onset Zone Pipeline — Built From the Ground Up

This repo documents a step-by-step rebuild of a deep learning pipeline
for localizing the Seizure Onset Zone (SOZ) from scalp EEG, using the
public [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/).

Rather than presenting a finished pipeline, this repo is structured as
a learning path: each numbered module builds one concept from first
principles, moving from basic signal processing up to the generative
and self-supervised methods used in the final architecture.

## Roadmap

1. **EEG signal basics & preprocessing** — loading, filtering, epoching *(this commit)*
2. Classical ML baseline (hand-crafted features + SVM/Random Forest, LOSO cross-validation)
3. Basic CNN classifier for seizure detection
4. Autoencoder fundamentals
5. Variational Autoencoder (VAE)
6. Self-supervised pretraining, simplified (masked-signal reconstruction)
7. Diffusion models from scratch
8. Reassembling the full three-stage pipeline

## Dataset

This project uses the CHB-MIT Scalp EEG Database (PhysioNet), a public
dataset of scalp EEG recordings from pediatric subjects with
intractable seizures. It is not included in this repo — download it
yourself from PhysioNet and place it under `data/` (gitignored).

> Shoeb, A. (2009) *Application of Machine Learning to Epileptic
> Seizure Onset Detection and Treatment.* PhD Thesis, MIT.
>
> Goldberger, A., et al. (2000) PhysioBank, PhysioToolkit, and
> PhysioNet. *Circulation* 101(23):e215–e220.

## Setup

```bash
pip install -r requirements.txt
```

## Author

Doaa Al-Desouky — PhD researcher, Computer Science, Mansoura University.
Deep learning for biomedical signals and images.
"# eeg-soz-pipeline" 
