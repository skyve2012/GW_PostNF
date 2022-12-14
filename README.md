# A deep learning model for GW posterior estimation
Posterior estimation of gravitational wave parameters across multiple events using deep learning models


This repository contains the implementation of the model presented in paper "[Statistically-informed deep learning for gravitational wave parameter estimation](https://arxiv.org/abs/1903.01998)". The model is only trained with GW150914 data and can generalize the posterior estimation of physical parameters to multiple events without training and fine-tuning.

`requirements.txt` contains all necessary packages to run this code.

`train_submit.sh` includes command for training the model.

`small_data_for_github.h5` contains sample data for reference.
