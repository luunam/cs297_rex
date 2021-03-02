# TEAM REX Spring 2021
In this project, we aim to predict supply and demand using hierarchical models. This repository stores the source code for our project.

The structure of the repository is as follow:
- ```data```: contains some of the datasets that we use. Large dataset are stored in [Google Drive](https://drive.google.com/drive/u/1/folders/0ADNiRHNQgWGcUk9PVA) (permissions required). When develop locally, large dataset can be pulled to local machine/deepnote, but must be put to ```.gitignore```, or exclude manually, when commit
- ```notebook```: notebooks for analysis work. All collaborations on notebooks should be done on [DeepNote](https://deepnote.com/project/1c850c61-d934-4c85-b16d-3cb283df0c84). To run the notebooks locally, pull the repository and run ```jupyter notebook``` from the root folder
    - Baseline_model.ipynb: main notebook to 
    - Developed_model.ipynb: main notebook to build hierarchical model using PyMC3
    - ```archive```: store old notebooks that are no longer use, but should be kept for future references. Explicit storage is necessary since GitHub version control does not work well with ```jupyter notebook```
- ```scripts```: utility Python scripts 
- ```milestone1```: contain reports and the presentation for milestone 1