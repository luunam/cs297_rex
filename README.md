# TEAM REX Spring 2021
In this project, we aim to predict supply and demand using hierarchical models. This repository stores the source code for our project.

The structure of the repository is as follow:
- ```data```: contains some of the datasets that we use. Large dataset are stored in [Google Drive](https://drive.google.com/drive/u/1/folders/0ADNiRHNQgWGcUk9PVA) (permissions required). When develop locally, large dataset can be pulled to local machine/deepnote, but must be put to ```.gitignore```, or exclude manually, when commit
- ```notebook```: notebooks for analysis work. All collaborations on notebooks should be done on [DeepNote](https://deepnote.com/project/1c850c61-d934-4c85-b16d-3cb283df0c84). To run the notebooks locally, pull the repository and run ```jupyter notebook``` from the root folder
    - Baseline_model.ipynb: main notebook for baseline model 
    - Developed_model.ipynb: main notebook to build hierarchical model using PyMC3
    - ```archive```: store old notebooks that are no longer use, but should be kept for future references. Explicit storage is necessary since Git version control does not work well with ```jupyter notebook```
- ```submissions```: contain reports and the presentation for 3 milestones
- ```scripts```: include python scripts for the project. Inside this folder:
    - ```em_hierarchical.py``` contains EM implementation with XGBoost. An example on how to use the function can be found in ```notebooks/em_run_sample.ipynb```