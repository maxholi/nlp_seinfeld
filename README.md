# nlp_seinfeld
Modularized code to run a predictive model for Seinfeld character quote attribution using NLP

This repo contains scripts which perform machine learning models on the scripts of the TV show Seinfeld in order to predict which character said which quote. The results can be hosted as an API REST service as well.

## Dataset

The raw data can be found at https://www.kaggle.com/thec03u5/seinfeld-chronicles/data#scripts.csv and contains 54,616 lines from 174 episodes of  Seinfeld. This raw data is located in the `data/` folder of the repo.

## Repo structure 

```
├── README.md                         <- You are here
│
├── data
|   ├── scripts.csv                   <- raw data containing Seinfeld scripts 
|
├── artifacts                         <- Directory for all artifacts created during modeling process such as model files and intermediate data files after pre-processing
|
├── scripts                           <- Directory containing scripts for data processing and modeling 
|   ├── create_data.py                <- Script acquiring raw data and doing initial data cleaning and feature creation
|   ├── pre_process.py                <- Script for pre_processing text (tokenizing, normalizing, etc) 
|   ├── model_pre_training.py         <- Script for creating data combining tf-idf features and other features for model training     
|   ├── training_experiments.py       <- Script for running experiments and model tuning to find optimal ML model
|   |── fit_model.py                  <- Script for fitting the optimal model and evaluating the results
|   |── predict.py                    <- Script for running predictions on new data
|   |── api.py                        <- Script for hosting predictions as API REST service
|
├── requirements.txt                  <- Python package dependencies for the Flask App only
├── api_result_example.png            <- example of a new prediction on the API rest service

```


