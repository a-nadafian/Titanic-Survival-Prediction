# Titanic Survival Prediction

This project is a classic data science challenge to predict the survival of passengers on the Titanic. The goal is to build a model that can predict whether a passenger survived or not based on features like age, sex, class, etc.


This repository contains both the exploratory analysis notebooks and a refactored, documented data processing and model training pipeline.
=======
This repository contains both the exploratory analysis notebooks and a refactored, documented data processing pipeline.


## Project Structure

The project is structured as follows:

- `data/`: Contains the raw and processed data.
  - `raw/`: The original `train.csv` and `test.csv` files.

  - `processed/`: The output of the data processing pipeline, including the final submission file.
- `notebooks/`: Contains the original Jupyter notebooks used for exploratory data analysis and initial modeling. These are useful for understanding the step-by-step process of data exploration and feature discovery.
- `src/`: Contains the Python scripts for the refactored and documented data processing and model training pipeline.
  - `data_preprocessing.py`: Functions for loading and cleaning the data.
  - `feature_engineering.py`: Functions for creating new features.
  - `model_training.py`: Functions for training the model, tuning hyperparameters, and generating a submission file.

  - `processed/`: The output of the data processing pipeline.
- `notebooks/`: Contains the original Jupyter notebooks used for exploratory data analysis and initial modeling. These are useful for understanding the step-by-step process of data exploration and feature discovery.
- `src/`: Contains the Python scripts for the refactored and documented data processing pipeline.
  - `data_preprocessing.py`: Functions for loading and cleaning the data.
  - `feature_engineering.py`: Functions for creating new features.

  - `main.py`: The main script to run the entire pipeline.
- `requirements.txt`: The Python dependencies for this project.

## Setup

To set up the environment, you will need to install the required Python packages. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

## Usage

### Running the Pipeline


To run the entire data processing and model training pipeline, simply execute the `main.py` script from the root of the repository:

To run the entire data processing pipeline, simply execute the `main.py` script from the root of the repository:


```bash
python src/main.py
```

This will:
1. Load the raw data from `data/raw`.
2. Perform data cleaning and preprocessing.
3. Engineer new features.

4. Tune the hyperparameters for the CatBoost model using Optuna.
5. Train the final model with the best hyperparameters.
6. Generate a submission file named `submission.csv` in the `data/processed` directory.

After running the pipeline, the submission file will be ready to be submitted to the Kaggle competition.

4. Save the fully processed data to `data/processed/titanic_fully_processed.csv`.

After running the pipeline, the processed data will be ready for model training and evaluation.


### Exploratory Notebooks

The `notebooks` directory contains the original Jupyter notebooks, which provide a detailed, step-by-step walkthrough of the data analysis process. These are a valuable resource for understanding the data and the feature engineering decisions.
