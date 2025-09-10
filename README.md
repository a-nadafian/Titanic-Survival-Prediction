# Titanic Survival Prediction

This project is a classic data science challenge to predict the survival of passengers on the Titanic. The goal is to build a model that can predict whether a passenger survived or not based on features like age, sex, class, etc.

This repository contains a complete data processing pipeline to clean the data, engineer new features, and prepare the data for modeling.

## Project Structure

The project is structured as follows:

- `data/`: Contains the raw and processed data.
  - `raw/`: The original `train.csv` and `test.csv` files.
  - `processed/`: The output of the data processing pipeline.
- `src/`: Contains the Python scripts for the data processing pipeline.
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

To run the entire data processing pipeline, simply execute the `main.py` script from within the `src` directory:

```bash
python src/main.py
```

This will:
1. Load the raw data from `data/raw`.
2. Perform data cleaning and preprocessing.
3. Engineer new features.
4. Save the fully processed data to `data/processed/titanic_fully_processed.csv`.

After running the pipeline, the processed data will be ready for model training and evaluation.
