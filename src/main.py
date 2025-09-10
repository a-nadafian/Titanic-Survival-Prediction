import os
import pandas as pd
import data_preprocessing
import feature_engineering
import model_training

def main():
    """
    Main function to run the Titanic data processing pipeline.
    """
    # Define the data directory relative to this script's location
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "..", "data")

    # Preprocess the data
    print("Starting data preprocessing...")
    df = data_preprocessing.preprocess(data_dir)
    print("Data preprocessing complete.")

    # Engineer features
    print("Starting feature engineering...")
    df = feature_engineering.engineer_features(df)
    print("Feature engineering complete.")

    # Separate train and test data
    train_df = df[df['Survived'].notna()]
    test_df = df[df['Survived'].isna()].drop(columns=['Survived'])

    X = train_df.drop(columns=['Survived', 'PassengerId'])
    y = train_df['Survived']

    # Tune hyperparameters
    print("Tuning hyperparameters...")
    best_params = model_training.tune_hyperparameters(X, y)
    print("Hyperparameter tuning complete.")

    # Train final model
    print("Training final model...")
    model = model_training.train_model(X, y, best_params)
    print("Model training complete.")

    # Generate submission file
    print("Generating submission file...")
    test_passenger_ids = test_df['PassengerId']
    test_df = test_df.drop(columns=['PassengerId'])
    submission_df = model_training.generate_submission(model, test_df, test_passenger_ids)

    # Save the submission file
    output_dir = os.path.join(data_dir, "processed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    submission_path = os.path.join(output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to '{submission_path}'")

if __name__ == '__main__':
    main()
