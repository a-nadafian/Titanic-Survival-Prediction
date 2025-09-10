import os
import data_preprocessing
import feature_engineering

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

    # Save the fully processed data
    output_dir = os.path.join(data_dir, "processed")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'titanic_fully_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"Fully processed data saved to '{output_path}'")

if __name__ == '__main__':
    main()
