import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # To save the model

# File paths
processed_data_path = 'S:\\vertex\\Final\\data\\processed_data.csv'
model_output_path = 'S:\\vertex\\Final\\models\\regression_model.pkl'

def load_data():
    """
    Load the processed data and handle missing values in the target column.

    Returns:
        pd.DataFrame: Loaded dataset containing features and target.
    """
    try:
        # Load the processed data
        data = pd.read_csv(processed_data_path)
        
        # Check if required columns are present
        if not {'MolecularWeight', 'LogP', 'Activity'}.issubset(data.columns):
            raise ValueError("The dataset is missing one of the required columns: 'MolecularWeight', 'LogP', 'Activity'.")
        
        # Check for NaN values in the 'Activity' column (target)
        print(f"Number of NaN values in 'Activity' column: {data['Activity'].isna().sum()}")

        # Remove rows with NaN values in the target column 'Activity'
        data = data.dropna(subset=['Activity'])

        # Check after removing NaNs
        print(f"Data after removing NaNs: {len(data)} records remaining.")

        print("Data loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def train_model(data):
    """
    Train a regression model on the processed data.

    Args:
        data (pd.DataFrame): Processed dataset with features and target.

    Returns:
        RandomForestRegressor: Trained regression model.
    """
    try:
        # Define features (X) and target (y)
        X = data[['MolecularWeight', 'LogP']]
        y = data['Activity']
        
        # Check for NaN values in target (y) before training
        if y.isna().sum() > 0:
            raise ValueError(f"Target contains NaN values: {y.isna().sum()} NaNs.")
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model training complete. Mean Squared Error (MSE) on test data: {mse}")
        
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

def save_model(model):
    """
    Save the trained model to disk.

    Args:
        model (RandomForestRegressor): Trained regression model.
    """
    try:
        joblib.dump(model, model_output_path)
        print(f"Model saved to {model_output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def main():
    """
    Main function to load data, train the model, and save it.
    """
    data = load_data()
    
    if data is not None:
        # Train the model
        model = train_model(data)
        
        if model:
            # Save the trained model
            save_model(model)
        else:
            print("Model training failed.")
    else:
        print("Data loading failed.")

# Corrected main execution check
if __name__ == "__main__":  # Corrected line here
    main()
