# analysis/src/model_training.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def buildTS(df, player_id, stat):
    """
    Build a time series DataFrame for a specific player based on a specific stat.
    This function processes the input DataFrame by:
    1. Dropping rows with missing values in the specified stat column.
    2. Sorting the DataFrame by the 'Date' column.
    3. Calculating rolling averages, medians, and standard deviations for the last 3, 5, and 10 games.
    4. Dropping rows with any missing values after the calculations.
    5. Adding a 'Player ID' column to the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing player game data.
        player_id (int): The unique identifier for the player.
        stat (str): The specific stat to be analyzed.
    Returns:
        pd.DataFrame: A processed DataFrame with additional time series features for the player.
    Raises:
        ValueError: If there is an error processing the specified stat.
    """
    # Drop rows with missing values in the specified stat column
    df = df.dropna(subset=[stat])

    # Convert the 'date' column to datetime format
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Sort the DataFrame by the 'date' column
    df = df.sort_values(by='game_date')

    # Convert the stat column to numeric, forcing non-numeric values to NaN
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

    # Calculate rolling statistics
    df['prevgm'] = df[stat].shift(1)
    df['pavg3'] = df[stat].rolling(window=3).mean()
    df['pavg5'] = df[stat].rolling(window=5).mean()
    df['pavg10'] = df[stat].rolling(window=10).mean()
    df['pmed3'] = df[stat].rolling(window=3).median()
    df['pmed5'] = df[stat].rolling(window=5).median()
    df['pmed10'] = df[stat].rolling(window=10).median()
    df['pstd3'] = df[stat].rolling(window=3).std()
    df['pstd5'] = df[stat].rolling(window=5).std()
    df['pstd10'] = df[stat].rolling(window=10).std()
    
    df.dropna(inplace=True)
    df['player_id'] = player_id  # Add player ID to the DataFrame

    return df

def build_TrainTest(df, stat):
    """
    Splits the input DataFrame into features and target variable for training and testing based on a chosen stat.
    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        stat (str): The specific stat to be used as the target variable.
    Returns:
        tuple: A tuple containing two elements:
            - X (pandas.DataFrame): The DataFrame containing the feature columns.
            - y (pandas.Series): The Series containing the target variable.
    """
    # Define the features and target variable
    features = ['prevgm', 'pavg3', 'pavg5', 'pavg10', 'pmed3', 'pmed5', 'pmed10', 'pstd3', 'pstd5', 'pstd10']
    target = stat
    
    X = df[features]
    y = df[target]
    
    return X, y

def RunLinearModel(x_train, y_train, x_test):
    """
    Trains a linear regression model using the provided training data and makes predictions on the test data.
    Parameters:
    x_train (array-like or pandas DataFrame): The input features for training the model.
    y_train (array-like or pandas Series): The target values for training the model.
    x_test (array-like or pandas DataFrame): The input features for testing the model.
    Returns:
    yhat_lm (array-like): The predicted values for the test data.
    """
    # Initialize and train the linear regression model
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    
    # Make predictions
    yhat_lm = lm.predict(x_test)
        
    return yhat_lm

def randomForest(x_train, y_train, x_test):
    """
    Trains a Random Forest regressor on the provided training data and evaluates it on the test data.
    Parameters:
    x_train (pd.DataFrame): Training features.
    y_train (pd.Series or np.array): Training target values.
    x_test (pd.DataFrame): Test features.
    Returns:
    tuple: A tuple containing:
        - yhat_rf (np.array): Predictions made by the Random Forest model on the test data.
        - feature_importances (pd.DataFrame): DataFrame containing feature importances sorted in descending order.
    """
    # Initialize and train the random forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    
    # Make predictions
    yhat_rf = rf.predict(x_test)
        
    # Get feature importances
    features = x_train.columns
    importances = rf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    return yhat_rf, feature_importances