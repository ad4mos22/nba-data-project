# analysis2/src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def buildTS(df, player_id):
    """
    Build a time series DataFrame for a specific player based on their game minutes played.
    This function processes the input DataFrame by:
    1. Dropping rows with missing values in the 'Minutes Played' column.
    2. Filtering rows where 'Minutes Played' is in the 'mm:ss' format.
    3. Sorting the DataFrame by the 'Date' column.
    4. Converting 'Minutes Played' from 'mm:ss' format to total minutes.
    5. Calculating rolling averages, medians, and standard deviations for the last 3, 5, and 10 games.
    6. Dropping rows with any missing values after the calculations.
    7. Adding a 'Player ID' column to the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing player game data.
        player_id (int): The unique identifier for the player.
    Returns:
        pd.DataFrame: A processed DataFrame with additional time series features for the player.
    Raises:
        ValueError: If there is an error converting 'Minutes Played' to timedelta.
    """
    # Drop rows with missing values in 'Minutes Played' column
    df = df.dropna(subset=['Minutes Played'])

    # Check if 'Minutes Played' is in 'mm:ss' format
    valid_minutes_played = df['Minutes Played'].str.match(r'^\d+:\d{2}$')
    df = df[valid_minutes_played].copy()

    df = df.sort_values(by='Date')

    # Convert 'Minutes Played' from mm:ss to timedelta
    try:
        df['Minutes Played'] = pd.to_timedelta(df['Minutes Played'] + ':00')
    except Exception as e:
        raise ValueError(f"Error converting 'Minutes Played' to timedelta: {e}")

    # Convert to minutes
    df['Minutes Played'] = df['Minutes Played'].dt.total_seconds() / 60

    df['prevgm'] = df['Minutes Played'].shift(1)
    df['pavg3'] = df['Minutes Played'].rolling(window=3).mean()
    df['pavg5'] = df['Minutes Played'].rolling(window=5).mean()
    df['pavg10'] = df['Minutes Played'].rolling(window=10).mean()
    df['pmed3'] = df['Minutes Played'].rolling(window=3).median()
    df['pmed5'] = df['Minutes Played'].rolling(window=5).median()
    df['pmed10'] = df['Minutes Played'].rolling(window=10).median()
    df['pstd3'] = df['Minutes Played'].rolling(window=3).std()
    df['pstd5'] = df['Minutes Played'].rolling(window=5).std()
    df['pstd10'] = df['Minutes Played'].rolling(window=10).std()
    
    df.dropna(inplace=True)
    df['Player ID'] = player_id  # Add player ID to the DataFrame

    return df

def build_TrainTest(df):
    """
    Splits the input DataFrame into features and target variable for training and testing.
    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
    Returns:
        tuple: A tuple containing two elements:
            - X (pandas.DataFrame): The DataFrame containing the feature columns.
            - y (pandas.Series): The Series containing the target variable.
    """
    # Define the features and target variable
    features = ['prevgm', 'pavg3', 'pavg5', 'pavg10', 'pmed3', 'pmed5', 'pmed10', 'pstd3', 'pstd5', 'pstd10']
    target = 'Minutes Played'
    
    X = df[features]
    y = df[target]
    
    return X, y

def RunLinearModel(x_train, y_train, x_test, y_test):
    """
    Trains a linear regression model using the provided training data and makes predictions on the test data.
    Parameters:
    x_train (array-like or pandas DataFrame): The input features for training the model.
    y_train (array-like or pandas Series): The target values for training the model.
    x_test (array-like or pandas DataFrame): The input features for testing the model.
    y_test (array-like or pandas Series): The target values for testing the model.
    Returns:
    lm (LinearRegression): The trained linear regression model.
    yhat_lm (array-like): The predicted values for the test data.
    """
    # Initialize and train the linear regression model
    lm = LinearRegression()
    lm.fit(x_train, y_train)
    
    # Make predictions
    yhat_lm = lm.predict(x_test)
        
    return lm, yhat_lm

def randomForest(x_train, y_train, x_test, y_test):
    """
    Trains a Random Forest regressor on the provided training data and evaluates it on the test data.
    Parameters:
    x_train (pd.DataFrame): Training features.
    y_train (pd.Series or np.array): Training target values.
    x_test (pd.DataFrame): Test features.
    y_test (pd.Series or np.array): Test target values.
    Returns:
    tuple: A tuple containing:
        - rf (RandomForestRegressor): The trained Random Forest model.
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
    
    return rf, yhat_rf, feature_importances