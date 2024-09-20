# src/evaluation.py

from src.model_training import prepare_data, train_linear_model, train_random_forest, evaluate_model

def evaluate_all_models(df):
    trainX, testX, trainY, testY = prepare_data(df)
    
    # Linear Model
    lm = train_linear_model(trainX, trainY)
    lm_rmse = evaluate_model(lm, testX, testY)
    print(f"Linear Model RMSE: {lm_rmse}")
    
    # Random Forest Model
    rf = train_random_forest(trainX, trainY)
    rf_rmse = evaluate_model(rf, testX, testY)
    print(f"Random Forest Model RMSE: {rf_rmse}")

    return lm, rf
