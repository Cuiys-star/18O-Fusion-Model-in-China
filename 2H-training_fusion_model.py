import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import os

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load preprocessed data and models
scaler = joblib.load("your_path/d2H_xgb.2.pp")
numeric_predictors_xgb = pd.read_csv("your_path/d.model.inputs.2.csv")['Variable'].tolist()
numeric_predictors_fusion = pd.read_csv("your_path/d.model.inputs.2.csv")['Variable'].tolist()
xgb_model = joblib.load("your_path/d2H_xgb.model2.pkl")
rf_model = joblib.load("your_path/d2H_rf.model2.pkl")


# Load the data and preprocess
def load_and_clean_data(file_path):
    logging.info(f'Loading data from {file_path}')
    data = pd.read_csv(file_path, encoding='utf-8')
    data.columns = data.columns.str.strip()
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = data[col].str.replace(',', '').astype(float)
            except ValueError:
                pass
    return data


def process_time_columns(data):
    def convert_timedelta_to_seconds(timedelta_str):
        try:
            days, time = timedelta_str.split(' days ')
            hours, minutes, seconds = map(float, time.split(':'))
            total_seconds = int(days) * 86400 + int(hours) * 3600 + int(minutes) * 60 + seconds
            return total_seconds
        except:
            return timedelta_str

    for col in data.columns:
        if data[col].dtype == 'object' and 'days' in data[col].iloc[0]:
            data[col] = data[col].apply(convert_timedelta_to_seconds)
    return data


# Load data
training_import = load_and_clean_data("your_path/training-all.csv")
training_import = process_time_columns(training_import)
test_import = load_and_clean_data("your_path/test-all.csv")
test_import = process_time_columns(test_import)
validation_import = load_and_clean_data("your_path/validation-all.csv")
validation_import = process_time_columns(validation_import)

training = training_import.copy()
test = test_import.copy()
validation = validation_import.copy()

# Data preprocessing
training_tr_xgb = scaler.transform(training[numeric_predictors_xgb])
test_tr_xgb = scaler.transform(test[numeric_predictors_xgb])
validation_tr_xgb = scaler.transform(validation[numeric_predictors_xgb])

training_tr_fusion = scaler.transform(training[numeric_predictors_fusion])
test_tr_fusion = scaler.transform(test[numeric_predictors_fusion])
validation_tr_fusion = scaler.transform(validation[numeric_predictors_fusion])

training_target = training["d2Hf"].values
test_target = test["d2Hf"].values
validation_target = validation["d2Hf"].values

dtrain = xgb.DMatrix(training_tr_xgb, label=training_target)
dtest = xgb.DMatrix(test_tr_xgb, label=test_target)
dvalidation = xgb.DMatrix(validation_tr_xgb, label=validation_target)


# Calculate multiple indicators
def calculate_metrics(model, features, target, model_type='xgb'):
    if model_type == 'xgb':
        predictions = model.predict(features)
    else:
        predictions = model.predict(features)
    rmse = np.sqrt(mean_squared_error(target, predictions))
    r2 = r2_score(target, predictions)
    return rmse, r2, predictions


# XGBoost prediction
train_rmse_xgb, train_r2_xgb, train_pred_xgb = calculate_metrics(xgb_model, dtrain, training_target, model_type='xgb')
test_rmse_xgb, test_r2_xgb, test_pred_xgb = calculate_metrics(xgb_model, dtest, test_target, model_type='xgb')
validation_rmse_xgb, validation_r2_xgb, validation_pred_xgb = calculate_metrics(xgb_model, dvalidation,
                                                                                validation_target, model_type='xgb')

# Random Forest prediction
train_rmse_fusion, train_r2_fusion, train_pred_fusion = calculate_metrics(rf_model, training_tr_fusion, training_target,
                                                              model_type='rf')
test_rmse_fusion, test_r2_fusion, test_pred_fusion = calculate_metrics(rf_model, test_tr_fusion, test_target, model_type='rf')
validation_rmse_fusion, validation_r2_fusion, validation_pred_fusion = calculate_metrics(rf_model, validation_tr_fusion,
                                                                             validation_target, model_type='rf')

# Weight Fixing
a1 = 2 / train_rmse_xgb
a2 = 1 / train_rmse_fusion
total_a = a1 + a2

weight_xgb = a1 / total_a
weight_fusion = a2 / total_a


# Weighted average
def weighted_average(pred1, pred2, weight1, weight2):
    return weight1 * pred1 + weight2 * pred2


train_pred_fusion = weighted_average(train_pred_xgb, train_pred_fusion, weight_xgb, weight_fusion)
test_pred_fusion = weighted_average(test_pred_xgb, test_pred_fusion, weight_xgb, weight_fusion)
validation_pred_fusion = weighted_average(validation_pred_xgb, validation_pred_fusion, weight_xgb, weight_fusion)


# Calculate metrics for the fusion model
def fusion_metrics(pred_fusion, target):
    rmse = np.sqrt(mean_squared_error(target, pred_fusion))
    r2 = r2_score(target, pred_fusion)
    return rmse, r2


train_rmse_fusion, train_r2_fusion = fusion_metrics(train_pred_fusion, training_target)
test_rmse_fusion, test_r2_fusion = fusion_metrics(test_pred_fusion, test_target)
validation_rmse_fusion, validation_r2_fusion = fusion_metrics(validation_pred_fusion, validation_target)

logging.info(f"Fusion model train RMSE: {train_rmse_fusion}, R^2: {train_r2_fusion}")
logging.info(f"Fusion model test RMSE: {test_rmse_fusion}, R^2: {test_r2_fusion}")
logging.info(f"Fusion model validation RMSE: {validation_rmse_fusion}, R^2: {validation_r2_fusion}")

# Save fusion model
save_path = "E:/learn/data/test/model/d2H/"
os.makedirs(save_path, exist_ok=True)

fusion_model = {
    'xgb_model': xgb_model,
    'rf_model': rf_model,
    'scaler': scaler,
    'weights': {
        'xgb': weight_xgb,
        'rf': weight_fusion
    }
}
joblib.dump(fusion_model, os.path.join(save_path, "d2H_fusion.model.pkl"))
joblib.dump(scaler, os.path.join(save_path, "d2H_fusion.pp"))


# Importance of preserving integration model features
def save_fusion_feature_importance(xgb_model, rf_model, predictors, weight_xgb, weight_fusion, file_path):
    # The feature importance of the XGBoost model
    xgb_importance = xgb_model.get_score(importance_type='weight')
    feature_names = {f"f{i}": name for i, name in enumerate(predictors)}
    xgb_importance_df = pd.DataFrame(list(xgb_importance.items()), columns=['Feature', 'XGB_Importance'])
    xgb_importance_df['Variable'] = xgb_importance_df['Feature'].map(feature_names)
    xgb_importance_df = xgb_importance_df[['Variable', 'XGB_Importance']]

    # The feature importance of the Random Forest model
    rf_importance = rf_model.feature_importances_
    rf_importance_df = pd.DataFrame({'Variable': predictors, 'RF_Importance': rf_importance})

    # Combine the feature importance and calculate the weighted average
    importance_df = pd.merge(xgb_importance_df, rf_importance_df, on='Variable', how='outer').fillna(0)
    importance_df['Importance'] = (importance_df['XGB_Importance'] * weight_xgb) + (
                importance_df['RF_Importance'] * weight_fusion)

    # Save to file
    importance_df = importance_df[['Variable', 'Importance']]
    importance_df.to_csv(file_path, index=False)


save_fusion_feature_importance(xgb_model, rf_model, numeric_predictors_xgb, weight_xgb, weight_fusion,
                               os.path.join(save_path, "d2H_fusion_importance.csv"))
