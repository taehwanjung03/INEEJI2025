
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.inspection import partial_dependence

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import shap

# Set font for plots
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Read the CSV
df = pd.read_csv('C:/Users/User/Downloads/Kyushu Datasheets/data_with_press_and_weather.csv', low_memory=False)

# Drop rows with excessive missing values
missing_count = df.isnull().sum().sum()
if missing_count > 0:
    threshold = df.shape[1] // 2
    df = df.dropna(thresh=threshold + 1)
    print(f"Rows with more than half NaN dropped. Remaining rows: {len(df)}")
else:
    print("No missing values found.")

# Target variable
target = '사용전력량'

# Clean and filter '강번'
df['강번'] = pd.to_numeric(df['강번'], errors='coerce')
df = df.dropna(subset=['강번'])
df = df[df['강번'].between(9000, 500000000)]

# Outlier removal
if target in df.columns:
    initial_count_oxygen = len(df)
    df = df[df[target] >= 15000]
    removed_count_oxygen = initial_count_oxygen - len(df)
    print(f"Removed {removed_count_oxygen} rows where {target} < 15000.")
else:
    print(f"Warning: '{target}' column not found.")

# Optional export for inspection
df.to_csv('For_my_use_1.csv', index=False)

# Feature engineering
df['slag time'] = df['소요시간_추가1'] + df['소요시간_추가2'] + df['소요시간_산화'] + df['소요시간_환원']
df['slag rate'] = df['CaO'] / 0.24 / df['slag time']

variables = ['slag rate', 'WindSpeed_Avg (m/s)', 'WindSpeed_Max (m/s)', '소요시간_로 보수/수리',
             '소요시간_주요장입', 'Precipitation (mm)', '연회회수', 'Temperature (°C)']

# --- Define manual repair dates ---
repair_dates = ['2024-02-01', '2024-03-12', '2024-04-25']  # Example
repair_dates = pd.to_datetime(repair_dates)

# --- Convert to datetime ---
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')

# --- Get repair rows ---
repair_df = df[df['날짜'].isin(repair_dates)].copy()

# --- Prediction offset ---
N = 3
repair_df['target_date'] = repair_df['날짜'] + pd.Timedelta(days=N)

# --- Fetch target usage ---
df_by_date = df.set_index('날짜')
df_by_date = df_by_date[~df_by_date.index.duplicated(keep='first')]
repair_df['target_usage'] = repair_df['target_date'].map(df_by_date['사용전력량'])
repair_df = repair_df.dropna(subset=['target_usage'])

# --- Final dataset ---
X = repair_df[variables].fillna(0)
y = repair_df['target_usage']

# --- Split ---
split_index = int(len(X) * 0.7)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# --- Evaluation ---
def evaluate_model(name, model, X_train, y_train, X_test, y_test, df):
    print(f"\n--- {name} Model Evaluation ---")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train, train_pred)
    mape_train = np.mean(np.abs((y_train - train_pred) / y_train)) * 100
    r2_train = r2_score(y_train, train_pred)

    mse_test = mean_squared_error(y_test, test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, test_pred)
    mape_test = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
    r2_test = r2_score(y_test, test_pred)

    print(f"{name} Train Scores:")
    print(f"  MSE:  {mse_train:.2f}")
    print(f"  RMSE: {rmse_train:.2f}")
    print(f"  MAE:  {mae_train:.2f}")
    print(f"  MAPE: {mape_train:.2f}%")
    print(f"  R²:   {r2_train:.3f}")

    print(f"\n{name} Test Scores:")
    print(f"  MSE:  {mse_test:.2f}")
    print(f"  RMSE: {rmse_test:.2f}")
    print(f"  MAE:  {mae_test:.2f}")
    print(f"  MAPE: {mape_test:.2f}%")
    print(f"  R²:   {r2_test:.3f}")

    train_df = df[['날짜']].iloc[X_train.index].copy()
    train_df['Actual'] = y_train.values
    train_df['Predicted'] = train_pred
    train_df['Set'] = 'Train'

    test_df = df[['날짜']].iloc[X_test.index].copy()
    test_df['Actual'] = y_test.values
    test_df['Predicted'] = test_pred
    test_df['Set'] = 'Test'

    combined_df = pd.concat([train_df, test_df]).sort_values('작업일').reset_index(drop=True)
    combined_df['Time Index'] = range(len(combined_df))

    plt.figure(figsize=(16, 4))
    plt.plot(combined_df['Time Index'], combined_df['Actual'], label='Actual', linewidth=2)
    plt.plot(combined_df['Time Index'], combined_df['Predicted'], label='Predicted', linestyle='--', linewidth=2)
    test_start = combined_df[combined_df['Set'] == 'Test']['Time Index'].min()
    plt.axvline(x=test_start, color='red', linestyle=':', label='Test Start')
    plt.xlabel('Time Index (chronological)')
    plt.ylabel(target)
    plt.title(f'Actual vs Predicted {target} ({name}) - R² = {r2_test:.3f}, RMSE = {rmse_test:.1f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    test_df = test_df.sort_values('날짜').reset_index(drop=True)
    test_df['Time Index'] = range(len(test_df))

    plt.figure(figsize=(18, 3))
    plt.plot(test_df['Time Index'], test_df['Actual'], label='Actual', linewidth=2)
    plt.plot(test_df['Time Index'], test_df['Predicted'], label='Predicted', linestyle='--', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Train and Evaluate ---
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
evaluate_model('XGBoost', model, X_train, y_train, X_test, y_test, repair_df)
