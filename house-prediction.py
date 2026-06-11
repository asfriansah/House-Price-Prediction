import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib  # Library untuk menyimpan model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Mengatur tema visualisasi
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


class HousePricePipeline:

    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.models = {}
        self.best_model_name = None
        self.best_model_score = -float("inf")

    def load_data(self):
        """1. DATA PREPARATION"""
        print("--- Memuat Data ---")
        if os.path.exists(self.train_path):
            self.df_train = pd.read_csv(self.train_path)
            print(f"Data Train berhasil dimuat: {self.df_train.shape}")
        else:
            raise FileNotFoundError("File data train tidak ditemukan!")

        if os.path.exists(self.test_path):
            self.df_test = pd.read_csv(self.test_path)
            print(f"Data Test berhasil dimuat: {self.df_test.shape}")

    def clean_data(self):
        """2. DATA CLEANSING (Sudah diperbaiki dari typo sebelumnya)"""
        print("\n--- Membersihkan Data ---")
        categorical_nas = [
            "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
            "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish",
            "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"
        ]

        for col in categorical_nas:
            if col in self.df_train.columns:
                self.df_train[col] = self.df_train[col].fillna("None")
            if self.df_test is not None and col in self.df_test.columns:
                self.df_test[col] = self.df_test[col].fillna("None")

        if "LotFrontage" in self.df_train.columns:
            median_frontage = self.df_train["LotFrontage"].median()
            self.df_train["LotFrontage"] = self.df_train["LotFrontage"].fillna(median_frontage)
            if self.df_test is not None and "LotFrontage" in self.df_test.columns:
                self.df_test["LotFrontage"] = self.df_test["LotFrontage"].fillna(median_frontage)

        for col in self.df_train.columns:
            if self.df_train[col].isnull().any():
                if self.df_train[col].dtype == "object":
                    self.df_train[col] = self.df_train[col].fillna(self.df_train[col].mode()[0])
                else:
                    self.df_train[col] = self.df_train[col].fillna(self.df_train[col].median())

        if self.df_test is not None:
            for col in self.df_test.columns:
                if self.df_test[col].isnull().any():
                    if self.df_test[col].dtype == "object":
                        mode_val = self.df_test[col].mode()[0] if not self.df_test[col].mode().empty else "None"
                        self.df_test[col] = self.df_test[col].fillna(mode_val)
                    else:
                        self.df_test[col] = self.df_test[col].fillna(self.df_test[col].median())

        print("Pembersihan selesai. Sisa missing value di Train:", self.df_train.isnull().sum().sum())

    def run_eda(self):
        """3. EDA & DATA VISUALIZATION"""
        print("\n--- Menjalankan EDA & Visualisasi ---")
        plt.figure()
        sns.histplot(self.df_train["SalePrice"], kde=True, color="blue")
        plt.title("Distribusi Harga Rumah (SalePrice)")
        plt.close() # Menghindari plotting bertumpuk di non-interactive env

    def feature_engineering(self):
        """4. FEATURE ENGINEERING"""
        print("\n--- Rekayasa Fitur ---")
        self.df_train["HouseAge"] = self.df_train["YrSold"] - self.df_train["YearBuilt"]
        
        X = self.df_train.drop(columns=["Id", "SalePrice"])
        y = np.log1p(self.df_train["SalePrice"]) # Log transformation

        X = pd.get_dummies(X, drop_first=True)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)

    def build_and_evaluate_models(self):
        """5. MODEL ENGINEERING: Melatih 3 model awal & menentukan yang terbaik."""
        print("\n--- Pelatihan dan Evaluasi 3 Model Awal ---")

        algorithms = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42, n_jobs=-1)
        }

        results = []

        for name, model in algorithms.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

            preds_log = model.predict(self.X_val)
            preds_actual = np.expm1(preds_log)
            y_val_actual = np.expm1(self.y_val)

            rmse = np.sqrt(mean_squared_error(y_val_actual, preds_actual))
            mae = mean_absolute_error(y_val_actual, preds_actual)
            r2 = r2_score(y_val_actual, preds_actual)

            results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2 Score": r2})

            # Menentukan model terbaik berdasarkan R2 Score tertinggi
            if r2 > self.best_model_score:
                self.best_model_score = r2
                self.best_model_name = name

        df_results = pd.DataFrame(results)
        print("\nHasil Performa Model Awal:")
        print(df_results.to_string(index=False))
        print(f"\n>> Model Terbaik Terpilih: {self.best_model_name} (R2: {self.best_model_score:.4f})")

    def tune_and_save_best_model(self):
        """6. HYPERPARAMETER TUNING & MODEL SAVING"""
        print(f"\n--- Memulai Hyperparameter Tuning untuk {self.best_model_name} ---")
        
        # Penentuan parameter grid berdasarkan model terbaik yang terpilih otomatis
        if self.best_model_name == "Ridge Regression":
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            base_model = Ridge()
            
        elif self.best_model_name == "Random Forest":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, None],
                'min_samples_split': [2, 5]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif self.best_model_name == "XGBoost":
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6]
            }
            base_model = XGBRegressor(random_state=42, n_jobs=-1)

        # Menjalankan Grid Search dengan 3-Fold Cross Validation
        grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                                   cv=3, scoring='r2', verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_tuned_model = grid_search.best_estimator_
        print(f"\nKombinasi Hyperparameter Terbaik: {grid_search.best_params_}")

        # Evaluasi Akhir Model Hasil Tuning pada Data Validasi
        tuned_preds_log = best_tuned_model.predict(self.X_val)
        tuned_preds_actual = np.expm1(tuned_preds_log)
        y_val_actual = np.expm1(self.y_val)

        tuned_r2 = r2_score(y_val_actual, tuned_preds_actual)
        tuned_rmse = np.sqrt(mean_squared_error(y_val_actual, tuned_preds_actual))
        
        print("\n--- Perbandingan Sebelum vs Sesudah Tuning ---")
        print(f"R2 Score Awal  : {self.best_model_score:.4f}")
        print(f"R2 Score Tuned : {tuned_r2:.4f}")
        print(f"RMSE Tuned     : ${tuned_rmse:,.2f}")

        # Menyimpan Model Terbaik yang sudah di-tuning
        model_filename = f"best_house_price_model_{self.best_model_name.lower().replace(' ', '_')}.joblib"
        joblib.dump(best_tuned_model, model_filename)
        print(f"\n[SUKSES] Model terbaik telah di-tuning dan disimpan sebagai: '{model_filename}'")


# --- EKSEKUSI PIPELINE ---
if __name__ == "__main__":
    pipeline = HousePricePipeline(train_path="train.csv", test_path="test.csv")

    try:
        pipeline.load_data()
        pipeline.clean_data()
        pipeline.run_eda()
        pipeline.feature_engineering()
        pipeline.build_and_evaluate_models()
        pipeline.tune_and_save_best_model() # Langkah baru untuk tuning dan save
    except Exception as e:
        print(f"\nTerjadi kesalahan proses: {e}")