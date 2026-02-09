import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Memulai proses hyperparameter tuning model...")

# --- Muat dataset yang sudah diproses --- 
try:
    df = pd.read_csv('telco_churn_preprocessing.csv')
    print("Dataset berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'telco_churn_preprocessing.csv' tidak ditemukan.")
    exit()

# --- Persiapan Data --- 
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data berhasil dibagi menjadi data latih dan uji.")

# --- Definisikan model dan parameter grid untuk tuning --- 
model = LogisticRegression(max_iter=1000, random_state=42)
param_grid = {
    'C': [0.1, 1, 10],  # Mencoba beberapa nilai regularisasi
    'solver': ['liblinear', 'saga'] # Mencoba solver yang berbeda
}

# --- Inisialisasi GridSearchCV --- 
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# --- Atur eksperimen MLflow --- 
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Telco Churn - Hyperparameter Tuning")

# --- Mulai sesi MLflow run --- 
with mlflow.start_run(run_name="logistic-regression-tuning"):
    print("Memulai GridSearchCV...")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV selesai.")

    # Ambil model dan parameter terbaik
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Parameter terbaik ditemukan: {best_params}")

    # MANUAL LOGGING: Mencatat parameter terbaik secara manual
    print("Mencatat parameter terbaik ke MLflow...")
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)

    # Lakukan prediksi pada data uji dengan model terbaik
    y_pred = best_model.predict(X_test)

    # Evaluasi model terbaik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # MANUAL LOGGING: Mencatat metrik evaluasi secara manual
    print("Mencatat metrik evaluasi ke MLflow...")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # MANUAL LOGGING: Mencatat model terbaik sebagai artefak
    print("Mencatat model terbaik sebagai artefak MLflow...")
    mlflow.sklearn.log_model(best_model, "model")

    print(f"\nTraining dan tuning selesai.")
    print(f"Akurasi Model Terbaik: {accuracy:.4f}")
    print(f"F1 Score Model Terbaik: {f1:.4f}")
    print("\nHasil eksperimen telah dicatat secara manual di MLflow.")
