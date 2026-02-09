import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Pengaturan MLflow ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Telco Churn Prediction Basic")

# --- Memuat Data ---
DATA_PATH = 'telco_churn_preprocessing.csv'

print("Memulai proses training model...")
try:
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset berhasil dimuat dari '{DATA_PATH}'.")
except FileNotFoundError:
    print(f"Error: File '{DATA_PATH}' tidak ditemukan. Pastikan path sudah benar.")
    exit()

# --- Persiapan Data ---
# Pisahkan fitur (X) dan target (y)
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data berhasil dibagi menjadi data latih dan uji.")

# --- Autologging MLflow ---
# Mengaktifkan autologging untuk library Scikit-learn
mlflow.sklearn.autolog()

# --- Sesi Pelatihan ---
with mlflow.start_run(run_name="telco-logistic-regression"):
    # Inisialisasi dan latih model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("Model berhasil dilatih.")

    # Lakukan prediksi pada data uji
    y_pred = model.predict(X_test)

    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"\nTraining selesai. Run 'telco-logistic-regression' dicatat.")
    print(f"   Akurasi: {accuracy:.4f}")