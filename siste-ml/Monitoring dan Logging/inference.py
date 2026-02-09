import requests
import json
import pandas as pd
import time
import random

# URL Server
MLFLOW_URL = 'http://127.0.0.1:5001/invocations'
EXPORTER_URL = 'http://127.0.0.1:8001/update'
headers = {'Content-Type': 'application/json'}

# Muat data mentah dan data yang sudah diproses
try:
    # Data mentah dibutuhkan untuk memonitor data drift pada fitur asli
    df_raw = pd.read_csv('Telco-Customer-Churn_raw.csv')
    df_processed = pd.read_csv('telco_churn_preprocessing.csv')
    X_test = df_processed.drop('Churn_Yes', axis=1)
except FileNotFoundError:
    print("Error: Pastikan folder 'Telco-Customer-Churn_raw.csv' dan file 'telco_churn_preprocessing.csv' ada di direktori Anda.")
    exit()

# Variabel untuk menghitung rata-rata bergerak (moving average)
monthly_charges_history = []
AVG_WINDOW = 50 # Hitung rata-rata dari 50 permintaan terakhir

print("Memulai pengiriman data inferensi untuk monitoring advance...")
while True:
    try:
        # Ambil sampel acak dari data
        random_index = random.randint(0, len(X_test) - 1)
        sample_processed = X_test.iloc[[random_index]]
        sample_raw = df_raw.iloc[[random_index]]

        inference_data = sample_processed.to_dict(orient='split')
        if 'index' in inference_data:
            del inference_data['index']
        
        payload = {"dataframe_split": inference_data}
        
        # Ukur latensi saat mengirim permintaan ke model
        start_time = time.time()
        response = requests.post(MLFLOW_URL, data=json.dumps(payload), headers=headers)
        latency = time.time() - start_time
        
        # Siapkan payload metrik untuk dikirim ke exporter
        metrics_payload = {"latency": latency}

        if response.status_code == 200:
            prediction_val = response.json()['predictions'][0]
            result = "Churn" if prediction_val == 1 else "No Churn"
            print(f"Hasil Prediksi: {result} (Latency: {latency:.4f}s)")
            
            # Tambahkan semua data lain yang dibutuhkan oleh exporter
            metrics_payload["prediction"] = result
            metrics_payload["contract"] = sample_raw["Contract"].iloc[0]
            metrics_payload["internet_service"] = sample_raw["InternetService"].iloc[0]
            
            charge = float(sample_raw["MonthlyCharges"].iloc[0])
            monthly_charges_history.append(charge)
            if len(monthly_charges_history) > AVG_WINDOW:
                monthly_charges_history.pop(0)
            avg_charges = sum(monthly_charges_history) / len(monthly_charges_history)
            metrics_payload["avg_monthly_charges"] = avg_charges
            
        else:
            print(f"Permintaan prediksi MLflow gagal: {response.status_code}")
            metrics_payload["prediction_error"] = True

        # Kirim payload metrik lengkap ke exporter
        requests.post(EXPORTER_URL, json=metrics_payload)
            
    except requests.exceptions.ConnectionError:
        print("Tidak dapat terhubung ke server. Pastikan semua berjalan.")
        requests.post(EXPORTER_URL, json={"prediction_error": True})
    except Exception as e:
        print(f"Terjadi error: {e}")
        
    time.sleep(3)