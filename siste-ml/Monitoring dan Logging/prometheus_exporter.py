from flask import Flask, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# 1. Total Permintaan
REQUESTS_TOTAL = Counter(
    'prediction_requests_total', 'Total number of prediction requests.'
)

# 2. Total Hasil Prediksi
PREDICTION_RESULTS_TOTAL = Counter(
    'prediction_results_total', 'Total predictions by result.', ['result']
)

# 3. Total Permintaan Gagal
INVALID_REQUESTS_TOTAL = Counter(
    'invalid_requests_total', 'Total invalid requests.'
)

# 4. Latensi Prediksi (Performa Sistem)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 'Model prediction latency in seconds.'
)

# 5. Rata-rata Tagihan Bulanan (Data Drift)
AVG_MONTHLY_CHARGES = Gauge(
    'average_monthly_charges', 'Average monthly charges of incoming requests.'
)

# 6. Jumlah Berdasarkan Tipe Kontrak (Data Drift)
CONTRACT_TYPE_COUNTER = Counter(
    'contract_type_total', 'Count of requests by contract type.', ['type']
)

# 7. Jumlah Berdasarkan Layanan Internet (Data Drift)
INTERNET_SERVICE_COUNTER = Counter(
    'internet_service_total', 'Count of requests by internet service type.', ['service']
)

# Endpoint ini akan dipanggil oleh script inference.py
@app.route('/update', methods=['POST'])
def update_metrics():
    data = request.get_json()
    if not data:
        INVALID_REQUESTS_TOTAL.inc()
        return "Invalid request", 400

    # Tingkatkan nilai metrik berdasarkan data dari inference.py
    REQUESTS_TOTAL.inc()

    # Data dari prediksi yang sukses
    PREDICTION_RESULTS_TOTAL.labels(result=data['prediction']).inc()
    PREDICTION_LATENCY.observe(data['latency'])
    AVG_MONTHLY_CHARGES.set(data['avg_monthly_charges'])
    CONTRACT_TYPE_COUNTER.labels(type=data['contract']).inc()
    INTERNET_SERVICE_COUNTER.labels(service=data['internet_service']).inc()

    return "Metrics updated successfully", 200

if __name__ == '__main__':
    app.run(port=8001, host='0.0.0.0')