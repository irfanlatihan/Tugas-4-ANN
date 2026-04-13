import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # WAJIB untuk Flask (no GUI)
import matplotlib.pyplot as plt
import io
import base64

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

# =============================
# LOAD & TRAIN MODEL (SEKALI SAJA)
# =============================
np.random.seed(42)
tf.random.set_seed(42)

# Load CSV (auto separator)
df = pd.read_csv("jumlah_lulus.csv")
if len(df.columns) == 1:
    df = pd.read_csv("jumlah_lulus.csv", sep=';')

df = df.iloc[:, :2]
kolom_x = df.columns[0]
kolom_y = df.columns[1]

# Preprocessing
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[[kolom_x, kolom_y]])

X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Model ANN
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, Y_train, epochs=200, verbose=0)

# =============================
# FUNCTION PREDIKSI
# =============================
def prediksi_tahun(tahun_input):
    nilai = np.array([[tahun_input]])

    nilai_scaled = scaler.transform(
        np.column_stack((nilai, np.zeros(len(nilai))))
    )[:, 0].reshape(-1, 1)

    pred_scaled = model.predict(nilai_scaled)
    hasil = scaler.inverse_transform(
        np.column_stack((nilai_scaled[:, 0], pred_scaled))
    )[:, 1]

    return int(hasil[0])

# =============================
# FUNCTION GRAFIK
# =============================
def generate_plot(tahun_input, hasil_prediksi):
    plt.figure(figsize=(6,4))

    # Data asli
    plt.scatter(df[kolom_x], df[kolom_y], label="Data Aktual")

    # Prediksi ANN pada test
    Y_pred = model.predict(X)
    plt.scatter(df[kolom_x], scaler.inverse_transform(
        np.column_stack((X.flatten(), Y_pred))
    )[:,1], label="Prediksi ANN")

    # Titik prediksi baru
    plt.scatter(tahun_input, hasil_prediksi, color='red', s=100, marker='x', label="Prediksi Baru")

    plt.xlabel(kolom_x)
    plt.ylabel(kolom_y)
    plt.title("Grafik Prediksi ANN")
    plt.legend()

    # Convert ke base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return img

# =============================
# ROUTE
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    grafik = None

    if request.method == "POST":
        tahun = int(request.form["tahun"])
        hasil = prediksi_tahun(tahun)
        grafik = generate_plot(tahun, hasil)

    return render_template("index.html", hasil=hasil, grafik=grafik, kolom_x=kolom_x, kolom_y=kolom_y)

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(debug=True)