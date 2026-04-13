import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =============================
# SET SEED (BIAR HASIL STABIL)
# =============================
np.random.seed(42)
tf.random.set_seed(42)

# =============================
# Baca CSV (AUTO HANDLE ERROR)
# =============================
try:
    df = pd.read_csv("jumlah_lulus.csv")  # coba default (koma)
    
    # kalau cuma 1 kolom → kemungkinan separator salah
    if len(df.columns) == 1:
        df = pd.read_csv("jumlah_lulus.csv", sep=';')

except Exception as e:
    raise ValueError(f"Gagal membaca CSV: {e}")

# =============================
# Validasi kolom
# =============================
print("Jumlah kolom:", len(df.columns))
print("Nama kolom:", df.columns)

if len(df.columns) < 2:
    raise ValueError("CSV harus memiliki minimal 2 kolom!")

# Ambil hanya 2 kolom pertama
df = df.iloc[:, :2]

# Ambil nama kolom asli
kolom_x = df.columns[0]
kolom_y = df.columns[1]

print(df.head())

# =============================
# Visualisasi awal
# =============================
plt.figure(figsize=(8,5))
sns.scatterplot(x=df[kolom_x], y=df[kolom_y], color="blue", label="Data Aktual")
plt.xlabel(kolom_x)
plt.ylabel(kolom_y)
plt.title(f"Pertumbuhan {kolom_y}")
plt.legend()
plt.show()

# =============================
# Preprocessing (SAMA MODUL)
# =============================
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[[kolom_x, kolom_y]])

X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =============================
# Model ANN
# =============================
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(
    X_train, Y_train,
    epochs=200,
    validation_data=(X_test, Y_test),
    verbose=1
)

# =============================
# Evaluasi
# =============================
loss, mae = model.evaluate(X_test, Y_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# =============================
# Prediksi (DATA BARU)
# =============================
nilai_baru = np.array([
    [df[kolom_x].max() + 1],
    [df[kolom_x].max() + 5]
])

nilai_scaled = scaler.transform(
    np.column_stack((nilai_baru, np.zeros(len(nilai_baru))))
)[:, 0].reshape(-1, 1)

prediksi_scaled = model.predict(nilai_scaled)

prediksi = scaler.inverse_transform(
    np.column_stack((nilai_scaled[:, 0], prediksi_scaled))
)[:, 1]

# Output hasil prediksi
for x, y in zip(nilai_baru.flatten(), prediksi):
    print(f"Prediksi {kolom_y} pada {kolom_x} {int(x)}: {int(y)}")

# =============================
# Visualisasi hasil
# =============================
Y_pred = model.predict(X_test)

plt.figure(figsize=(8,5))
plt.scatter(X_test, Y_test, color='blue', label="Data Aktual")
plt.scatter(X_test, Y_pred, color='red', label="Prediksi ANN")

plt.xlabel(f"{kolom_x} (scaled)")
plt.ylabel(f"{kolom_y} (scaled)")
plt.title("Hasil Prediksi ANN vs Data Aktual")
plt.legend()
plt.show()