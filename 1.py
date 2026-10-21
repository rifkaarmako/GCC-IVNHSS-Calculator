import streamlit as st
import numpy as np
import pandas as pd

# Persamaan (3.1): Energi Informasional Diperumum
def calc_informational_energy(matrix, theta):
    return np.sum(matrix ** theta)

# Persamaan (3.2): Ukuran Korelasi Diperumum
def calc_correlation_measure(matrix1, matrix2, theta):
    return np.sum((matrix1 ** (theta / 2.0)) * (matrix2 ** (theta / 2.0)))

# Persamaan (3.3): Koefisien Korelasi Diperumum
def calc_correlation_coefficient(matrix1, matrix2, theta):
    corr_measure = calc_correlation_measure(matrix1, matrix2, theta)
    energy_1 = calc_informational_energy(matrix1, theta)
    energy_2 = calc_informational_energy(matrix2, theta)
    
    if energy_1 == 0 or energy_2 == 0:
        return 0.0
    
    return corr_measure / (np.sqrt(energy_1) * np.sqrt(energy_2))

st.set_page_config(page_title="Kalkulator IVNHSS", layout="wide")
st.title("Kalkulator Generalized Correlation Coefficient of IVNHSS")

# Pengaturan Variabel Utama
st.sidebar.header("Pengaturan Parameter")
theta = st.sidebar.number_input("Nilai Theta (θ)", min_value=0.1, value=1.0, step=0.1)
k_objects = st.sidebar.number_input("Jumlah Objek Alternatif (k)", min_value=1, value=2, step=1)
m_params = st.sidebar.number_input("Jumlah Kriteria (m)", min_value=1, value=3, step=1)

# Fungsi untuk membuat kerangka tabel kosong dengan 6 komponen IVNHSS
def create_default_df(m):
    return pd.DataFrame({
        "Kriteria": [f"e_{i+1}" for i in range(m)],
        "T_L": [0.0]*m, "T_U": [0.0]*m,
        "I_L": [0.0]*m, "I_U": [0.0]*m,
        "F_L": [0.0]*m, "F_U": [0.0]*m,
    })

# Bagian UI untuk Input Data
st.subheader("1. Input Data IVNHSS Objek Ideal")
st.write("Ketikkan nilai interval ekspektasi batas bawah (L) dan batas atas (U) pada tabel berikut:")
# st.data_editor merender tabel yang bisa diketik langsung oleh user
df_ideal = st.data_editor(create_default_df(m_params), key="ideal", hide_index=True, use_container_width=True)

st.subheader("2. Input Data IVNHSS Objek Alternatif")
objects_data = []
for i in range(k_objects):
    st.write(f"**Objek $u_{i+1}$**")
    df_obj = st.data_editor(create_default_df(m_params), key=f"obj_{i}", hide_index=True, use_container_width=True)
    objects_data.append(df_obj)

# Eksekusi Algoritma (Langkah 3 hingga 6)
if st.button("Hitung dan Peringkatkan"):
    # Mengambil nilai angka dari tabel (membuang kolom nama 'Kriteria')
    ideal_matrix = df_ideal.drop(columns=["Kriteria"]).to_numpy(dtype=float)
    
    results = []
    for i, df_obj in enumerate(objects_data):
        obj_matrix = df_obj.drop(columns=["Kriteria"]).to_numpy(dtype=float)
        
        # Kalkulasi berdasarkan rumus
        energy = calc_informational_energy(obj_matrix, theta)
        corr_measure = calc_correlation_measure(obj_matrix, ideal_matrix, theta)
        corr_coeff = calc_correlation_coefficient(obj_matrix, ideal_matrix, theta)
        
        results.append({
            "Objek": f"u_{i+1}",
            "Energi Informasional (ζ)": energy,
            "Ukuran Korelasi (C)": corr_measure,
            "Koefisien Korelasi (δ)": corr_coeff
        })
        
    # Memproses hasil akhir dan mengurutkan dari yang terbesar
    df_results = pd.DataFrame(results).sort_values(by="Koefisien Korelasi (δ)", ascending=False).reset_index(drop=True)
    df_results.index += 1
    
    st.markdown("---")
    st.subheader("Hasil Perhitungan dan Analisis Peringkat")
    st.success(f"Berdasarkan nilai koefisien korelasi maksimum, objek terbaik adalah **{df_results.iloc[0]['Objek']}**.")
    st.table(df_results)
