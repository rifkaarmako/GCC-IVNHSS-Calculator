import streamlit as st
import numpy as np
import pandas as pd

# Persamaan (3.1): Energi Informasional Diperumum
# Karena input berbentuk matriks 3D (penilai, kriteria, 6 komponen), np.sum akan otomatis 
# menjumlahkan seluruh elemen sesuai dengan dua notasi sigma (untuk j=1 sampai n, dan i=1 sampai m)
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
k_objects = st.sidebar.number_input("Jumlah Objek (k)", min_value=1, value=2, step=1)
n_evaluators = st.sidebar.number_input("Jumlah Penilai (n)", min_value=1, value=2, step=1)
m_params = st.sidebar.number_input("Jumlah Kriteria (m)", min_value=1, value=3, step=1)

# Fungsi membuat kerangka tabel
def create_default_df(m):
    return pd.DataFrame({
        "Kriteria": [f"e_{i+1}" for i in range(m)],
        "T_L": [0.0]*m, "T_U": [0.0]*m,
        "I_L": [0.0]*m, "I_U": [0.0]*m,
        "F_L": [0.0]*m, "F_U": [0.0]*m,
    })

# 1. Input Data Objek Ideal
st.subheader("1. Input Data IVNHSS Objek Ideal")
ideal_data = []
tabs_ideal = st.tabs([f"Penilai {j+1}" for j in range(n_evaluators)])

for j, tab in enumerate(tabs_ideal):
    with tab:
        df_ideal = st.data_editor(create_default_df(m_params), key=f"ideal_eval_{j}", hide_index=True, use_container_width=True)
        ideal_data.append(df_ideal)

st.markdown("---")

# 2. Input Data Objek Alternatif
st.subheader("2. Input Data IVNHSS Objek Alternatif")
objects_data = [] # Menyimpan data seluruh objek dari seluruh penilai

for i in range(k_objects):
    st.write(f"**Data Objek $u_{i+1}$**")
    tabs_obj = st.tabs([f"Penilai {j+1}" for j in range(n_evaluators)])
    
    obj_eval_data = []
    for j, tab in enumerate(tabs_obj):
        with tab:
            df_obj = st.data_editor(create_default_df(m_params), key=f"obj_{i}_eval_{j}", hide_index=True, use_container_width=True)
            obj_eval_data.append(df_obj)
    
    objects_data.append(obj_eval_data)
    st.write("") # spasi

# Eksekusi Algoritma (Langkah 3 hingga 6)
if st.button("Hitung dan Lihat Peringkat", type="primary"):
    # Mengonversi list of DataFrames menjadi matriks NumPy 3D: (n_evaluators, m_params, 6)
    ideal_matrix = np.array([df.drop(columns=["Kriteria"]).to_numpy(dtype=float) for df in ideal_data])
    
    results = []
    for i, obj_eval_data in enumerate(objects_data):
        obj_matrix = np.array([df.drop(columns=["Kriteria"]).to_numpy(dtype=float) for df in obj_eval_data])
        
        # Kalkulasi
        energy = calc_informational_energy(obj_matrix, theta)
        corr_measure = calc_correlation_measure(obj_matrix, ideal_matrix, theta)
        corr_coeff = calc_correlation_coefficient(obj_matrix, ideal_matrix, theta)
        
        results.append({
            "Objek": f"u_{i+1}",
            "Energi Informasional (ζ)": energy,
            "Ukuran Korelasi (C)": corr_measure,
            "Koefisien Korelasi (δ)": corr_coeff
        })
        
    # Peringkat
    df_results = pd.DataFrame(results).sort_values(by="Koefisien Korelasi (δ)", ascending=False).reset_index(drop=True)
    df_results.index += 1
    
    st.markdown("---")
    st.subheader("Hasil Perhitungan dan Analisis Peringkat")
    st.success(f"Berdasarkan nilai koefisien korelasi maksimum, objek terbaik adalah **{df_results.iloc[0]['Objek']}**.")
    st.table(df_results)
