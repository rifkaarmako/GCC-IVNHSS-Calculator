import numpy as np
import pandas as pd
import streamlit as st


def _safe_pow(arr, exp):
    """Menghitung pangkat secara aman untuk nilai non-negatif."""
    arr = np.asarray(arr, dtype=float)
    return np.power(np.clip(arr, 0.0, None), exp)


def calc_weighted_energy(matrix_n_x_m_x_6, alpha, beta, theta):
    """
    Persamaan (3.1.1):
    zeta^theta(psi, Lambda) = sum_{j} beta_j * sum_{i=1}^{m} alpha_i * sum_{6 komponen} (x_ij)^theta
    matrix shape: (num_outer, m, 6)
    alpha shape: (m,)
    beta shape: (num_outer,)
    """
    powered = _safe_pow(matrix_n_x_m_x_6, theta)          # (num_outer, m, 6)
    sum_components = np.sum(powered, axis=-1)               # (num_outer, m)
    weighted_criteria = np.sum(sum_components * alpha[None, :], axis=-1)  # (num_outer,)
    return float(np.sum(weighted_criteria * beta))


def calc_weighted_corr_measure(mat1, mat2, alpha, beta, theta):
    """
    Persamaan (3.1.2):
    C^theta((psi_1, Lambda_1), (psi_2, Lambda_2))
    mat1, mat2 shape: (num_outer, m, 6)
    """
    half_theta = theta / 2.0
    p1 = _safe_pow(mat1, half_theta)
    p2 = _safe_pow(mat2, half_theta)
    prod_comp = np.sum(p1 * p2, axis=-1)                    # (num_outer, m)
    weighted_criteria = np.sum(prod_comp * alpha[None, :], axis=-1)       # (num_outer,)
    return float(np.sum(weighted_criteria * beta))


def calc_weighted_corr_coeff(mat1, mat2, alpha, beta, theta):
    """
    Persamaan (3.1.3):
    delta^theta = C^theta / (sqrt(zeta_1) * sqrt(zeta_2))
    """
    c_val = calc_weighted_corr_measure(mat1, mat2, alpha, beta, theta)
    e1 = calc_weighted_energy(mat1, alpha, beta, theta)
    e2 = calc_weighted_energy(mat2, alpha, beta, theta)

    denom = np.sqrt(max(e1, 0.0)) * np.sqrt(max(e2, 0.0))
    if denom == 0.0:
        return 0.0, c_val, e1, e2
    return float(c_val / denom), c_val, e1, e2


def calc_ivns_entropy_per_criterion(eval_matrix_k_x_m_x_6):
    """
    Langkah 4 & Persamaan (2.3.2):
    Untuk satu penilai r, terdapat m kriteria; tiap kriteria i adalah IVNS atas U = {u_1, ..., u_k}.
    eval_matrix_k_x_m_x_6 shape: (k, m, 6) -> kolom: [T^L, T^U, I^L, I^U, F^L, F^U]
    Mengembalikan array entropi E_r(A_i) berukuran (m,).
    """
    t_l = eval_matrix_k_x_m_x_6[:, :, 0]
    t_u = eval_matrix_k_x_m_x_6[:, :, 1]
    i_l = eval_matrix_k_x_m_x_6[:, :, 2]
    i_u = eval_matrix_k_x_m_x_6[:, :, 3]
    f_l = eval_matrix_k_x_m_x_6[:, :, 4]
    f_u = eval_matrix_k_x_m_x_6[:, :, 5]

    tf_l = np.abs(t_l - f_l)
    tf_u = np.abs(t_u - f_u)

    # (I^c)^L = 1 - I^U, (I^c)^U = 1 - I^L
    # |I^L - (1 - I^U)| = |I^L + I^U - 1|
    ii_c_l = np.abs(i_l + i_u - 1.0)
    ii_c_u = np.abs(i_u + i_l - 1.0)

    num = 4.0 - (tf_l + tf_u) - (ii_c_l + ii_c_u)
    den = 4.0 + (tf_l + tf_u) + (ii_c_l + ii_c_u)

    # Rata-rata atas k objek (axis=0) -> hasil berukuran (m,)
    return np.mean(num / den, axis=0)


st.set_page_config(page_title="Kalkulator IVNHSS Berbobot", layout="wide")
st.title("Kalkulator Koefisien Korelasi Diperumum Berbobot pada IVNHSS")
st.caption(
    "Perhitungan Bobot Penilai (Efisiensi Korelasi Ternormalisasi), Bobot Kriteria (Entropi & Divergensi), "
    "serta Peringkat Objek Alternatif pada *Interval-Valued Neutrosophic Hypersoft Set*"
)
st.caption("Penjelasan selengkapnya dapat dilihat pada artikel pada tautan berikut: (*link artikel setelah terbit*)")

st.sidebar.header("Input Awal")
theta = st.sidebar.number_input("Nilai Theta (θ)", value=1.0, step=0.1)
k_objects = int(st.sidebar.number_input("Jumlah Objek Alternatif (k)", min_value=1, value=2, step=1))
n_evaluators = int(st.sidebar.number_input("Jumlah Pengambil Keputusan (n)", min_value=1, value=2, step=1))
m_params = int(st.sidebar.number_input("Jumlah Pasangan Kriteria (m)", min_value=1, value=3, step=1))


def create_default_df(m):
    return pd.DataFrame({
        "Pasangan Kriteria": [f"A_{i+1}" for i in range(m)],
        "T^L": [0.0] * m, "T^U": [0.0] * m,
        "I^L": [0.0] * m, "I^U": [0.0] * m,
        "F^L": [0.0] * m, "F^U": [0.0] * m,
    })


st.subheader("1. Input Data Objek Ideal IVNHSS (ψ̂, Λ)")
ideal_data = []
tabs_ideal = st.tabs([f"Pengambil Keputusan P_{r+1}" for r in range(n_evaluators)])

for r, tab in enumerate(tabs_ideal):
    with tab:
        df_ideal = st.data_editor(
            create_default_df(m_params),
            key=f"ideal_eval_{r}",
            hide_index=True,
            use_container_width=True
        )
        ideal_data.append(df_ideal)

st.markdown("---")

st.subheader("2. Input Data Objek Alternatif IVNHSS")
objects_data = []

for j in range(k_objects):
    st.write(f"**Data Objek Alternatif $u_{j+1}$**")
    tabs_obj = st.tabs([f"Pengambil Keputusan P_{r+1}" for r in range(n_evaluators)])

    obj_eval_data = []
    for r, tab in enumerate(tabs_obj):
        with tab:
            df_obj = st.data_editor(
                create_default_df(m_params),
                key=f"obj_{j}_eval_{r}",
                hide_index=True,
                use_container_width=True
            )
            obj_eval_data.append(df_obj)

    objects_data.append(obj_eval_data)
    st.write("")

if st.button("Hitung dan Tampilkan Peringkat", type="primary"):
    # Susun raw_matrix dengan shape: (n_evaluators, k_objects, m_params, 6)
    raw_matrix = np.array([
        [
            objects_data[j][r].drop(columns=["Pasangan Kriteria"]).to_numpy(dtype=float)
            for j in range(k_objects)
        ]
        for r in range(n_evaluators)
    ])

    # Ideal matrix dengan shape: (n_evaluators, m_params, 6)
    ideal_matrix = np.array([
        ideal_data[r].drop(columns=["Pasangan Kriteria"]).to_numpy(dtype=float)
        for r in range(n_evaluators)
    ])

    # =========================================================================
    # LANGKAH 2 & 3: Koefisien Korelasi antar (psi_r, Lambda), Efisiensi & Bobot beta_r
    # =========================================================================
    alpha_init = np.full(m_params, 1.0 / m_params)
    beta_init = np.full(k_objects, 1.0 / k_objects)

    delta_rs = np.zeros((n_evaluators, n_evaluators))
    for r in range(n_evaluators):
        for s in range(n_evaluators):
            delta_val, _, _, _ = calc_weighted_corr_coeff(
                raw_matrix[r], raw_matrix[s], alpha_init, beta_init, theta
            )
            delta_rs[r, s] = delta_val

    if n_evaluators > 1:
        gamma_r = np.array([
            np.sum([delta_rs[r, l] for l in range(n_evaluators) if l != r]) / (n_evaluators - 1)
            for r in range(n_evaluators)
        ])
    else:
        gamma_r = np.ones(n_evaluators)

    sum_gamma = np.sum(gamma_r)
    if sum_gamma != 0.0:
        beta_weights = gamma_r / sum_gamma
    else:
        beta_weights = np.full(n_evaluators, 1.0 / n_evaluators)

    # =========================================================================
    # LANGKAH 4: Entropi IVNS, Entropi Kumulatif, Divergensi & Bobot Kriteria alpha_i
    # =========================================================================
    entropy_r_i = np.array([
        calc_ivns_entropy_per_criterion(raw_matrix[r])  # shape: (m_params,)
        for r in range(n_evaluators)
    ])  # shape: (n_evaluators, m_params)

    # Persamaan (3.2.1): E_bar(A_i) = sum_{r=1}^n beta_r * E_r(A_i)
    cum_entropy_i = np.sum(beta_weights[:, None] * entropy_r_i, axis=0)

    # Persamaan (3.2.2): DV_i = 1 - E_bar(A_i)
    divergence_i = 1.0 - cum_entropy_i

    # Persamaan (3.2.3): alpha_i = DV_i / sum_{l=1}^m DV_l
    sum_dv = np.sum(divergence_i)
    if sum_dv != 0.0:
        alpha_weights = divergence_i / sum_dv
    else:
        alpha_weights = np.full(m_params, 1.0 / m_params)

    # =========================================================================
    # LANGKAH 5, 6 & 7: Konstruksi (psi_bar_j, Lambda), Korelasi ke Objek Ideal & Peringkat
    # =========================================================================
    # Untuk objek u_j, psi_bar_j atas P memiliki penilaian dari P_1..P_n pada kriteria A_1..A_m
    # yaitu raw_matrix[:, j, :, :] -> shape: (n_evaluators, m_params, 6)
    results = []
    for j in range(k_objects):
        psi_bar_j = raw_matrix[:, j, :, :]  # shape: (n, m, 6)
        corr_coeff, corr_measure, energy_obj, _ = calc_weighted_corr_coeff(
            psi_bar_j, ideal_matrix, alpha_weights, beta_weights, theta
        )
        results.append({
            "Objek Alternatif": f"u_{j+1}",
            "Energi Informasional (ζ)": energy_obj,
            "Ukuran Korelasi (C)": corr_measure,
            "Koefisien Korelasi (δ)": corr_coeff,
        })

    df_results = (
        pd.DataFrame(results)
        .sort_values(by="Koefisien Korelasi (δ)", ascending=False)
        .reset_index(drop=True)
    )
    df_results.index += 1
    df_results.index.name = "Peringkat"

    st.markdown("---")
    st.subheader("Hasil Perhitungan dan Analisis Peringkat")
    st.success(
        f"Berdasarkan nilai koefisien korelasi tertinggi, objek alternatif terbaik adalah **{df_results.iloc[0]['Objek Alternatif']}**."
    )
    st.table(df_results)

    # Rincian bobot antar-langkah untuk pengecekan / lampiran
    with st.expander("Lihat Detail Bobot Penilai (β_r) & Bobot Kriteria (α_i)"):
        st.write("**Matriks Korelasi Antar Penilai δ^θ{(ψ_r, Λ), (ψ_s, Λ)} (Langkah 2)**")
        eval_labels = [f"P_{r+1}" for r in range(n_evaluators)]
        st.dataframe(pd.DataFrame(delta_rs, index=eval_labels, columns=eval_labels), use_container_width=True)

        st.write("**Efisiensi Korelasi & Bobot Penilai β_r (Langkah 3)**")
        df_beta = pd.DataFrame({
            "Penilai": eval_labels,
            "Efisiensi γ^θ": gamma_r,
            "Bobot Penilai (β_r)": beta_weights,
        })
        st.dataframe(df_beta, hide_index=True, use_container_width=True)

        st.write("**Entropi Kumulatif, Divergensi & Bobot Kriteria α_i (Langkah 4)**")
        df_alpha = pd.DataFrame({
            "Pasangan Kriteria": [f"A_{i+1}" for i in range(m_params)],
            "Entropi Kumulatif Ē(A_i)": cum_entropy_i,
            "Divergensi (DV_i)": divergence_i,
            "Bobot Kriteria (α_i)": alpha_weights,
        })
        st.dataframe(df_alpha, hide_index=True, use_container_width=True)
