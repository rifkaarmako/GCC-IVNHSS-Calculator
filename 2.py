import numpy as np
import pandas as pd
import streamlit as st


def _safe_pow(arr, exp):
    """Menghitung pangkat secara aman untuk nilai non-negatif."""
    arr = np.asarray(arr, dtype=float)
    return np.power(np.clip(arr, 0.0, None), exp)


def calc_weighted_energy(matrix, alpha, beta, theta):
    """
    Persamaan (3.1.1):
    zeta^theta(psi, Lambda) = sum_j beta_j * sum_{i=1}^m alpha_i * sum_{6 komponen} (x_ij)^theta
    matrix shape: (num_outer, m, 6)
    alpha shape: (m,)
    beta shape: (num_outer,)
    """
    powered = _safe_pow(matrix, theta)
    sum_components = np.sum(powered, axis=-1)
    weighted_criteria = np.sum(sum_components * alpha[None, :], axis=-1)
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
    prod_comp = np.sum(p1 * p2, axis=-1)
    weighted_criteria = np.sum(prod_comp * alpha[None, :], axis=-1)
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
    ii_c_l = np.abs(i_l + i_u - 1.0)
    ii_c_u = np.abs(i_u + i_l - 1.0)

    num = 4.0 - (tf_l + tf_u) - (ii_c_l + ii_c_u)
    den = 4.0 + (tf_l + tf_u) + (ii_c_l + ii_c_u)

    return np.mean(num / den, axis=0)


def validate_ivnhss_tables(ideal_dfs, objects_dfs):
    """Memeriksa rentang [0, 1] dan konsistensi batas bawah <= batas atas."""
    warnings = []
    all_tables = [("Objek Ideal", r, df) for r, df in enumerate(ideal_dfs)]
    for j, obj_list in enumerate(objects_dfs):
        for r, df in enumerate(obj_list):
            all_tables.append((f"Objek u_{j+1}", r, df))

    for label, r, df in all_tables:
        vals = df.drop(columns=["Pasangan Kriteria"]).to_numpy(dtype=float)
        if np.any(vals < 0.0) or np.any(vals > 1.0):
            warnings.append(f"{label} (P_{r+1}): terdapat nilai di luar rentang [0, 1].")
        for _, row in df.iterrows():
            crit = row["Pasangan Kriteria"]
            if row["T^L"] > row["T^U"]:
                warnings.append(f"{label} (P_{r+1}, {crit}): T^L > T^U.")
            if row["I^L"] > row["I^U"]:
                warnings.append(f"{label} (P_{r+1}, {crit}): I^L > I^U.")
            if row["F^L"] > row["F^U"]:
                warnings.append(f"{label} (P_{r+1}, {crit}): F^L > F^U.")
    return warnings


def render_math_table(df, col_labels=None, index_label=None, decimals=6):
    """Menampilkan DataFrame sebagai tabel HTML yang mendukung tag <sub> dan <sup>."""
    cols = df.columns.tolist()
    headers = []
    if index_label is not None:
        headers.append(index_label)
    for c in cols:
        headers.append(col_labels.get(c, c) if col_labels else c)

    th_html = "".join(
        f"<th style='text-align:left; padding:8px 10px; border-bottom:2px solid rgba(128,128,128,0.4);'>{h}</th>"
        for h in headers
    )

    rows_html = []
    for idx, row in df.iterrows():
        cells = []
        if index_label is not None:
            idx_str = f"{idx:.{decimals}f}" if isinstance(idx, (float, np.floating)) else str(idx)
            cells.append(f"<td style='padding:7px 10px; border-bottom:1px solid rgba(128,128,128,0.2);'>{idx_str}</td>")
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cell = f"{val:.{decimals}f}"
            else:
                cell = str(val)
            cells.append(f"<td style='padding:7px 10px; border-bottom:1px solid rgba(128,128,128,0.2);'>{cell}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    html = (
        "<div style='overflow-x:auto; margin:6px 0 14px 0;'>"
        "<table style='width:100%; border-collapse:collapse; font-size:0.95rem;'>"
        f"<thead><tr>{th_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


st.set_page_config(page_title="Kalkulator IVNHSS Berbobot", layout="wide")
st.title("Kalkulator Koefisien Korelasi Berbobot Diperumum pada IVNHSS")
st.caption(
    "Perhitungan Bobot Penilai (Efisiensi Korelasi Ternormalisasi), Bobot Kriteria (Entropi & Divergensi), "
    "serta Peringkat Objek Alternatif pada *Interval-Valued Neutrosophic Hypersoft Set*"
)

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


col_cfg = {
    "Pasangan Kriteria": st.column_config.TextColumn(disabled=True),
    "T^L": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    "T^U": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    "I^L": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    "I^U": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    "F^L": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
    "F^U": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
}

st.subheader("1. Input Data Objek Ideal IVNHSS")
ideal_data = []
tabs_ideal = st.tabs([f"Pengambil Keputusan P_{r+1}" for r in range(n_evaluators)])

for r, tab in enumerate(tabs_ideal):
    with tab:
        st.caption()
        df_ideal = st.data_editor(
            create_default_df(m_params),
            column_config=col_cfg,
            key=f"ideal_eval_{r}",
            hide_index=True,
            use_container_width=True,
        )
        ideal_data.append(df_ideal)

st.markdown("---")

st.subheader("2. Input Data Objek Alternatif IVNHSS")
st.latex(r"(\psi_r, \Lambda) \text{ atas } U = \{u_1, u_2, \dots, u_k\} \text{ untuk setiap penilai } P_r,\; r = 1, 2, \dots, n")
objects_data = []

for j in range(k_objects):
    st.markdown(f"**Data Objek Alternatif $u_{{{j+1}}}$**")
    tabs_obj = st.tabs([f"Pengambil Keputusan P_{r+1}" for r in range(n_evaluators)])

    obj_eval_data = []
    for r, tab in enumerate(tabs_obj):
        with tab:
            st.caption(f"Penilaian objek $u_{{{j+1}}}$ oleh penilai $P_{{{r+1}}}$ pada setiap pasangan kriteria $A_i$.")
            df_obj = st.data_editor(
                create_default_df(m_params),
                column_config=col_cfg,
                key=f"obj_{j}_eval_{r}",
                hide_index=True,
                use_container_width=True,
            )
            obj_eval_data.append(df_obj)

    objects_data.append(obj_eval_data)
    st.write("")

if st.button("Hitung dan Tampilkan Peringkat", type="primary"):
    input_warnings = validate_ivnhss_tables(ideal_data, objects_data)
    if input_warnings:
        st.warning("Perhatian pada data input:\n- " + "\n- ".join(input_warnings))

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
    results = []
    for j in range(k_objects):
        psi_bar_j = raw_matrix[:, j, :, :]  # shape: (n, m, 6)
        corr_coeff, corr_measure, energy_obj, _ = calc_weighted_corr_coeff(
            psi_bar_j, ideal_matrix, alpha_weights, beta_weights, theta
        )
        results.append({
            "objek": f"u<sub>{j+1}</sub>",
            "objek_plain": f"u_{j+1}",
            "energi": energy_obj,
            "ukuran": corr_measure,
            "koef": corr_coeff,
        })

    df_sorted = (
        pd.DataFrame(results)
        .sort_values(by="koef", ascending=False)
        .reset_index(drop=True)
    )
    df_sorted.index += 1

    max_corr = df_sorted["koef"].max()
    best_objects = df_sorted.loc[df_sorted["koef"] == max_corr, "objek_plain"].tolist()
    best_label = ", ".join(f"**{obj}**" for obj in best_objects)

    st.markdown("---")
    st.subheader("Hasil Perhitungan dan Analisis Peringkat")
    st.success(
        f"Berdasarkan nilai koefisien korelasi tertinggi ({max_corr:.6f}), "
        f"objek alternatif terbaik adalah {best_label}."
    )

    df_rank_disp = df_sorted[["objek", "energi", "ukuran", "koef"]]
    render_math_table(
        df_rank_disp,
        index_label="Peringkat",
        col_labels={
            "objek": "Objek Alternatif",
            "energi": "Energi Informasional (ζ<sup>θ</sup>)",
            "ukuran": "Ukuran Korelasi (C<sup>θ</sup>)",
            "koef": "Koefisien Korelasi (δ<sup>θ</sup>)",
        },
    )

    with st.expander("Lihat Detail Bobot Penilai (β<sub>r</sub>) & Bobot Kriteria (α<sub>i</sub>)"):
        eval_labels_html = [f"P<sub>{r+1}</sub>" for r in range(n_evaluators)]
        crit_labels_html = [f"A<sub>{i+1}</sub>" for i in range(m_params)]

        st.markdown("**Matriks Korelasi Antar Penilai δ<sup>θ</sup>{(ψ<sub>r</sub>, Λ), (ψ<sub>s</sub>, Λ)} (Langkah 2)**", unsafe_allow_html=True)
        df_delta_disp = pd.DataFrame(delta_rs, index=eval_labels_html, columns=eval_labels_html)
        render_math_table(df_delta_disp, index_label="Penilai")

        st.markdown("**Efisiensi Korelasi & Bobot Penilai β<sub>r</sub> (Langkah 3)**", unsafe_allow_html=True)
        df_beta_disp = pd.DataFrame({
            "penilai": eval_labels_html,
            "efisiensi": gamma_r,
            "beta": beta_weights,
        })
        render_math_table(
            df_beta_disp,
            col_labels={
                "penilai": "Penilai",
                "efisiensi": "Efisiensi (γ<sup>θ</sup>)",
                "beta": "Bobot Penilai (β<sub>r</sub>)",
            },
        )

        st.markdown("**Nilai Entropi Masing-Masing IVNS E<sub>r</sub>(A<sub>i</sub>) per Penilai & Kriteria (Langkah 4 - Persamaan 2.3.2)**", unsafe_allow_html=True)
        df_entropy_disp = pd.DataFrame(entropy_r_i, index=eval_labels_html, columns=crit_labels_html)
        render_math_table(df_entropy_disp, index_label="Penilai")

        st.markdown("**Entropi Kumulatif, Divergensi & Bobot Kriteria α<sub>i</sub> (Langkah 4 - Persamaan 3.2.1–3.2.3)**", unsafe_allow_html=True)
        df_alpha_disp = pd.DataFrame({
            "kriteria": crit_labels_html,
            "entropi_kum": cum_entropy_i,
            "dv": divergence_i,
            "alpha": alpha_weights,
        })
        render_math_table(
            df_alpha_disp,
            col_labels={
                "kriteria": "Pasangan Kriteria",
                "entropi_kum": "Entropi Kumulatif Ē(A<sub>i</sub>)",
                "dv": "Divergensi (DV<sub>i</sub>)",
                "alpha": "Bobot Kriteria (α<sub>i</sub>)",
            },
        )
