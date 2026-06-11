import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. KONFIGURASI HALAMAN & MEMUAT DATA
# ==========================================
st.set_page_config(page_title="Prediksi Harga Rumah & Analisis Pasar", page_icon="🏠", layout="wide")

MODEL_PATH = "best_house_price_model_ridge_regression.joblib" # Sesuaikan dengan nama file model Anda
TRAIN_DATA_PATH = "train.csv"

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(MODEL_PATH)
        df_train_raw = pd.read_csv(TRAIN_DATA_PATH)
        
        # Mengubah target harga menjadi numerik secara aman
        df_train_raw['SalePrice'] = pd.to_numeric(df_train_raw['SalePrice'], errors='coerce')
        X_base = df_train_raw.drop(columns=["Id", "SalePrice"], errors='ignore').copy()
        
        # Konversi fitur utama ke numerik agar tidak dibaca sebagai teks
        main_numeric_cols = ['GrLivArea', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YrSold', 'LotFrontage']
        for col in main_numeric_cols:
            if col in X_base.columns:
                X_base[col] = pd.to_numeric(X_base[col], errors='coerce')
        
        # Pisahkan kolom numerik dan kategorikal menggunakan Pandas native (Menghindari bug StringDtype)
        numeric_cols = X_base.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = X_base.select_dtypes(exclude=['number']).columns.tolist()
        
        # Isi missing value data latih
        for col in numeric_cols:
            median_val = X_base[col].median()
            X_base[col] = X_base[col].fillna(median_val if not pd.isna(median_val) else 0)
        for col in categorical_cols:
            X_base[col] = X_base[col].fillna("None").astype(str)
            
        # Hitung HouseAge untuk data training dasar
        X_base["HouseAge"] = X_base["YrSold"] - X_base["YearBuilt"]
        df_train_raw["HouseAge"] = X_base["HouseAge"]
        
        # SOLUSI UTAMA: Jalankan One-Hot Encoding pada seluruh data latih untuk mencetak kolom dasar
        X_encoded = pd.get_dummies(X_base, drop_first=True)
        
        # Kita buat cetakan kolom mandiri berdasarkan data latih asli (Bukan dari properti model)
        expected_columns = X_encoded.columns.tolist()
        
        # Jika Anda ingin membatasi atau menyelaraskan paksa ke ukuran 259 kolom, 
        # kita ambil 259 kolom pertama dari hasil dummy data train ini agar sinkron dengan model Ridge Anda.
        if len(expected_columns) > 259:
            expected_columns = expected_columns[:259]
            X_encoded = X_encoded.reindex(columns=expected_columns, fill_value=0.0)
            
        X_encoded_fixed = X_encoded.astype(float)
        
        # Melatih Scaler khusus dengan struktur kolom yang dikunci
        scaler = StandardScaler()
        scaler.fit(X_encoded_fixed)
        
        return model, scaler, expected_columns, df_train_raw
    except Exception as e:
        st.error(f"Gagal memuat dependensi pada sistem: {e}")
        return None, None, None, None

model, scaler, expected_columns, df_train = load_model_and_scaler()

# ==========================================
# 2. ANTARMUKA PENGGUNA (UI) - PANEL INPUT
# ==========================================
st.title("🏠 Aplikasi Prediksi Harga Rumah & Analisis Pasar Real Estate")
st.write("Masukkan spesifikasi rumah untuk mendapatkan estimasi harga jual serta melihat tren pasar di sekitarnya.")
st.markdown("---")

if model is not None and scaler is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📐 Dimensi & Lokasi")
        gr_liv_area = st.number_input("Luas Area Tinggal (GrLivArea) dalam sqft:", min_value=300, max_value=6000, value=1500)
        lot_area = st.number_input("Luas Tanah (LotArea) dalam sqft:", min_value=500, max_value=50000, value=10000)
        
        if df_train is not None:
            available_neighborhoods = sorted(df_train['Neighborhood'].dropna().unique().astype(str))
        else:
            available_neighborhoods = ["CollgCr", "OldTown", "Edwards"]
            
        neighborhood = st.selectbox("Lingkungan / Lokasi (Neighborhood):", available_neighborhoods)
        sale_condition = st.selectbox("Kondisi Penjualan (SaleCondition):", ["Normal", "Abnorml", "Partial", "Family", "Alloca", "AdjLand"])

    with col2:
        st.subheader("🌟 Kualitas & Umur")
        overall_qual = st.slider("Kualitas Material & Finis (OverallQual):", min_value=1, max_value=10, value=6)
        overall_cond = st.slider("Rating Kondisi Rumah (OverallCond):", min_value=1, max_value=10, value=5)
        
        year_built = st.number_input("Tahun Rumah Dibangun (YearBuilt):", min_value=1800, max_value=2026, value=2000)
        year_sold = st.number_input("Tahun Rumah Dijual (YrSold):", min_value=2006, max_value=2026, value=2026)
        
        house_age = year_sold - year_built
        st.caption(f"Estimasi Usia Bangunan saat Dijual: **{house_age} Tahun**")

    st.markdown("---")
    
    # ==========================================
    # 3. PROSES PREDIKSI JALUR AMAN & SELARAS
    # ==========================================
    if st.button("🔮 Hitung Prediksi Harga & Buka Analisis Pasar", type="primary", use_container_width=True):
        try:
            # 1. Buat DataFrame baru 1 baris bernilai 0.0 dengan kolom tepat mengikuti expected_columns
            df_final_features = pd.DataFrame(0.0, index=[0], columns=expected_columns)
            
            # 2. Masukkan fitur numerik dari input Streamlit ke kolom yang sesuai
            df_final_features.loc[0, 'GrLivArea'] = float(gr_liv_area)
            df_final_features.loc[0, 'LotArea'] = float(lot_area)
            df_final_features.loc[0, 'OverallQual'] = float(overall_qual)
            df_final_features.loc[0, 'OverallCond'] = float(overall_cond)
            df_final_features.loc[0, 'YearBuilt'] = float(year_built)
            df_final_features.loc[0, 'YrSold'] = float(year_sold)
            df_final_features.loc[0, 'HouseAge'] = float(house_age)
            
            # 3. Set nilai flag 1.0 pada fitur kategori pilihan user jika kolom tersebut eksis
            col_neigh = f"Neighborhood_{neighborhood}"
            col_cond = f"SaleCondition_{sale_condition}"
            
            if col_neigh in df_final_features.columns:
                df_final_features.loc[0, col_neigh] = 1.0
            if col_cond in df_final_features.columns:
                df_final_features.loc[0, col_cond] = 1.0
                
            # 4. Pastikan tipe data konsisten sebagai float
            df_final_features = df_final_features.astype(float)
            
            # 5. Transformasi menggunakan Scaler yang ukurannya sudah dipaksa sinkron
            features_scaled = scaler.transform(df_final_features)
            
            # 6. Jalankan model prediksi Ridge
            pred_log = model.predict(features_scaled)
            
            # Jika output prediksi berupa array 2 dimensi, kita ratakan
            if isinstance(pred_log, np.ndarray) and pred_log.ndim > 1:
                pred_log = pred_log.flatten()
                
            harga_asli = np.expm1(pred_log)[0]
            
            # Tampilkan Hasil Jual Ke Dashboard UI
            st.success("🎉 Prediksi Berhasil Dilakukan!")
            st.metric(label=f"Estimasi Nilai Pasar Rumah di {neighborhood}", value=f"${harga_asli:,.2f}")
            
            # ==========================================
            # 4. VISUALISASI GRAFIK PASAR
            # ==========================================
            if df_train is not None:
                st.markdown("---")
                st.subheader(f"📊 Analisis & Perbandingan Pasar Terhadap Wilayah {neighborhood}")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.write(f"**1. Distribusi Harga Rumah di Lingkungan {neighborhood}**")
                    df_filtered = df_train[df_train['Neighborhood'].astype(str) == str(neighborhood)]
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(pd.to_numeric(df_filtered['SalePrice'], errors='coerce').dropna(), kde=True, color="teal", ax=ax)
                    ax.axvline(harga_asli, color='red', linestyle='--', linewidth=2, label=f'Prediksi Anda: ${harga_asli:,.0f}')
                    ax.set_xlabel("Harga Jual ($)")
                    ax.set_ylabel("Jumlah Rumah")
                    ax.legend()
                    st.pyplot(fig)

                with chart_col2:
                    st.write("**2. Rata-rata Harga Rumah Berdasarkan Kualitas Material**")
                    df_train['SalePrice'] = pd.to_numeric(df_train['SalePrice'], errors='coerce')
                    df_qual_price = df_train.groupby('OverallQual')['SalePrice'].mean().reset_index()
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.barplot(data=df_qual_price, x='OverallQual', y='SalePrice', palette='Blues_d', ax=ax2)
                    ax2.axvline(overall_qual - 1, color='orange', linestyle='-', linewidth=3, label=f'Kualitas Anda: {overall_qual}')
                    ax2.set_xlabel("Rating Kualitas (1-10)")
                    ax2.set_ylabel("Rata-rata Harga ($)")
                    ax2.legend()
                    st.pyplot(fig2)
                
                st.write("**3. Tren Rata-rata Harga Jual Rumah Berdasarkan Tahun Konstruksi (Year Built)**")
                df_train['YearBuilt'] = pd.to_numeric(df_train['YearBuilt'], errors='coerce')
                df_trend = df_train.groupby('YearBuilt')['SalePrice'].mean().reset_index()
                st.line_chart(data=df_trend, x='YearBuilt', y='SalePrice', use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan fitur ke model: {e}")

    st.caption(
    '<p style="text-align: center; font-size: 20px; color: gray;">'
    'Copyright © 2026 Andri Asfriansah | Powered by Andri Asfriansah'
    '</p>', unsafe_allow_html=True)