import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_risk_data(input_csv_path, output_dir):
    """
    Melakukan data preprocessing otomatis untuk dataset Risiko Kesehatan.

    Args:
        input_csv_path (str): Jalur ke file CSV data mentah.
        output_dir (str): Direktori tempat menyimpan data hasil preprocessing.
    """
    print("=== Memulai Preprocessing Dataset Risiko Kesehatan ===")

    # 1. Load data
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {input_csv_path}")
        return

    print(f"Data Loaded: {df.shape} baris.")

    # 2. Penanganan Fitur dan Target
    # Variabel Target: Status_Resiko (Kategorikal)
    # Semua Fitur Input: Numerik

    # 3. Encoding Target (Mengubah LOW RISK/HIGH RISK menjadi 0/1)
    df['Status_Resiko'] = df['Status_Resiko'].map({'LOW RISK': 0, 'HIGH RISK': 1})
    
    # 4. Cek dan Tangani Nilai Kosong/Duplikat (Berdasarkan EDA, data ini sudah bersih)
    # Jika ada nilai kosong: df.dropna(inplace=True) 

    # 5. Definisikan X dan y
    X = df.drop(columns='Status_Resiko')
    y = df['Status_Resiko']
    
    # 6. Scaling Data Numerik (StandardScaler)

    numerical_features = X.columns # Semua kolom adalah numerik
    
    # Inisialisasi Scaler
    scaler = StandardScaler()
    
    # Terapkan Scaling ke seluruh fitur
    X_scaled = scaler.fit_transform(X)
    
    # Konversi kembali ke DataFrame
    X_processed_df = pd.DataFrame(X_scaled, columns=numerical_features, index=X.index)

    # 7. Satukan kembali hasilnya (Fitur + Target)
    processed_df = pd.concat([X_processed_df, y], axis=1)

    # 8. Save hasil preprocessing
    output_file_name = "preprocessed_data.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Buat direktori jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    processed_df.to_csv(output_file_path, index=False)

    print("Preprocessing selesai.")
    print(f"Dataset disimpan di: {output_file_path}")
    print(f"Shape Dataset Akhir: {processed_df.shape}")


if __name__ == "__main__":
    # --- JALANKAN FUNGSI OTOMATISASI ---
    # Asumsi: File CSV mentah berada di 'namadataset_raw/raw_data.csv'
    # Asumsi: Data hasil preprocessing akan disimpan di 'preprocessing/namadataset_preprocessing/'
    
    input_data_path = "dataset_raw/data_lansia.csv"
    output_folder_path = "preprocessing/dataset_preprocessing"

    preprocess_risk_data(
        input_csv_path=input_data_path,
        output_dir=output_folder_path
    )