import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# update final
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
    
     # 6. Scaling Data Numerik tanpa Data Leakage
    numerical_features = X.columns

    # Lakukan Split Data terlebih dahulu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Inisialisasi Scaler
    scaler = StandardScaler()

    # Fit & Transform pada X_train, dan HANYA Transform pada X_test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Konversi kembali ke DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=numerical_features, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=numerical_features, index=X_test.index)

    # Gabungkan kembali Fitur dan Target
    X_processed_df = pd.concat([X_train_df, X_test_df])
    y_combined = pd.concat([y_train, y_test])

    # 7. Satukan kembali hasilnya
    processed_df = pd.concat([X_processed_df, y_combined], axis=1)

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
    
    input_data_path = "../dataset_raw/data_lansia.csv"
    output_folder_path = "dataset_preprocessing"

    preprocess_risk_data(
        input_csv_path=input_data_path,
        output_dir=output_folder_path
    )