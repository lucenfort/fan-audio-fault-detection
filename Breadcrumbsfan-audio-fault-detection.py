import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Função para coletar arquivos de áudio da máquina "fan"
def collect_fan_audio_files(dataset_dir):
    """
    Coleta os arquivos de áudio da máquina 'fan' e retorna um dataframe com os caminhos dos arquivos e seus respectivos status.
    O status será 0 para 'normal' e 1 para 'abnormal'.
    """
    audio_data = []

    machine_dir = os.path.join(dataset_dir, 'fan')
    if os.path.isdir(machine_dir):
        for model_id in os.listdir(machine_dir):
            model_dir = os.path.join(machine_dir, model_id)
            if os.path.isdir(model_dir):
                for category in ['normal', 'abnormal']:  # Substituindo por 'status'
                    category_dir = os.path.join(model_dir, category)
                    if os.path.isdir(category_dir):
                        for audio_file in os.listdir(category_dir):
                            if audio_file.endswith(".wav"):
                                file_path = os.path.join(category_dir, audio_file)
                                status = 0 if category == 'normal' else 1  # Atribuindo 0 para normal e 1 para abnormal
                                audio_data.append({
                                    'file_path': file_path,
                                    'status': status
                                })

    return pd.DataFrame(audio_data)


# Função para extrair MFCCs de arquivos de áudio
def extract_mfccs(file_path, n_mfcc=40):
    """
    Carrega um arquivo .wav usando librosa e extrai as características MFCC.
    """
    data, sample_rate = librosa.load(file_path, sr=None)  # Carrega o arquivo .wav com a taxa de amostragem original
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)  # Extrai MFCCs
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Calcula a média dos MFCCs ao longo do tempo
    return mfccs_mean


# Função para preprocessar os dados e salvar as características em um CSV
def preprocess_and_save_to_csv(df, output_csv):
    """
    Carrega os arquivos de áudio, extrai os MFCCs e salva as características em um arquivo CSV.
    """
    features_list = []
    labels_list = []

    # Usando tqdm para mostrar o progresso enquanto processa os arquivos
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extraindo MFCCs"):
        mfccs = extract_mfccs(row['file_path'])  # Extrai as características MFCC do arquivo .wav
        features_list.append(mfccs)
        labels_list.append(row['status'])

    # Convertendo as listas para um dataframe
    features_df = pd.DataFrame(features_list)
    features_df['status'] = labels_list  # Adiciona a coluna de status ao dataframe

    # Imprimir a quantidade de arquivos normais e anormais
    total = len(features_df)
    normais = np.sum(features_df['status'] == 0)
    anormais = np.sum(features_df['status'] == 1)

    print(f"Quantidade total de arquivos: {total}")
    print(f"Arquivos normais: {normais}")
    print(f"Arquivos anormais: {anormais}")

    # Salvando em um arquivo CSV
    features_df.to_csv(output_csv, index=False)
    print(f"Características extraídas salvas em {output_csv}")


# Função para treinar o modelo Random Forest
def train_random_forest(X_train, y_train):
    """
    Treina um modelo de Random Forest.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Função para calcular as métricas de avaliação e plotar gráficos
def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo e plota as métricas de avaliação.
    """
    y_pred = model.predict(X_test)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {accuracy:.5f}")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Anormal'],
                yticklabels=['Normal', 'Anormal'])
    plt.title("Matriz de Confusão")
    plt.xlabel("Rótulos Preditos")
    plt.ylabel("Rótulos Reais")
    plt.savefig("confusion_matrix.png", dpi=300)  # Salva a matriz de confusão em alta definição
    plt.show()

    print("Matriz de Confusão:")
    print(cm)

    # ROC e AUC
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.5f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title("Curva ROC")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png", dpi=300)  # Salva a curva ROC em alta definição
    plt.show()

    print(f"AUC: {auc:.5f}")


# Função principal
def main():
    # Caminho do diretório do dataset
    dataset_dir = 'dataset'

    # Coletar os arquivos de áudio da máquina 'fan'
    df = collect_fan_audio_files(dataset_dir)

    # Caminho para salvar o arquivo CSV com as características extraídas
    output_csv = 'fan_audio_features.csv'

    # Pré-processar os dados e salvar em CSV
    preprocess_and_save_to_csv(df, output_csv)

    # Carregar o CSV de características
    data = pd.read_csv(output_csv)
    X = data.drop(columns=['status']).values
    y = data['status'].values

    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Treinar o modelo Random Forest
    model = train_random_forest(X_train, y_train)

    # Avaliar o modelo
    evaluate_model(model, X_test, y_test)

    # Salvar o modelo treinado
    model_file = 'fan_random_forest_model.pkl'
    joblib.dump(model, model_file)
    print(f"Modelo salvo como {model_file}")


if __name__ == "__main__":
    main()
