# Fan Audio Fault Detection

<p align="center">
  <img src="./assets/banner.svg" alt="Project Banner" width="100%" />
</p>

<p align="left">
	<img src="https://img.shields.io/badge/Python-3.x-FFD700?style=for-the-badge&logo=python&logoColor=111111&labelColor=0B0B0B" alt="Python" />
	<img src="https://img.shields.io/badge/Scikit_Learn-9F00FF?style=for-the-badge&logo=scikit-learn&logoColor=111111&labelColor=0B0B0B" alt="Scikit-Learn" />
	<img src="https://img.shields.io/badge/Librosa-00FFF7?style=for-the-badge&logo=librosa&logoColor=111111&labelColor=0B0B0B" alt="Librosa" />
	<img src="https://img.shields.io/badge/Status-Estável-FF00FF?style=for-the-badge&logoColor=111111&labelColor=0B0B0B" alt="Status" />
</p>

Este projeto utiliza aprendizado de máquina para detectar falhas em ventiladores industriais através da análise de áudios. Baseado no dataset MIMII, o sistema extrai características MFCC dos sinais de áudio e classifica as condições como normais ou anômalas usando um modelo Random Forest.

## [>] SYS.NAVEGAÇÃO

[Descrição](#-descrição-do-problema) • [Stack](#-stack-tecnológica) • [Estrutura](#-estrutura-do-projeto) • [Execução](#-instalação-e-execução) • [Resultados](#-resultados) • [Teoria](#-fundamentação-teórica)

---

## [~] SYS.DESCRIÇÃO_PROBLEMA

Sistemas tradicionais de detecção de falhas em equipamentos industriais podem ser ineficientes, especialmente em ambientes ruidosos. Este projeto aborda a detecção precoce de anomalias em ventiladores industriais via análise de áudio, permitindo manutenção preditiva e redução de custos operacionais.

## [@] STACK_TECNOLÓGICA

- **Linguagem:** Python 3.x
- **Processamento de Áudio:** Librosa (extração de MFCC)
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Data Science:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn
- **Dataset:** MIMII Dataset (subconjunto de ventiladores)

## [=] ESTRUTURA_PROJETO

```
fan-audio-fault-detection/
├── assets/               # HUDs e Banner Cyberpunk
├── data/                 # Diretório para datasets (MIMII)
├── models/               # Modelos treinados salvos
├── src/                  # Código fonte
│   └── main.py           # Script principal de treinamento e avaliação
├── docs/                 # Documentação técnica
├── requirements.txt      # Dependências do sistema
└── README.md             # Documentação do sistema
```

### Estrutura Esperada do Dataset

O dataset deve seguir a estrutura do MIMII:
`data/fan/id_XX/normal|abnormal/*.wav`

---

## [*] INSTALAÇÃO_E_EXECUÇÃO

### 1. Clonar o sistema
```bash
git clone https://github.com/lucenfort/fan-audio-fault-detection.git
cd fan-audio-fault-detection
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Preparar dados
Baixe o [MIMII Dataset](https://zenodo.org/record/3384388) e coloque na pasta `data/`.

### 4. Executar pipeline
```bash
python src/main.py
```

---

## [#] RESULTADOS_SISTEMA

Após o treinamento, o modelo Random Forest alcançou os seguintes marcos de performance:

- **Acurácia Global:** 96.49%
- **AUC-ROC:** 0.9965

<p align="center">
  <img src="confusion_matrix.png" width="45%" alt="Matriz de Confusão" />
  <img src="roc_curve.png" width="45%" alt="Curva ROC" />
</p>

---

## [&] FUNDAMENTAÇÃO_TEÓRICA

### MFCC (Mel-frequency Cepstral Coefficients)
Representação das características espectrais do áudio, simulando a percepção auditiva humana para tarefas de classificação sonora.

### Random Forest
Algoritmo ensemble que combina múltiplas árvores de decisão para criar um classificador robusto e resistente a overfitting.

---


## 📚 Referências

- BREIMAN, L. Random forests. Machine Learning, v. 45, n. 1, p. 5–32, 2001.
- PUROHIT, H.; TANABE, R.; ICHIGE, K.; ENDO, T.; NIKAIDO, Y.; SUEFUSA, K.; KAWAGUCHI, Y. MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection. arXiv preprint, arXiv:1909.09347, 2019.
- PUROHIT, H.; TANABE, R.; ICHIGE, K.; ENDO, T.; NIKAIDO, Y.; SUEFUSA, K.; KAWAGUCHI, Y. MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection. In: Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 4., 2019. Proceedings [...]. [S.l.: s.n.], 2019.
- FAWCETT, T. An introduction to ROC analysis. Pattern Recognition Letters, v. 27, n. 8, p. 861–874, 2006.
