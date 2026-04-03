# AnalizeIA – Análise de Imagens com IA
### Atividade Integradora: Visão Computacional e Deep Learning

Sistema de visão computacional que integra processamento digital de imagem e inteligência artificial em uma interface interativa, cobrindo do pré-processamento à detecção de objetos em tempo real.

---

## 👥 Equipe
* **Arthur Saraiva de Souza** (RA: 2404043)
* **Lucas Gobbo Cruz** (RA: 2406898)
* **William** (RA: 2424242)

---

## 🚀 Funcionalidades
1. **Aquisição:** Webcam ou arquivos locais.
2. **Processamento:** Filtros de Cinza, Blur (Gaussiano) e Bordas (Canny).
3. **Cor e Segmentação:** Espaço HSV, binarização e manipulação de saturação.
4. **Diagnóstico:** Histogramas em tempo real (análise de iluminação).
5. **IA:** Detecção e contagem de objetos via **YOLOv8**.

---

## 🛠️ Tecnologias
* **Linguagem:** Python 3
* **Libs:** OpenCV, Ultralytics (YOLOv8), Tkinter, Matplotlib, NumPy, Pillow.

---

## 📦 Instalação e Execução
```powershell
# Clonar e acessar
git clone https://github.com/seu-usuario/AnalizeIA.git
cd AnalizeIA

# Configurar ambiente
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Executar
python AnalizeIA.py
```
