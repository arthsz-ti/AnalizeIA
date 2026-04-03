Aqui está um modelo de **README.md** completo e profissional para o seu projeto, já incluindo os nomes e registros acadêmicos (RAs) da equipe, além de detalhes técnicos extraídos dos seus arquivos.

---

# AnalizeIA – Sistema Inteligente de Análise de Imagens com IA
### Atividade Integradora - Visão Computacional e Deep Learning

Este projeto consiste em um sistema de visão computacional desenvolvido para integrar conceitos de processamento digital de imagem e inteligência artificial. A aplicação utiliza uma interface gráfica interativa para demonstrar o pipeline completo de análise de imagem, desde a aquisição até a detecção de objetos em tempo real.

---

## 👥 Equipe
* **Arthur Saraiva de Souza** - RA: 2404043
* **Lucas Gobbo Cruz** - RA: 2406898
* **William** - RA: 2424242

---

## 🚀 Funcionalidades Principais
O **AnalizeIA** executa um pipeline de processamento dividido em etapas fundamentais:

1.  **Aquisição de Imagem:** Captura via webcam ou carregamento de arquivos locais.
2.  **Processamento Digital:** Filtros de Escala de Cinza, Blur (Gaussiano) e Detecção de Bordas (Canny).
3.  **Análise Cromática:** Conversão para o espaço de cor **HSV** com manipulação de Matiz e Saturação.
4.  **Análise Estatística:** Geração de histogramas em tempo real com diagnóstico de iluminação (Subexposição/Superexposição).
5.  **Segmentação:** Binarização (Thresholding) para isolamento de objetos.
6.  **Inteligência Artificial:** Detecção, localização e contagem de objetos utilizando o modelo **YOLOv8** (You Only Look Once).

---

## 🛠️ Tecnologias e Dependências
O projeto foi desenvolvido em **Python 3** utilizando as seguintes bibliotecas:

* **OpenCV (`cv2`):** Processamento de imagem e vídeo.
* **Ultralytics (YOLOv8):** Modelo de Deep Learning para detecção de objetos.
* **Tkinter:** Interface gráfica (GUI).
* **Matplotlib:** Plotagem de histogramas e análise de dados.
* **NumPy:** Operações matemáticas em matrizes de imagem.
* **Pillow (PIL):** Manipulação de formatos de imagem para a interface.

---

## 📦 Como Instalar e Executar

1. **Clone o repositório:**
   ```powershell
   git clone https://github.com/seu-usuario/AnalizeIA.git
   cd AnalizeIA
   ```

2. **Crie e ative um ambiente virtual (venv):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Instale as dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```powershell
   python AnalizeIA.py
   ```

---

## 📂 Estrutura do Repositório
* `AnalizeIA.py`: Script principal contendo a lógica da GUI e do pipeline.
* `yolov8n.pt`: Pesos da rede neural (modelo Nano do YOLOv8).
* `relatorio.md`: Documentação técnica detalhada das etapas do projeto.
* `requirements.txt`: Lista de dependências do Python.
* `ATIVIDADE INTEGRADORA_Alunos.pdf`: Guia e requisitos da atividade.

---

## 📄 Notas de Implementação
* **Threading:** O sistema utiliza threads separadas para o processamento de IA, garantindo que a interface gráfica não trave durante a análise de vídeo.
* **Performance:** Foi utilizado o modelo `yolov8n` (nano) por ser otimizado para execução em CPUs domésticas com alta taxa de quadros (FPS).

---

> **Aviso:** Certifique-se de que a webcam não está sendo usada por outro aplicativo ao iniciar o modo de captura em tempo real.
