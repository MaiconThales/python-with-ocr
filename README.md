# 🐍 Projeto OCR com Python + Tesseract

Este projeto foi criado para **colocar em prática conhecimentos de Python e OCR (Reconhecimento Óptico de Caracteres)**, utilizando a biblioteca [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) e técnicas de **pré-processamento de imagens**.

---

## 📌 Tecnologias Utilizadas

- [Python 3](https://www.python.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenCV](https://opencv.org/) – para pré-processamento de imagens
- [Pillow](https://python-pillow.org/) – manipulação de imagens
- [pytesseract](https://pypi.org/project/pytesseract/) – integração Python ↔ Tesseract

---

## 🧠 Objetivo

O objetivo principal deste projeto é:

- Aprender e aplicar técnicas de OCR;
- Treinar habilidades de pré-processamento de imagens para melhorar a acurácia do reconhecimento;
- Explorar o funcionamento do Tesseract e suas opções.

---

## 🛠️ Pré-requisitos

Antes de rodar o projeto, você precisa ter:

### 1. Python 3 instalado

Verifique com:

```bash
python --version
```

### 2. Tesseract OCR instalado

Baixe o instalador [aqui](https://github.com/tesseract-ocr/tesseract).

Após instalar, adicione o caminho do executável do Tesseract às variáveis de ambiente do seu sistema:

**Exemplo no Windows:**
```
C:\Program Files\Tesseract-OCR
```

**Exemplo no Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

### 3. Configurar o pytesseract

Se estiver no Windows, no seu código você provavelmente precisará informar o caminho:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 📦 Instalação do Projeto

1. **Clone este repositório:**

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
```

2. **Crie e ative um ambiente virtual** (opcional, mas recomendado):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

Execute o script principal para testar o OCR:

```bash
python main.py

OR

py -m src.main
```

Ou substitua `main.py` pelo arquivo principal do seu projeto.

---

## 📝 Estrutura do Projeto

```
.
├── src/                # Código-fonte principal
│   ├── preprocessing.py
│   └── tesseractUtils.py
├── images/             # Imagens de teste
├── requirements.txt    # Dependências Python
├── README.md           # Este arquivo
└── .gitignore
```

---

## 🧪 Exemplos de Uso

```python
from tesseractUtils import readImageWithPIL, imageToText
from preprocessing import preprocess_image

img = readImageWithPIL("images/exemplo.png")
processed = preprocess_image(img)
texto = imageToText(processed)
print(texto)
```

---

## 📝 Notas

- A qualidade do reconhecimento depende muito da qualidade da imagem e das técnicas de pré-processamento aplicadas.
- Recomenda-se usar imagens com boa resolução e contraste.
- Teste diferentes técnicas de binarização, remoção de ruído e correção de rotação.

---

## 📄 Licença

Este projeto é de uso livre para estudos e experimentação.  
Sinta-se à vontade para adaptar e melhorar!

---

## 🤝 Contribuições

Contribuições são bem-vindas!  
Você pode abrir issues, sugerir melhorias ou enviar pull requests.

---

**✍️ Autor:** Maicon Thales Silva Gomes