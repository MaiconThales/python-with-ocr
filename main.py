from tesseractUtils import *
from preprocessing import *
from pytesseract import TesseractError
import numpy as np

"""
Módulo principal para execução de OCR com Tesseract e pré-processamento de imagens.

Contém funções para:
- Detectar orientação de imagens
- Desenhar caixas em textos detectados
- Pré-processamento avançado ou simples
- Remoção de ruído
- Execução do fluxo principal de OCR
"""

def imageOrientation(pathImagem):
    """
    Detecta e exibe a orientação da imagem usando Tesseract OSD.

    Args:
        pathImagem (str): Caminho da imagem a ser processada.

    Notes:
        Se houver erro no Tesseract, exibe mensagem de aviso.
    """
    try:
        img = readImageWithPIL(pathImagem)
        print(f'imageOrientation - {imageToOsd(img)}', end='\n\n')
    except TesseractError:
        print(f'imageOrientation - Erro no Tesseract, por favor olhar depois', end='\n\n')

def drawBox(config, img):
    """
    Desenha caixas em torno de textos detectados pela função Tesseract image_to_data
    e exibe a imagem resultante.

    Args:
        config (str): Configurações do Tesseract (ex: '--tessdata-dir tessdata').
        img (numpy.ndarray): Imagem a ser processada.
    """
    minConfi = 40

    result = imageToData(
        img, 
        'por', 
        config
    )

    print(f'ImageData - {result}', end='\n\n')
    data, imgCopy = travelImage(img.copy(), result, minConfi, 1)
    showImage(imgCopy)
    print(f'Resultado da busca: {data}', end='\n\n')

def removeNoise(gray):
    """
    Remove ruído da imagem aplicando técnicas de dilatação e erosão.
    Pode utilizar a tecnica de ABERTURA para ruidos fora do texto.
    Pode utilizar a tecnica de FECHAMENTO para ruidos dentro do texto.

    Args:
        gray (numpy.ndarray): Imagem em tons de cinza ou binarizada.

    Returns:
        numpy.ndarray: Imagem processada com ruído reduzido.
    """
    matriz = np.ones((5, 5), np.uint8)

    # Abertura
    # gray = removeNoiseErosionTechnique(gray, matriz)
    # gray = removeNoiseDilationTechnique(gray, matriz)

    # Fechamento
    gray = removeNoiseDilationTechnique(gray, matriz)
    gray = removeNoiseErosionTechnique(gray, matriz)

    return gray

def preProcessing(img, advancedProcessing):
    """
    Aplica pré-processamento na imagem antes do OCR.

    Args:
        img (numpy.ndarray): Imagem original.
        advancedProcessing (bool): Indica se deve usar processamento avançado (binarização, remoção de ruído, etc.).

    Returns:
        numpy.ndarray: Imagem pré-processada para OCR.
    """
    if advancedProcessing:
        img = grayscale(img)

        """binarization"""
        # img = binarizationSimple(img, 140, 255)
        # img = binarizationOtsu(img)
        # img = binarizationAdaptive(img)
        # img = binarizationAdaptiveGaussiana(img)

        """color inversion"""
        # img = colorInversion(img)

        """resizing"""
        # img = resizing(img, 1.5, 1.5, cv2.INTER_CUBIC)

        """removeNoise"""
        # img = removeNoise(img)

        """blur"""
        # img = blur(img)
        # img = blurByGaussian(img)
        # img = blurByMedia(img)
        # img = bilateralBlur(img)
    else:
        img = convertBGRtoRGB(img)
    return img

def main():
    """
    Fluxo principal de execução do OCR:
    - Configura o caminho do Tesseract
    - Define opções de visualização e pré-processamento
    - Lê a imagem
    - Aplica pré-processamento
    - Exibe a imagem (opcional)
    - Desenha caixas sobre textos detectados (opcional)
    - Extrai texto usando Tesseract
    """
    pytesseract.pytesseract.tesseract_cmd = r"C:\Development Environment\Tesseract-OCR\tesseract.exe"
    viewImage = True
    viewImageWithBox = True

    advancedProcessing = True

    pathImage = 'img\\frase.jpg'

    # config = '--tessdata-dir tessdata --psm 9'
    config = '--tessdata-dir tessdata'

    imageOrientation(pathImage)

    img = readImageWithOpenCV(pathImage)

    img = preProcessing(img, advancedProcessing)

    if viewImage:
        img = prepareWindow(img)
        showImage(img)
    if viewImageWithBox and not advancedProcessing:
        drawBox(config, img)

    print(f'Text to String:\n {imageToString(img,'por',config)}', end='\n\n')
    

if __name__ == "__main__":
    main()