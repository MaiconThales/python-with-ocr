import cv2 # OpenCV

"""
Módulo de pré-processamento de imagens para OCR.

Contém funções para conversão de cores, binarização, inversão e redimensionamento
de imagens, com foco em preparar imagens para ferramentas como Tesseract.
"""

def convertBGRtoRGB(image):
    """
    Converte uma imagem do espaço de cor BGR (padrão do OpenCV)
    para RGB (padrão do Pillow / Tesseract / exibição correta).

    Args:
        image (numpy.ndarray): Imagem em formato BGR.

    Returns:
        numpy.ndarray: Imagem convertida para RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def grayscale(img):
    """
    Converte uma imagem BGR para escala de cinza.

    Args:
        img (numpy.ndarray): Imagem em BGR.

    Returns:
        numpy.ndarray: Imagem em escala de cinza.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binarizationSimple(gray, threshold, thresholdMax):
    """
    Binariza a imagem em tons de cinza usando limiarização simples.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinza.
        threshold (int): Limiar mínimo.
        thresholdMax (int): Valor máximo (geralmente 255).

    Returns:
        numpy.ndarray: Imagem binarizada.
    """
    val, thresh = cv2.threshold(gray, threshold, thresholdMax, cv2.THRESH_BINARY)
    print(f'preprocessing - binarizationSimple - {val}', end='\n\n')
    return thresh

def binarizationOtsu(gray):
    """
    Binariza a imagem usando o método de Otsu, que calcula o melhor limiar automaticamente.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinza.

    Returns:
        numpy.ndarray: Imagem binarizada.
    """
    val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f'preprocessing - binarizationOtsu - {val}', end='\n\n')
    return otsu

def binarizationAdaptive(gray):
    """
    Binarização adaptativa por média local.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinza.

    Returns:
        numpy.ndarray: Imagem binarizada.
    """
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)

def binarizationAdaptiveGaussiana(gray):
    """
    Binarização adaptativa usando método Gaussiano.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinza.

    Returns:
        numpy.ndarray: Imagem binarizada.
    """
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 9)

def colorInversion(gray):
    """
    Inverte as cores da imagem em tons de cinza (texto preto → fundo branco).

    Recomendado para OCR, pois o Tesseract funciona melhor com texto escuro
    sobre fundo claro.

    Args:
        gray (numpy.ndarray): Imagem em escala de cinza.

    Returns:
        numpy.ndarray: Imagem com cores invertidas.
    """
    invert = 255 - gray
    return invert

def resizing(gray, fx, fy, interpolation):
    """
    Redimensiona a imagem com base em fatores de escala.

    Args:
        gray (numpy.ndarray): Imagem de entrada.
        fx (float): Fator de escala horizontal ( >1 aumenta, <1 diminui ).
        fy (float): Fator de escala vertical ( >1 aumenta, <1 diminui ).
        interpolation (int): Método de interpolação (ex: cv2.INTER_LINEAR).

    Returns:
        numpy.ndarray: Imagem redimensionada.
    """
    return cv2.resize(gray, None, fx=fx, fy=fy, interpolation=interpolation)

def removeNoiseErosionTechnique(gray, matriz):
    """
    Remove ruídos de uma imagem aplicando a técnica de **erosão**.

    A erosão "encolhe" os objetos brancos na imagem, removendo pequenos pontos
    isolados de ruído. Ideal para limpar pixels soltos no fundo.

    Args:
        gray (numpy.ndarray): Imagem em tons de cinza ou binarizada.
        matriz (numpy.ndarray): Elemento estruturante (kernel) para a erosão.

    Returns:
        numpy.ndarray: Imagem após a erosão.
    """
    erosion = cv2.erode(gray, matriz)
    print(f'preprocessing - removeNoiseErosionTechnique - {erosion}', end='\n\n')
    return erosion

def removeNoiseDilationTechnique(gray, matriz):
    """
    Remove ruídos ou reforça contornos aplicando a técnica de **dilatação**.

    A dilatação "expande" os objetos brancos, útil para preencher falhas em
    caracteres após erosão ou para conectar componentes desconectados.

    Args:
        gray (numpy.ndarray): Imagem em tons de cinza ou binarizada.
        matriz (numpy.ndarray): Elemento estruturante (kernel) para a dilatação.

    Returns:
        numpy.ndarray: Imagem após a dilatação.
    """
    return cv2.dilate(gray, matriz)

def blur(gray):
    return cv2.blur(gray, (5, 5))

def blurByGaussian(gray):
    # Mais usado para objetos
    return cv2.GaussianBlur(gray, (5, 5), 0)

def blurByMedia(gray):
    # Mais usado para objetos
    return cv2.medianBlur(gray, 3)

def bilateralBlur(gray):
    return cv2.bilateralFilter(gray, 15, 20, 45)