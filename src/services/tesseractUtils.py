import pytesseract
import numpy as np
import cv2 # OpenCV
import re
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output
from imutils.object_detection import non_max_suppression

"""
Módulo utilitário para leitura, processamento e anotação de imagens com OpenCV e Tesseract.

Contém funções para:
- Ler imagens com OpenCV ou PIL
- Preparar janelas de exibição
- Redimensionar imagens
- Mostrar imagens
- Desenhar caixas e textos sobre imagens
- Extrair texto e dados usando Tesseract
- Buscar padrões via regex
- Detectar regiões de texto usando rede neural EAST
"""

def readImageWithOpenCV(pathImage):
    """
    Lê uma imagem usando OpenCV.

    Args:
        pathImage (str): Caminho do arquivo de imagem.

    Raises:
        FileNotFoundError: Caso a imagem não seja encontrada.

    Returns:
        numpy.ndarray: Imagem lida em formato BGR.
    """
    img = cv2.imread(pathImage)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {pathImage}", end='\n\n')
    return img

def prepareWindow(image):
    """
    Cria e ajusta uma janela OpenCV para exibir a imagem.

    Args:
        image (numpy.ndarray): Imagem a ser exibida.

    Returns:
        numpy.ndarray: Mesma imagem recebida.
    """
    cv2.namedWindow('Imagem', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Imagem', image.shape[1], image.shape[0])
    return image

def resizeImageToFitScreen(image, width=800, height=600):
    """
    Redimensiona a imagem para caber melhor na tela.

    Args:
        image (numpy.ndarray): Imagem original.
        width (int, optional): Largura desejada. Padrão: 800.
        height (int, optional): Altura desejada. Padrão: 600.

    Returns:
        numpy.ndarray: Imagem redimensionada.
    """
    return cv2.resize(image, (width, height))

def showImage(image):
    """
    Exibe a imagem em uma janela OpenCV e aguarda até uma tecla ser pressionada.

    Args:
        image (numpy.ndarray): Imagem a ser exibida.
    """
    cv2.imshow('Imagem', image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imageToString(image, lang, config_tesseract):
    """
    Extrai texto da imagem usando Tesseract.

    Args:
        image (numpy.ndarray | PIL.Image.Image): Imagem de entrada.
        lang (str): Código do idioma (ex: 'por').
        config_tesseract (str): Configurações Tesseract (ex: '--psm 6').

    Returns:
        str: Texto extraído.
    """
    return pytesseract.image_to_string(image, lang=lang, config=config_tesseract)

def useNeuralNetwork(img, w, h):
    """
    Executa a rede neural EAST para detecção de regiões de texto em uma imagem.

    A rede EAST possui duas saídas:
      - feature_fusion/Conv_7/Sigmoid: mapa de confiança (se há texto).
      - feature_fusion/concat_3: mapa de geometria (tamanho e ângulo das caixas).

    Args:
        img (numpy.ndarray): Imagem de entrada (BGR).
        w (int): Largura para redimensionar a imagem antes da inferência.
        h (int): Altura para redimensionar a imagem antes da inferência.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: scores (confiança) e geometry (dados de caixas) retornados pela rede.
    """
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), swapRB=True, crop=False)
    # print(f'blob.shape - {blob.shape}')
    detector = r'src\resources\models\frozen_east_text_detection.pb'
    layerNames = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
    neuralNetwork = cv2.dnn.readNet(detector)
    neuralNetwork.setInput(blob)
    scores, geometry = neuralNetwork.forward(layerNames)
    # print(f'scores {scores.shape[2:4]}, geometry{geometry.shape}')
    return scores, geometry

def generateTextBoundingBoxes(lines, scores, geometry, columns, minimumTrust, trust, boxs):
    """
    Gera bounding boxes (caixas delimitadoras) para regiões de texto com base nos mapas
    de confiança e geometria retornados pela rede EAST, aplicando non-max suppression
    para remover sobreposições redundantes.

    Args:
        lines (int): Número de linhas do mapa de pontuação.
        scores (numpy.ndarray): Mapa de pontuação da rede EAST.
        geometry (numpy.ndarray): Mapa de geometria da rede EAST.
        columns (int): Número de colunas do mapa de pontuação.
        minimumTrust (float): Valor mínimo de confiança para considerar uma caixa.
        trust (list[float]): Lista onde serão acumuladas as confianças válidas.
        boxs (list[tuple[int, int, int, int]]): Lista onde serão acumuladas as caixas detectadas.

    Returns:
        tuple[int, int, int, int, numpy.ndarray]: Coordenadas startX, startY, endX, endY e lista final de caixas detectadas.
    """
    for y in range(0, lines):
        dataScores = scores[0, 0, y]
        dataAngle, xData0, xData1, xData2, xData3 = geometryData(geometry, y)
        for x in range(0, columns):
            if dataScores[x] < minimumTrust:
                continue
            startX, startY, endX, endY = geometryCalculations(dataAngle, xData0, xData1, xData2, xData3, x, y)
            trust.append(dataScores[x])
            boxs.append((startX, startY, endX, endY))
    
    # print(f'trust - {trust}')
    # print(f'boxs - {boxs}')
    detections = non_max_suppression(np.array(boxs), probs=trust)
    # print(f'detections - {detections}')
    return startX, startY, endX, endY, detections

def extractAndDrawROI(bkpImg, detections, proportionW, proportionH, startX, startY, endX, endY, margin):
    """
    Extrai e desenha as regiões de interesse (ROI) com base nas detecções geradas pela rede EAST.

    As coordenadas das caixas são ajustadas para o tamanho original da imagem, e cada ROI é
    destacada com um retângulo verde.

    Args:
        bkpImg (numpy.ndarray): Imagem original onde as caixas serão desenhadas.
        detections (numpy.ndarray): Lista de caixas detectadas após non-max suppression.
        proportionW (float): Proporção entre largura original e redimensionada.
        proportionH (float): Proporção entre altura original e redimensionada.
        startX (int): Coordenada inicial X da última caixa processada.
        startY (int): Coordenada inicial Y da última caixa processada.
        endX (int): Coordenada final X da última caixa processada.
        endY (int): Coordenada final Y da última caixa processada.
        margin (int): Margem extra em torno da ROI.

    Returns:
        numpy.ndarray: Última região de interesse extraída e redimensionada.
    """
    imageFromAlter = bkpImg.copy()
    for (startX, startY, endX, endY) in detections:
        startX = int(startX * proportionW)
        startY = int(startY * proportionH)
        endX = int(endX * proportionW)
        endY = int(endY * proportionH)
        regionOfInterest = imageFromAlter[startY - margin:endY + margin, startX - margin:endX + margin]
        cv2.rectangle(bkpImg, (startX - margin, startY - margin), (endX + margin, endY + margin), (0, 255, 0), 2)
    # showImage(bkpImg)
    regionOfInterest = cv2.resize(regionOfInterest, None, fx= 1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # showImage(regionOfInterest)
    return regionOfInterest

def geometryData(geometry, y):
    """
    Extrai os dados geométricos para uma linha específica do mapa de geometria.

    Args:
        geometry (numpy.ndarray): Mapa de geometria retornado pela rede EAST.
        y (int): Índice da linha desejada.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        Dados de ângulo e deslocamento (xData0, xData1, xData2, xData3) para a linha.
    """
    xData0 = geometry[0, 0 , y]
    xData1 = geometry[0, 1 , y]
    xData2 = geometry[0, 2 , y]
    xData3 = geometry[0, 3 , y]
    angleData = geometry[0, 4, y]
    return angleData, xData0, xData1, xData2, xData3

def geometryCalculations(angleData, xData0, xData1, xData2, xData3, x, y):
    """
    Calcula as coordenadas de início e fim (bounding box) para uma determinada posição
    no mapa de geometria.

    Args:
        angleData (numpy.ndarray): Dados de ângulo para a linha atual.
        xData0, xData1, xData2, xData3 (numpy.ndarray): Dados de deslocamento para cada lado da caixa.
        x (int): Coluna atual no mapa de geometria.
        y (int): Linha atual no mapa de geometria.

    Returns:
        tuple[int, int, int, int]: Coordenadas startX, startY, endX, endY da caixa calculada.
    """
    (offsetX, offsetY) = (x * 4.0, y * 4.0)
    angle = angleData[x]
    cos = np.cos(angle)
    sin = np.sin(angle)
    h = xData0[x] + xData2[x]
    w = xData1[x] + xData3[x]

    endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
    startX = int(endX - w)
    startY = int(endY - h)

    return startX, startY, endX, endY