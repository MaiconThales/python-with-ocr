import pytesseract
import numpy as np
import cv2 # OpenCV
import re
from PIL import Image, ImageFont, ImageDraw
from pytesseract import Output

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

def readImageWithPIL(pathImage):
    """
    Lê uma imagem usando Pillow (PIL).

    Args:
        pathImage (str): Caminho do arquivo de imagem.

    Returns:
        PIL.Image.Image: Objeto de imagem PIL.
    """
    img = Image.open(pathImage)
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

def resizeImageToFitScreen(image, max_width=800, max_height=600):
    """
    Redimensiona a imagem para caber dentro de uma área máxima, mantendo proporção.

    Args:
        image (numpy.ndarray): Imagem de entrada.
        max_width (int): Largura máxima.
        max_height (int): Altura máxima.

    Returns:
        numpy.ndarray: Imagem redimensionada.
    """
    h, w = image.shape[:2]
    scale = min(max_width/w, max_height/h, 1)
    new_dim = (int(w*scale), int(h*scale))
    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

def showImage(image):
    """
    Exibe a imagem em uma janela OpenCV e aguarda até uma tecla ser pressionada.

    Args:
        image (numpy.ndarray): Imagem a ser exibida.
    """
    cv2.imshow('Imagem', image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def textBox(result, img, i, color = (255, 100, 0)):
    """
    Desenha um retângulo em volta de um texto identificado pelo Tesseract.

    Args:
        result (dict): Resultado da função image_to_data do Tesseract.
        img (numpy.ndarray): Imagem onde desenhar.
        i (int): Índice do texto no resultado.
        color (tuple[int, int, int]): Cor do retângulo (B, G, R).

    Returns:
        tuple[int, int, numpy.ndarray]: Coordenadas x, y do retângulo e a imagem modificada.
    """
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    # cv2.putText(img, result['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 225))
    return x, y, img

def writeText(text, x, y, img, font, sizeFont, fill):
    """
    Escreve texto sobre a imagem usando Pillow.

    Args:
        text (str): Texto a escrever.
        x (int): Posição horizontal.
        y (int): Posição vertical.
        img (numpy.ndarray): Imagem em que escrever.
        font (str): Caminho para arquivo de fonte TTF.
        sizeFont (int): Tamanho da fonte.
        fill (tuple[int, int, int]): Cor do texto (R, G, B).

    Returns:
        numpy.ndarray: Imagem com o texto desenhado.
    """
    font = ImageFont.truetype(font, sizeFont)
    imgPil = Image.fromarray(img)
    draw = ImageDraw.Draw(imgPil)

    pos = (x, y - sizeFont)

    draw.text(pos, text, font=font, fill=fill)
    return np.array(imgPil)

def travelImage(img, result, minConfi, type):
    """
    Percorre os resultados do Tesseract e desenha caixas e textos conforme confiança mínima.

    Args:
        img (numpy.ndarray): Imagem original.
        result (dict): Resultado da função image_to_data do Tesseract.
        minConfi (int): Confiança mínima para considerar o texto.
        type (int): Tipo de processamento (1 ou 2).

    Returns:
        tuple[list[str], numpy.ndarray]: Lista de textos extraídos e imagem anotada.
    """
    font = 'fontes\calibri.ttf'
    data = []
    imgCopy = img.copy()
    for i in range(0, len(result['text'])):
        trust = int(result['conf'][i])
        text = result['text'][i]
        if trust > minConfi:
            if not text.isspace() and len(text) > 0:
                match type:
                    case 1:
                        x, y, img = textBox(result, imgCopy, i)
                        imgCopy = writeText(text, x, y, imgCopy, font, 15, (255, 0, 0))
                    case 2:
                        value = findWithRegex(result['text'][i], r"\b([0-3]?[0-9])/([0-1]?[0-9])/([0-9]{4})\b")
                        if value:
                            x, y, img = textBox(result, imgCopy, i, (0, 255, 0))
                            imgCopy = writeText(text, x, y, imgCopy, font, 15, (0, 255, 0))
                            data.append(text)
                        else:
                            x, y, img = textBox(result, imgCopy, i)
    return data, imgCopy

def findWithRegex(text, regex):
    """
    Procura correspondência usando expressão regular.

    Args:
        text (str): Texto a ser analisado.
        regex (str): Expressão regular.

    Returns:
        re.Match | None: Objeto de correspondência ou None.
    """
    return re.match(regex, text)

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

def imageToOsd(image):
    """
    Retorna informações de orientação e layout da imagem usando Tesseract OSD.

    Args:
        image (numpy.ndarray | PIL.Image.Image): Imagem de entrada.

    Returns:
        str: Informações OSD da imagem.
    """
    return pytesseract.image_to_osd(image)

def imageToData(image, lang, config_tesseract):
    """
    Extrai dados detalhados da imagem usando Tesseract (caixas, confiança, etc.).

    Args:
        image (numpy.ndarray | PIL.Image.Image): Imagem de entrada.
        lang (str): Código do idioma.
        config_tesseract (str): Configurações Tesseract.

    Returns:
        dict: Resultado no formato Output.DICT do pytesseract.
    """
    return pytesseract.image_to_data(
        image, 
        lang=lang, 
        config=config_tesseract, 
        output_type=Output.DICT
    )