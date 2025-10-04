import cv2
import pytesseract
from src.services import tesseractUtils as t_utils

pytesseract.pytesseract.tesseract_cmd = r"C:\Development Environment\Tesseract-OCR\tesseract.exe"

def main():
    width, height = 320, 320
    min_confidence = 0.9
    image_path = r'src\resources\img\caneca.jpg'

    print("[INFO] Carregando e preparando imagem...")
    original, backup, resized, prop_w, prop_h = load_and_prepare_image(image_path, width, height)
    print(f"[INFO] Original shape: {original.shape}")

    print("[INFO] Executando rede neural EAST...")
    scores, geometry = t_utils.useNeuralNetwork(resized, resized.shape[1], resized.shape[0])

    lines, columns = scores.shape[2:4]
    boxes, confidences = [], []
    startX, startY, endX, endY, detections = t_utils.generateTextBoundingBoxes(
        lines, scores, geometry, columns, min_confidence, confidences, boxes
    )

    if len(detections) == 0:
        print("[WARN] Nenhuma região de texto detectada.")
        return

    roi = t_utils.extractAndDrawROI(backup, detections, prop_w, prop_h, startX, startY, endX, endY, 5)
    if roi is None or roi.size == 0:
        print("[WARN] ROI vazia.")
        return

    config = '--tessdata-dir tessdata'
    text = t_utils.imageToString(roi, lang='por', config_tesseract=config)
    print("[INFO] Texto reconhecido:")
    print(text)


def load_and_prepare_image(path, target_width, target_height):
    img = t_utils.readImageWithOpenCV(path)
    bkp = img.copy()
    prop_h = img.shape[0] / float(target_height)
    prop_w = img.shape[1] / float(target_width)
    resized = t_utils.resizeImageToFitScreen(img, target_width, target_height)
    return img, bkp, resized, prop_w, prop_h


if __name__ == "__main__":
    main()