import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

import json
from typing import Tuple, List

import cv2
import editdistance
from path import Path

from dataloader_iam import DataLoaderIAM, Batch
from model import Model, DecoderType
from preprocessor import Preprocessor

class FilePaths:
    """Filenames and paths to data."""
    fn_char_list = '../model/charList.txt'
    fn_summary = '../model/summary.json'
    fn_corpus = '../data/corpus.txt'


def get_img_height() -> int:
    """Fixed height for NN."""
    return 32


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """Height is fixed for NN, width is set according to training mode (single words or text lines)."""
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()


def write_summary(average_train_loss: List[float], char_error_rates: List[float], word_accuracies: List[float]) -> None:
    """Writes training summary file for NN."""
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'averageTrainLoss': average_train_loss, 'charErrorRates': char_error_rates, 'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def infer(model: Model, img: np.ndarray) -> Tuple[str, float]:
    """Recognizes text in the provided image."""
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    return recognized[0], probability[0]


class OCRDemo:
    def __init__(self):
        self.model = Model(char_list_from_file(), must_restore=True)
        self.window = tk.Tk()
        self.window.title("Démo OCR")

        # Créer un canevas pour le dessin
        self.canvas = tk.Canvas(self.window, width=400, height=300, bg="white")
        self.canvas.pack()

        # Lier les événements de souris au canevas
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<Button-1>", self.on_mouse_down)

        # Créer un bouton pour effectuer la prédiction OCR
        self.predict_button = tk.Button(self.window, text="Prédire", command=self.perform_ocr)
        self.predict_button.pack()

         # Créer un bouton pour effacer le canevas
        self.clear_button = tk.Button(self.window, text="Effacer", command=self.clear_canvas)
        self.clear_button.pack()

        # Créer un affichage pour les résultats
        self.result_label = tk.Label(self.window, text="Résultat :", font=("Arial", 24, "bold"))
        self.result_label.pack()

        self.image = np.ones((300, 400), dtype=np.uint8) * 255  # Image blanche pour dessiner
        self.is_drawing = False

    def on_mouse_down(self, event):
        self.is_drawing = True

    def on_mouse_drag(self, event):
        if self.is_drawing:
            x, y = int(event.x), int(event.y)
            self.canvas.create_oval(x, y, x + 8, y + 8, fill="black")
            self.image[y - 5: y + 5, x - 5: x + 5] = 0

    def perform_ocr(self):
        # Convertir l'image du canevas en une image PIL
        img_pil = Image.fromarray(self.image)

        # Redimensionner l'image à la taille requise pour l'OCR
        img_resized = img_pil.resize(get_img_size(), Image.ANTIALIAS)

        # Convertir l'image en niveaux de gris et en tableau NumPy
        img_gray = img_resized.convert("L")
        img_np = np.array(img_gray, dtype=np.uint8)

        # Appeler la fonction infer() avec le modèle et l'image
        recognized_text, probability = infer(self.model, img_np)

        # Mettre à jour l'affichage des résultats
        self.result_label.config(text="Résultat : {}\nProbabilité : {:.2f}%".format(recognized_text, probability * 100))

    def clear_canvas(self):
        # Effacer le canevas
        self.canvas.delete("all")
        self.image.fill(255)

    def run(self):
        self.window.mainloop()


# Créer et exécuter l'application de démonstration OCR
demo = OCRDemo()
demo.run()
