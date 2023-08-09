import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
from path import Path

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

def infer(model: Model, fn_img: Path) -> Tuple[str, float]:
    """Recognizes text in image provided by file path."""
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    return recognized[0], probability[0]


def create_ocr_gui():
    # Charger le modèle et autres opérations d'initialisation ici
    model = Model(char_list_from_file(), must_restore=True)

    # La fonction pour effectuer la prédiction OCR
    def perform_ocr():
        img_file_path = filedialog.askopenfilename(title="Sélectionner une image")
        if img_file_path:
            # Afficher l'image sélectionnée
            img = Image.open(img_file_path)
            img = img.resize((400, 300))  # Redimensionner l'image selon vos besoins
            img_thumbnail = ImageTk.PhotoImage(img)
            image_label.config(image=img_thumbnail)
            image_label.image = img_thumbnail

            # Appeler la fonction infer() avec le modèle et le chemin de l'image sélectionnée
            recognized_text, probability = infer(model, img_file_path)
            # Mettre à jour l'affichage des résultats
            update_result_text(recognized_text, probability)

    # Créer une fenêtre Tkinter
    window = tk.Tk()
    window.title("Démo OCR")

    # Créer un bouton pour sélectionner une image
    select_image_button = tk.Button(window, text="Sélectionner une image", command=perform_ocr, font=("Arial", 16))
    select_image_button.pack()

    # Créer un widget pour afficher l'image
    image_label = tk.Label(window)
    image_label.pack()

    # Créer un affichage pour les résultats
    result_label = tk.Label(window, text="Résultat :", font=("Arial", 24, "bold"))
    result_label.pack()

    # Fonction pour mettre à jour l'affichage des résultats
    def update_result_text(result, probability):
        result_label.config(text="Résultat : {}\nProbabilité : {:.2f}%".format(result, probability * 100), font=("Arial", 18))

    # Démarrer la boucle principale de l'interface utilisateur
    window.mainloop()


# Appeler la fonction pour créer et exécuter l'interface utilisateur
create_ocr_gui()
