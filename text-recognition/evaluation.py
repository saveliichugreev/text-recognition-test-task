from pathlib import Path
from ocr_model import OCRModel
from Levenshtein import distance
import numpy as np
import json


def text_accuracy(src, recognized):
    dist = distance(src, recognized)
    return 1 - dist / max(len(src), len(recognized))


def evaluate_model(model: OCRModel, dataset_path: Path, ):
    with open(dataset_path / 'info.json') as f:
        dataset_info = json.load(f)
    accuracies = []
    for file_info in dataset_info:
        predicted_text = model.recognize_text(file_info['path'])
        accuracy = text_accuracy(file_info['text'], predicted_text)
        accuracies.append(accuracy)
    return np.mean(accuracies)
