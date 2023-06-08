import tensorflow as tf
import numpy as np
import enum


def softmax(x):
    """
    :param x: numpy array where each row corresponds to the scores (outputs) of the mode (e.g.: [[0.2, 0.6, 0.7], [0.1, 0.5, 0.4]]) 
    :param axis: axis where to compute values along (e.g., axis=1 computes the softmax along the second axis, i.e., the rows)
    :return: an array of the same shape as `x`. The result will sum to 1 along the specified axis.
    """
    x_max = np.amax(x, axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Language(enum.Enum):
    SPANISH = "es"
    ENGLISH = "en"
    GERMAN = "de"

    def __str__(self):
        return self.value


class LanguageDetector:

    LABELS = [Language.GERMAN, Language.ENGLISH, Language.SPANISH]
    
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = self._load_model(model_path)
        self.vectorizer = self._load_vectorizer(vectorizer_path)

    def _load_model(self, path: str):
        saved_model = tf.keras.models.load_model(path)
        return saved_model

    def _load_vectorizer(self, path: str):
        loaded_model = tf.keras.models.load_model(path)
        loaded_vectorizer = loaded_model.layers[0]
        return loaded_vectorizer

    def detect_language(self, text: str) -> Language:
        vectorized = self.vectorizer([text])
        print(text, vectorized)
        logits = self.model.predict(vectorized)
        lang_index = np.argmax(logits, axis=1)[0]
        lang = self.LABELS[lang_index]
        print(lang)
        return lang