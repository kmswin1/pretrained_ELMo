import numpy as np


class EarlyStopping:
    """
    Early stopping
    patience: conter가 patience 값 이상으로 쌓이면 early stop
    min_delta: 최상의 score 보다 낮더라도 conter를 증가시키지 않도록 하는 margin 값
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = np.NINF

    def __call__(self, loss):
        score = -1 * loss
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            return False