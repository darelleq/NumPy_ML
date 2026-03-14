import numpy as np

from .._numpy_f32_constants import EPSILON, ONE, ZERO
from .._typealiases import Matrix

class CrossEntropyLoss:
    __slots__ = ('prob_true', 'prob_pred')

    def __init__(self, *, prob_true: Matrix, prob_pred: Matrix) -> None:
        if (pts := prob_true.shape) != (pps := prob_pred.shape):
            raise ValueError(f"Shape mismatch: prob_true: {pts} != prob_pred {pps}")

        # np.clip creates a copy
        self.prob_pred = np.clip(prob_pred, EPSILON(), ONE() - EPSILON())

        self.prob_true = np.clip(prob_true, ZERO(), ONE())

    @property
    def losses(self) -> Matrix:
        with np.errstate(under='ignore', over='raise', divide='raise', invalid='raise'):
            ln_softmax: Matrix = np.log(self.prob_pred)
            losses: Matrix = -np.sum(self.prob_true * ln_softmax, axis=1, keepdims=True)

        return losses

    @property
    def dcost_dpred(self) -> Matrix:

        return np.divide(np.negative(self.prob_true), self.prob_pred, dtype=np.float32)

    @property
    def cost(self) -> np.float32:

        return np.sum(self.losses)

    @property
    def avg_loss(self) -> np.float32:

        return np.mean(self.losses)
