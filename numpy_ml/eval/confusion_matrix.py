import numpy as np

from .._typealiases import UInt8_Vector, Mask_Vector, Int64_Matrix, Shape2D

class ConfusionMatrix:
    __slots__ = ('true_labels', 'pred_labels', 'scorecard', 'array')

    def __init__(
        self,
        categories: int,
        *,
        true_labels: UInt8_Vector,
        pred_labels: UInt8_Vector,
        from_copies: bool = True
    ) -> None:
        if (tls := true_labels.shape) != (pls := pred_labels.shape):
            raise ValueError(f"True labels shape {tls} must equal Pred labels shape {pls}")

        self.true_labels: UInt8_Vector = true_labels
        self.pred_labels: UInt8_Vector = pred_labels
        if from_copies is True:
            self.true_labels = self.true_labels.copy()
            self.pred_labels = self.pred_labels.copy()

        self.scorecard: Mask_Vector = np.equal(self.true_labels, self.pred_labels)

        # True labels along rows; Predicted labels along columns
        self.array: Int64_Matrix = np.zeros((categories, categories), dtype=np.int64)

        for i in range(true_labels.size):
            self.array[true_labels[i], pred_labels[i]] += 1

        assert np.sum(self.array) == true_labels.size == pred_labels.size

    @property
    def accuracy(self) -> np.float64:
        return np.mean(self.scorecard, dtype=np.float64)

    @property
    def n_correct(self) -> np.int64:
        return np.sum(self.scorecard, dtype=np.int64)

    @property
    def n_incorrect(self) -> np.int64:
        return np.sum(~self.scorecard, dtype=np.int64)

    @property
    def shape(self) -> Shape2D:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype
