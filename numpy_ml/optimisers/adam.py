import numpy as np
import uuid

from ..parameter import Parameter
from ..networks import NeuralNetwork
from .._typealiases import Tensor
from .._numpy_f32_constants import EPSILON, ONE

class AdamVelocities[T: Tensor]:
    __slots__ = ("param", "v", "s")

    def __init__(self, parameter: Parameter[T]) -> None:
        self.param: Parameter[T] = parameter
        self.v: T = np.zeros_like(parameter.values)
        self.s: T = np.zeros_like(parameter.values)

    @property
    def param_uuid(self) -> uuid.UUID:

        return self.param.uid

    def __repr__(self) -> str:

        return (
            f'AdamVelocities(param.name={self.param.name} ,'
            f'v.shape: {self.v.shape}, s.shape: {self.s.shape})'
        )

def ewa(*, value_t: Tensor, ewa_tm1: Tensor, beta: np.float32) -> Tensor:
    with np.errstate(under='ignore', over='raise', divide='raise', invalid='raise'):
        one_minus_beta: np.float32 = ONE() - beta
        ewa: Tensor = np.add(beta * ewa_tm1, one_minus_beta * value_t, dtype=np.float32)

    return ewa

def correct_ewa(*, ewa: Tensor, beta: np.float32, t: int) -> Tensor:
    with np.errstate(under='ignore', over='raise', divide='raise', invalid='raise'):
        inv_corr_factor: np.float32 = ONE() - beta ** t + EPSILON()
        ewa_corrected: Tensor = np.divide(ewa, inv_corr_factor, dtype=np.float32)

    return ewa_corrected

def compute_adam_delta(*, lr: np.float32, v_corr: Tensor, s_corr: Tensor) -> Tensor:
    with np.errstate(under='ignore', over='raise', divide='raise', invalid='raise'):
        delta: Tensor = np.divide(v_corr, np.sqrt(s_corr) + EPSILON(), dtype=np.float32)
        delta *= lr

    return delta


class Adam:
    __slots__ = ('network', 'lr', 't', 'beta1', 'beta2', 'velocities_dict')
    def __init__(
        self,
        network: NeuralNetwork,
        lr: float=0.001,
        *,
        beta1: float=0.9,
        beta2: float=0.999,
        t: int=0
    ) -> None:
        if lr <= 0:
            raise ValueError(f"Learning rate must be non-zero positive float, got {lr}")

        if t < 0:
            raise ValueError(f"t must be zero or positive int, got {t}")

        if not (0 < beta1 < 1) or not (0 < beta2 < 1):
            raise ValueError(
                "Adam betas must be in the exclusive range (0, 1), "
                f"got beta1: {beta1}, beta2: {beta2}"
            )

        self.network = network

        self.lr = np.float32(lr)

        self.t: int = t

        self.beta1: np.float32 = np.float32(beta1)

        self.beta2: np.float32 = np.float32(beta2)

        self.velocities_dict: dict[uuid.UUID, AdamVelocities[Tensor]] = dict()

        self._hook_network_parameters()

    def _hook_network_parameters(self) -> None:
        # WARNING: Dynamic parameter hooks post nn initialisation not implemented
        optiminfo: str = f"[Adam t={self.t}] "

        for depth, layer in enumerate(self.network.topology):
            layerinfo: str = f"[Layer {depth}, type={layer.__class__.__name__}] "

            for param in layer.parameters.values():
                paraminfo: str = f"[Parameter {param.name}, shape={param.shape}) "
                pid: uuid.UUID = param.uid

                if pid not in self.velocities_dict: # which it shouldn't be
                    self.velocities_dict[pid] = AdamVelocities(param) # i.e. always

                if layer.trainable is False:
                    # Note: Some Parameters are Non-Trainable e.g. in the case of Batchnorm
                    # But there should be no "Parameters" in a Non-Trainable Layer
                    raise Exception(f"{optiminfo}{layerinfo}{paraminfo} Parameter present in a Non-Trainable Layer")

    def step(self) -> None:
        self.t += 1

        optiminfo: str = f"[Adam t={self.t}] "

        for depth, layer in enumerate(self.network.topology):
            layerinfo: str = f"[Layer {depth}, type={layer.__class__.__name__}] "

            if layer.trainable is False:

                continue

            for param in layer.parameters.values():
                paraminfo: str = f"[Parameter {param.name}, shape={param.shape}) "

                pid: uuid.UUID = param.uid

                if param.grad is None:
                    raise ValueError(f"{optiminfo}{layerinfo}{paraminfo} Parameter.grad is None")

                gradinfo: str = f"[Parameter.grad.shape {param.grad.shape}]"

                grad: Tensor = param.grad

                velocities: AdamVelocities[Tensor] = self.velocities_dict[pid]

                v: Tensor = velocities.v

                s: Tensor = velocities.s

                try:
                    v = ewa(value_t=grad, ewa_tm1=v, beta=self.beta1)
                    s = ewa(value_t=np.square(grad), ewa_tm1=s, beta=self.beta2)

                    v_cor: Tensor = correct_ewa(ewa=v, beta=self.beta1, t=self.t)
                    s_cor: Tensor = correct_ewa(ewa=s, beta=self.beta2, t=self.t)

                    delta: Tensor = compute_adam_delta(lr=self.lr, v_corr=v_cor, s_corr=s_cor)

                    param.values = np.subtract(param.values, delta, dtype=np.float32)

                    velocities.v = v
                    velocities.s = s

                except Exception as e:
                    raise Exception(
                        f"{e}\n"
                        f"\t{optiminfo}{layerinfo}{paraminfo}{gradinfo}\n"
                        f"\t[velocities: {repr(velocities)}"
                    )
