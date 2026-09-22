import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SolverConfig:
    K: int = 2
    p: int = 3
    batch: int = 128
    steps: int = 128
    learning_rate: float = 1e-2
    updates: int = 500
    clip_norm: float = 1.0
    validation_samples: int = 4096
    max_validation_samples: int = 16384
    max_rounds: int = 12
    max_seconds: float = 7200.
    residual_target: float = 4e-3
    max_steps: int = 4096

    def __post_init__(self):
        ints = (self.K, self.p, self.batch, self.steps, self.updates, self.validation_samples,
                self.max_validation_samples, self.max_rounds, self.max_steps)
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in ints):
            raise ValueError("Counts and degrees must be positive integers")
        if self.validation_samples < 2:
            raise ValueError("Paired validation needs at least two samples")
        if self.K > 8 or self.p > 15 or self.batch > 1024 or self.updates > 500:
            raise ValueError("v1 capacity/update limit exceeded")
        if self.steps > self.max_steps or self.validation_samples > self.max_validation_samples:
            raise ValueError("Invalid numerical or validation bounds")
        if self.max_validation_samples > 16384 or self.max_rounds > 12:
            raise ValueError("v1 validation/round limit exceeded")
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.learning_rate, self.clip_norm, self.max_seconds, self.residual_target)):
            raise ValueError("Positive finite optimization/budget values required")
        if self.max_seconds > 7200:
            raise ValueError("Instance budget cannot exceed two hours")

    def to_dict(self):
        return asdict(self)
