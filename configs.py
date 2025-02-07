import dataclasses
class OptimizerConfig:
    """Base class for optimizer configurations."""
    def __init__(self, learning_rate, weight_decay, ranges):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ranges = ranges

class AdamConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 0.0,
            ranges = {
                "lr_range": [1e-5, 1e-2],
                "beta1_range": [0.7, 0.99],
                "beta2_range": [0.8, 0.9999],
                "weight_decay_range": [1e-6, 1e-2]
            }
        )
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def __str__(self):
        return f" lr:{self.learning_rate}, beta1:{self.beta1}, beta2:{self.beta2}, eps:{self.eps}, weight_decay:{self.weight_decay}"

class AdamWConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 1e-2,
            ranges={
                "lr_range": [1e-5, 1e-2],
                "beta1_range": [0.7, 0.99],
                "beta2_range": [0.8, 0.9999],
                "weight_decay_range": [1e-6, 1e-1]
            }
        )
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def __str__(self):
        return f" lr:{self.learning_rate}, beta1:{self.beta1}, beta2:{self.beta2}, eps:{self.eps}, weight_decay:{self.weight_decay}"


class RMSPropConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 0.0,
            ranges={
                "lr_range": [1e-5, 1e-2],
                "alpha_range": [0.8, 0.999],
                "momentum_range": [0.0, 0.9],
                "weight_decay_range": [1e-6, 1e-2]
            }
        )
        self.alpha = 0.99
        self.eps = 1e-8
        self.momentum = 0.0

    def __str__(self):
        return f" lr:{self.learning_rate}, alpha:{self.alpha}, eps:{self.eps}, momentum:{self.momentum}, weight_decay:{self.weight_decay}"


class SAMConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-2,
            weight_decay = 0.0,
            ranges={
                "lr_range": [1e-4, 1e-1],
                "momentum_range": [0.0, 0.99],
                "rho_range": [0.01, 0.3],
                "weight_decay_range": [1e-6, 1e-2]
            }
        )
        self.momentum = 0.9
        self.rho = 0.05

    def __str__(self):
        return f" lr:{self.learning_rate}, momentum:{self.momentum}, rho:{self.rho}, weight_decay:{self.weight_decay}"


class LAMBConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 0.01,
            ranges={
                "lr_range": [1e-5, 1e-2],
                "beta1_range": [0.7, 0.99],
                "beta2_range": [0.8, 0.9999],
                "weight_decay_range": [1e-6, 1e-2]
            }
        )
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def __str__(self):
        return f" lr:{self.learning_rate}, beta1:{self.beta1}, beta2:{self.beta2}, eps:{self.eps}, weight_decay:{self.weight_decay}"


class NovoGradConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 0.001,
            ranges={
                "lr_range": [1e-5, 1e-2],
                "beta1_range": [0.7, 0.99],
                "beta2_range": [0.8, 0.9999],
                "weight_decay_range": [1e-6, 1e-2]
            }
        )
        self.beta1 = 0.95
        self.beta2 = 0.98
        self.eps = 1e-8

    def __str__(self):
        return f" lr:{self.learning_rate}, beta1:{self.beta1}, beta2:{self.beta2}, eps:{self.eps}, weight_decay:{self.weight_decay}"


class AdoptConfig(OptimizerConfig):

    def __init__(self):
        super().__init__(
            learning_rate = 1e-3,
            weight_decay = 0.0,
            ranges = {
                "lr_range": [1e-5, 1e-2],
                "beta1_range": [0.7, 0.99],
                "beta2_range": [0.8, 0.9999],
                "weight_decay_range": [1e-6, 1e-2]
            }
        )
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def __str__(self):
        return f" lr:{self.learning_rate}, beta1:{self.beta1}, beta2:{self.beta2}, eps:{self.eps}, weight_decay:{self.weight_decay}"

class Config:

    def __init__(self, optimizer_name="adam", batch_size=64):

        self.optimizer = optimizer_name.lower()
        self.batch_size = batch_size
        self.seed = 42
        self.n_trials = 20
        self.timeout = 3600
        self.hpo_epochs = 3
        self.cv_epochs = 5
        self.cv_splits = 5
        self.epochs = 20

        self.optimizers = {
            "adam": AdamConfig(),
            "adamw": AdamWConfig(),
            "rmsprop": RMSPropConfig(),
            "sam": SAMConfig(),
            "lamb": LAMBConfig(),
            "novograd": NovoGradConfig(),
            "adopt": AdoptConfig()
        }

        if self.optimizer not in self.optimizers:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")

    def get_optimizer_config(self):
        return self.optimizers[self.optimizer]

