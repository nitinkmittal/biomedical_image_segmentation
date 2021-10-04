from torch import nn


def activations(act: str):
    return nn.ModuleDict(
        [
            ["relu", nn.ReLU()],
            ["lrelu", nn.LeakyReLU()],
            ["tanh", nn.Tanh()],
            ["identity", nn.Identity(requires_grad=False)],
        ]
    )[act]


def norm2d(norm: str, num_features: int):
    return nn.ModuleDict(
        [
            ["identity", nn.Identity(require_grad=False)],
            ["batchnorm", nn.BatchNorm2d(num_features=num_features)],
            [
                "instancenorm",
                nn.InstanceNorm2d(num_features=num_features),
            ],
        ]
    )[norm]
