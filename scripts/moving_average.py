import torch
import numpy as np


def average(model_list):
    model_state = torch.load(model_list[0])

    for name in model_list[1:]:
        ms = torch.load(name)
        for param in model_state:
            model_state[param] += ms[param]

    for param in model_state:
        model_state[param] /= float(len(model_list))

    torch.save(model_state, "averaged.th")


if __name__ == "__main__":
    average(["model_state_epoch_1.2019-02-20-12-53-32.th",
             "model_state_epoch_1.2019-02-20-13-03-32.th",
             "model_state_epoch_1.2019-02-20-13-13-32.th",
             "model_state_epoch_1.2019-02-20-13-23-32.th",
             "model_state_epoch_1.2019-02-20-13-33-32.th"])