import torch.cuda
from aifeynman import run_aifeynman


def main():
    # disables and checks PyTorch GPU support
    torch.cuda.is_available = lambda: False
    print(torch.cuda.is_available())
    # runs AI feynman on a generated sine wave
    run_aifeynman("./data/", "noisy_sine.dat", 30, "14ops.txt", polyfit_deg=3, NN_epochs=100)
    # runs AI feynman on a generated square wave
    # run_aifeynman("./data/", "noisy_sine.dat", 30, "14ops.txt", polyfit_deg=3, NN_epochs=100)


if __name__ == '__main__':
    main()
