import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str, default="result.png")
args = parser.parse_args()

for input_path in args.input.split(","):
    data = np.load(input)
    plt.plot(data[0, :], data[1, :])
plt.savefig(args.output, dpi=1200)
