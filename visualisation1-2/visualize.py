import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn_core


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].astype(float).values
    y = df.iloc[:, -1].astype(int).values
    return X, y


def visualize(model_path, dataset_path, output="decision_boundary.png"):
    X, y = load_dataset(dataset_path)

    if X.shape[1] != 2:
        print("График decision boundary строится только для датасета с 2 признаками.")
        return

    model = nn_core.NeuralNetwork(2, 16)
    model.load(model_path)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = np.array([model.forward(row.tolist()) for row in grid])
    zz = probs.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, levels=50, alpha=0.6)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0", edgecolors="k")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1", edgecolors="k")

    plt.title("Decision boundary neural network")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)

    plt.savefig(output, dpi=150)
    plt.show()

    print("График сохранён:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.bin")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="decision_boundary.png")
    args = parser.parse_args()

    visualize(args.model, args.data, args.output)
