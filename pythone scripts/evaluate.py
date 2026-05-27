import argparse
import pandas as pd
import nn_core
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].astype(float).values.tolist()
    y = df.iloc[:, -1].astype(int).values.tolist()
    return X, y


def evaluate(model_path, dataset_path):
    X, y = load_dataset(dataset_path)

    model = nn_core.NeuralNetwork(len(X[0]), 64)
    model.load(model_path)

    preds = [model.predict_class(row) for row in X]

    print("Dataset:", dataset_path)
    print("Model:", model_path)
    print("Accuracy:", accuracy_score(y, preds))
    print("F1:", f1_score(y, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification report:")
    print(classification_report(y, preds))

    return f1_score(y, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default="model.bin")
    parser.add_argument("data", required=True)
    args = parser.parse_args()

    evaluate(args.model, args.data)