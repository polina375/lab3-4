import nn_core
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def csv_to_features_labels(filepath):
    df = pd.read_csv(filepath)

    features = df.iloc[:, :-1].values.astype(np.float32).tolist()
    labels = df.iloc[:, -1].values.astype(np.float32).tolist()

    return features, labels


def train_and_evaluate(filepath):
    X, y = csv_to_features_labels(filepath)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    input_size = len(X_train[0])
    hidden_size = 64

    net = nn_core.NeuralNetwork(input_size, hidden_size)

    nn_core.train_model(
        net,
        X_train,
        y_train,
        epochs=3000,
        learning_rate=0.03
    )

    preds = [net.predict_class(feat) for feat in X_test]
    f1 = f1_score(y_test, preds)

    net.save(filepath.replace(".csv", "_weights.bin"))

    return f1


def main():
    f1_d1 = train_and_evaluate("dataset1.csv")
    f1_d2 = train_and_evaluate("dataset2.csv")

    total_score = 0.5 * f1_d1 + 0.5 * f1_d2

    print(f"F1(d1) = {f1_d1:.4f}")
    print(f"F1(d2) = {f1_d2:.4f}")
    print(f"Final score = {total_score:.4f}")


if __name__ == "__main__":
    main()
