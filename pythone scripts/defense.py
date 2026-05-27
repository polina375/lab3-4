import argparse
import pandas as pd
import nn_core
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].astype(float).values.tolist()
    y = df.iloc[:, -1].astype(int).values.tolist()
    return X, y


def defense_train(dataset_path, model_out="defense_model.bin"):
    X, y = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = nn_core.NeuralNetwork(len(X[0]), 64)

    nn_core.train_model(
        model,
        X_train,
        y_train,
        epochs=3000,
        learning_rate=0.03
    )

    preds = [model.predict_class(row) for row in X_test]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("DEFENSE RESULT ")
    print("Dataset:", dataset_path)
    print("Accuracy:", acc)
    print("F1:", f1)

    model.save(model_out)
    print("Модель сохранена:", model_out)
    return f1


if __name__ == "__main__":
    f1_d1 = defense_train("dataset1.csv", "dataset1_weights.bin")

    f1_d2 = defense_train("dataset2.csv", "dataset2_weights.bin")

    total_score = 0.5 * f1_d1 + 0.5 * f1_d2

    print("F1 d1:", f1_d1)
    print("F1 d2:", f1_d2)
    print("Final score:", total_score)