#pragma once
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "head/neuralnet.h"
#include "head/console.h"
#include "head/types.h"   

namespace Metrics {

    // Структура метрик 
    struct ClassificationMetrics {
        float accuracy = 0.0f;
        float precision = 0.0f;
        float recall = 0.0f;
        float f1 = 0.0f;
    };

    // 1. Вычисление метрик 
    ClassificationMetrics evaluate(Neural::NeuralNetwork<float>& net,
        const std::vector<std::vector<float>>& features,
        const std::vector<float>& labels)
    {
        int TP = 0, TN = 0, FP = 0, FN = 0;

        for (size_t i = 0; i < features.size(); ++i) {
            int true_label = static_cast<int>(labels[i]);
            int pred = net.predictClass(features[i]);

            if (true_label == 1 && pred == 1) TP++;
            else if (true_label == 0 && pred == 0) TN++;
            else if (true_label == 0 && pred == 1) FP++;
            else if (true_label == 1 && pred == 0) FN++;
        }

        float total = static_cast<float>(TP + TN + FP + FN);
        float accuracy = (total > 0) ? static_cast<float>(TP + TN) / total : 0.0f;
        float precision = (TP + FP > 0) ? static_cast<float>(TP) / (TP + FP) : 0.0f;
        float recall = (TP + FN > 0) ? static_cast<float>(TP) / (TP + FN) : 0.0f;
        float f1 = (precision + recall > 0.0f) ? 2.0f * precision * recall / (precision + recall) : 0.0f;

        return { accuracy, precision, recall, f1 };
    }

   
        // 2. Сохранение предсказаний в CSV

    bool savePredictions(Neural::NeuralNetwork<float>& net,
        const std::vector<std::vector<float>>& features,
        const std::vector<float>& labels,
        const char* filename)
    {
        std::ofstream file(filename);
        if (!file) {
            Console::info("Error: cannot open predictions file");
            return false;
        }

        file << "feature1,feature2,true_label,pred_label,probability\n";  

        for (size_t i = 0; i < features.size(); ++i) {
            float prob = net.forward(features[i]);
            int pred = net.predictClass(features[i]);
            file << features[i][0] << "," << features[i][1] << ","    // сейчас для 2 признаков
                << static_cast<int>(labels[i]) << ","
                << pred << "," << prob << "\n";
        }

        Console::info("Predictions saved to");
        Console::info(filename);
        return true;
    }

   // 3. Вывод Confusion Matrix

    void printConfusionMatrix(Neural::NeuralNetwork<float>& net,
        const std::vector<std::vector<float>>& features,
        const std::vector<float>& labels)
    {
        int TP = 0, TN = 0, FP = 0, FN = 0;

        for (size_t i = 0; i < features.size(); ++i) {
            int true_label = static_cast<int>(labels[i]);
            int pred = net.predictClass(features[i]);

            if (true_label == 1 && pred == 1) TP++;
            else if (true_label == 0 && pred == 0) TN++;
            else if (true_label == 0 && pred == 1) FP++;
            else if (true_label == 1 && pred == 0) FN++;
        }

        std::cout << "\n=== Confusion Matrix ===\n";
        std::cout << "          Predicted\n";
        std::cout << "          0     1\n";
        std::cout << "True  0   " << TN << "    " << FP << "\n";
        std::cout << "      1   " << FN << "    " << TP << "\n\n";
    }

} // namespace Metrics
