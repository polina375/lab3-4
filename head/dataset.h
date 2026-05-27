#pragma once
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include "types.h"
#include "console.h"   // если используешь Console

namespace Dataset {

    // Генерация синтетического датасета (для тестов)
    template<typename T>
    static std::vector<Point2D> generate(int n, T k, T b) {
        std::vector<Point2D> data;
        data.reserve(n);

        for (int i = 0; i < n; i++) {
            float x = -2.0f + (float)rand() / RAND_MAX * 4.0f;
            float y = -2.0f + (float)rand() / RAND_MAX * 4.0f;
            float label = (y > static_cast<float>(k) * x + static_cast<float>(b)) ? 1.0f : 0.0f;
            data.push_back({ x, y, label });
        }
        return data;
    }

    // Сохранение в CSV
    static bool saveCSV(const std::vector<Point2D>& data, const char* filename) {
        std::ofstream file(filename);
        if (!file) return false;

        file << "x,y,label\n";
        for (const auto& p : data) {
            file << p.x << "," << p.y << "," << p.label << "\n";
        }
        return true;
    }

    // Разбиение на train/test
    static std::pair<std::vector<Point2D>, std::vector<Point2D>>
        split(const std::vector<Point2D>& data, float trainRatio = 0.8f) {
        std::vector<Point2D> train, test;
        train.reserve(static_cast<size_t>(data.size() * trainRatio));
        test.reserve(data.size() - train.capacity());

        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < data.size(); ++i) indices[i] = i;

        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t train_size = static_cast<size_t>(data.size() * trainRatio);
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < train_size) {
                train.push_back(data[indices[i]]);
            }
            else {
                test.push_back(data[indices[i]]);
            }
        }
        return { train, test };
    }

} // namespace Dataset