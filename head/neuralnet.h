#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "types.h"
#include "console.h"

namespace Neural {

    template<typename T = float>
    class NeuralNetwork {
    public:
        // inputSize - количество признаков в датасете
        // hiddenSize - количество нейронов в скрытом слое
        explicit NeuralNetwork(int inputSize = 2, int hiddenSize = 16);

        T forward(const std::vector<T>& features);
        int predictClass(const std::vector<T>& features, T threshold = T(0.5)) const;

        bool save(const std::string& filename) const;
        bool load(const std::string& filename);

        int getInputSize() const { return inputSize; }
        int getHiddenSize() const { return hiddenSize; }

        // Геттеры для Trainer
        const std::vector<T>& getHiddenA() const { return a_h; }
        const std::vector<T>& getWeightsHO() const { return w_ho; }
        const std::vector<T>& getWeightsIH() const { return w_ih; }
        const std::vector<T>& getBiasesH() const { return b_h; }
        T getBiasO() const { return b_o; }

        void setWeightsHO(const std::vector<T>& w) { w_ho = w; }
        void setWeightsIH(const std::vector<T>& w) { w_ih = w; }
        void setBiasesH(const std::vector<T>& b) { b_h = b; }
        void setBiasO(T b) { b_o = b; }

    private:
        void initWeights();

        int inputSize;
        int hiddenSize;

        std::vector<T> w_ih;   // inputSize * hiddenSize
        std::vector<T> b_h;    // hiddenSize
        std::vector<T> w_ho;   // hiddenSize
        T b_o;

        // Кэш
        std::vector<T> z_h, a_h;
        T z_o = 0, a_o = 0;

        T sigmoid(T x) const {
            return T(1) / (T(1) + std::exp(-x));
        }
    };

    // ===================== Реализация =====================

    template<typename T>
    NeuralNetwork<T>::NeuralNetwork(int inSize, int hSize)
        : inputSize(inSize), hiddenSize(hSize) {
        w_ih.resize(inputSize * hiddenSize);
        b_h.resize(hiddenSize);
        w_ho.resize(hiddenSize);
        initWeights();
    }

    template<typename T>
    void NeuralNetwork<T>::initWeights() {
        for (auto& w : w_ih) w = T(rand()) / RAND_MAX - T(0.5f);
        for (auto& b : b_h) b = T(rand()) / RAND_MAX - T(0.5f);
        for (auto& w : w_ho) w = T(rand()) / RAND_MAX - T(0.5f);
        b_o = T(rand()) / RAND_MAX - T(0.5f);
    }

    template<typename T>
    T NeuralNetwork<T>::forward(const std::vector<T>& features) {
        z_h.resize(hiddenSize);
        a_h.resize(hiddenSize);

        // Скрытый слой
        for (int i = 0; i < hiddenSize; ++i) {
            T sum = b_h[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += w_ih[i * inputSize + j] * features[j];
            }
            z_h[i] = sum;
            a_h[i] = sigmoid(z_h[i]);
        }

        // Выходной слой
        z_o = b_o;
        for (int i = 0; i < hiddenSize; ++i) {
            z_o += w_ho[i] * a_h[i];
        }
        a_o = sigmoid(z_o);
        return a_o;
    }

    template<typename T>
    int NeuralNetwork<T>::predictClass(const std::vector<T>& features, T threshold) const {
        std::vector<T> a(hiddenSize);
        for (int i = 0; i < hiddenSize; ++i) {
            T sum = b_h[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += w_ih[i * inputSize + j] * features[j];
            }
            a[i] = sigmoid(sum);
        }

        T z = b_o;
        for (int i = 0; i < hiddenSize; ++i) {
            z += w_ho[i] * a[i];
        }
        return sigmoid(z) >= threshold ? 1 : 0;
    }

    template<typename T>
    bool NeuralNetwork<T>::save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;

        file.write((const char*)&inputSize, sizeof(int));
        file.write((const char*)&hiddenSize, sizeof(int));
        file.write((const char*)w_ih.data(), w_ih.size() * sizeof(T));
        file.write((const char*)b_h.data(), b_h.size() * sizeof(T));
        file.write((const char*)w_ho.data(), w_ho.size() * sizeof(T));
        file.write((const char*)&b_o, sizeof(T));
        return true;
    }

    template<typename T>
    bool NeuralNetwork<T>::load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        file.read((char*)&inputSize, sizeof(int));
        file.read((char*)&hiddenSize, sizeof(int));

        w_ih.resize(inputSize * hiddenSize);
        b_h.resize(hiddenSize);
        w_ho.resize(hiddenSize);

        file.read((char*)w_ih.data(), w_ih.size() * sizeof(T));
        file.read((char*)b_h.data(), b_h.size() * sizeof(T));
        file.read((char*)w_ho.data(), w_ho.size() * sizeof(T));
        file.read((char*)&b_o, sizeof(T));
        return true;
    }

} // namespace Neural