#pragma once
#include "neuralnet.h"
#include "console.h"
#include <vector>

namespace Neural {
    namespace Trainer {

        template<typename T = float>
        T binaryCrossEntropy(T y_true, T y_pred) {
            const T eps = T(1e-7);
            y_pred = std::max(eps, std::min(T(1) - eps, y_pred));
            return -(y_true * std::log(y_pred) + (T(1) - y_true) * std::log(T(1) - y_pred));
        }

        template<typename T = float>
        void train(NeuralNetwork<T>& net,
            const std::vector<std::vector<T>>& features,
            const std::vector<T>& labels,
            int epochs = 500,
            T learningRate = 0.1f)
        {
            int n = features.size();

            for (int epoch = 0; epoch < epochs; ++epoch) {
                T totalLoss = 0;

                for (int i = 0; i < n; ++i) {
                    T y_pred = net.forward(features[i]);
                    T y_true = labels[i];
                    totalLoss += binaryCrossEntropy(y_true, y_pred);

                    int H = net.getHiddenSize();
                    const auto& a_h = net.getHiddenA();
                    T dz_o = y_pred - y_true;

                    auto w_ho = net.getWeightsHO();
                    T b_o = net.getBiasO();
                    for (int j = 0; j < H; ++j) {
                        w_ho[j] -= learningRate * dz_o * a_h[j];
                    }
                    b_o -= learningRate * dz_o;

                    auto w_ih = net.getWeightsIH();
                    auto b_h = net.getBiasesH();

                    for (int j = 0; j < H; ++j) {
                        T dz_h = dz_o * w_ho[j] * a_h[j] * (T(1) - a_h[j]);
                        for (int k = 0; k < net.getInputSize(); ++k) {
                            w_ih[j * net.getInputSize() + k] -= learningRate * dz_h * features[i][k];
                        }
                        b_h[j] -= learningRate * dz_h;
                    }

                    net.setWeightsHO(w_ho);
                    net.setBiasO(b_o);
                    net.setWeightsIH(w_ih);
                    net.setBiasesH(b_h);
                }

                totalLoss /= n;

                if (epoch % 50 == 0 || epoch == epochs - 1) {
                    Console::value("Epoch", epoch);
                    Console::value("Loss", (float)totalLoss);
                }
            }
        }

    } // namespace Trainer
} // namespace Neural