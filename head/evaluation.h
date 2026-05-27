#pragma once
#include <vector>
#include "head/types.h"
#include "head/linear.h"

namespace Evaluation {

    // Точность линейной модели (y = kx + b)
    static float accuracyLinear(const std::vector<Point2D>& data, float k, float b) {
        int correct = 0;
        for (const auto& p : data) {
            int predicted = (p.y > k * p.x + b) ? 1 : 0;
            if (predicted == static_cast<int>(p.label)) {
                correct++;
            }
        }
        return static_cast<float>(correct) / data.size();
    }

} // namespace Evaluation
