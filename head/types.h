#pragma once
#include <vector>
#include <type_traits>

// Концепт для арифметических типов (int, float, double и т.д.)
concept Arithmetic = std::is_arithmetic_v<T>;

// Основная структура точки ( для совместимости со старым кодом)
struct Point2D {
    float x;
    float y;
    float label;
};

// Полезный using для удобства
using Features = std::vector<float>;
