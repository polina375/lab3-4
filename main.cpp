#include "console.h"
#include "neuralnet.h"
#include "trainer.h"
#include <vector>
#include <cstdlib>
#include <ctime>

int main()
{
    srand(42);   // фиксируем случайность, чтобы результаты повторялись
    Console::info("Laboratory Work 3-4: Neural Network");

    // ====================== Тестовые данные ======================
    std::vector<std::vector<float>> X = {
        {0.1f, 0.2f}, {0.2f, 0.1f}, {0.15f, 0.3f}, {0.3f, 0.2f},
        {0.8f, 0.9f}, {0.9f, 0.8f}, {0.75f, 0.95f}, {0.95f, 0.75f}
    };
    std::vector<float> y = { 0, 0, 0, 0, 1, 1, 1, 1 };

    Console::value("Количество примеров", (float)X.size());

    // ====================== Создаём сеть ======================
    Console::info("Создание нейронной сети...");
    Neural::NeuralNetwork<float> net(2, 8);   // 2 признака, 8 нейронов в скрытом слое

    // ====================== Обучение ======================
    Console::info("Обучение сети...");
    Neural::Trainer::train(net, X, y, 500, 0.1f);

    // ====================== Проверка ======================
    Console::info("Тестирование предсказаний...");

    std::vector<float> sample1 = { 0.1f, 0.15f };
    std::vector<float> sample2 = { 0.95f, 0.9f };

    Console::value("Предсказание sample1", net.predictClass(sample1));
    Console::value("Предсказание sample2", net.predictClass(sample2));

    // ====================== Сохранение и загрузка ======================
    Console::info("Сохранение модели...");
    net.save("model.bin");

    Console::info("Загрузка модели...");
    Neural::NeuralNetwork<float> loaded(2, 8);   // важно указать те же размеры!

    if (loaded.load("model.bin")) {
        Console::info("Модель успешно загружена");
        Console::value("Предсказание загруженной модели", loaded.predictClass(sample2));
    }
    else {
        Console::info("Ошибка загрузки модели");
    }

    Console::info("Программа завершена успешно!");
    return 0;
}