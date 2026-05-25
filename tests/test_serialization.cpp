#include <cstdio> // Для std::remove
#include <gtest/gtest.h>
#include <nn/core/activation_function.hpp>
#include <nn/layers/activation_layer.hpp>
#include <nn/layers/linear_layer.hpp>
#include <nn/sequential.hpp>

TEST(SerializationTest, SaveAndLoadProducesIdenticalOutputs) {
  std::string filename = "test_model.bin";

  // 1. Создаем Модель А (при создании инициализируются случайные веса)
  nn::Sequential model_a;
  model_a.add(std::make_unique<nn::LinearLayer>(2, 4));
  model_a.add(std::make_unique<nn::ActivationLayer>(nn::ReLU()));
  model_a.add(std::make_unique<nn::LinearLayer>(4, 1));

  // 2. Сохраняем Модель А на диск
  model_a.save(filename);

  // 3. Создаем Модель Б (с другими случайные веса)
  nn::Sequential model_b;
  model_b.add(std::make_unique<nn::LinearLayer>(2, 4));
  model_b.add(std::make_unique<nn::ActivationLayer>(nn::ReLU()));
  model_b.add(std::make_unique<nn::LinearLayer>(4, 1));

  // 4. Входные данные для проверки
  nn::Matrix X(2, 2);
  X << 1.0f, 0.5f, 0.2f, 1.5f;

  // 5. Убеждаемся, что до загрузки модели работают по-разному
  nn::Matrix pred_a = model_a.forward(X);
  nn::Matrix pred_b_before = model_b.forward(X);
  EXPECT_FALSE(pred_a.isApprox(pred_b_before));

  // 6. Загружаем веса из файла в Модель Б
  model_b.load(filename);

  // 7. Проверяем, что теперь Модель Б думает точно так же, как Модель А
  nn::Matrix pred_b_after = model_b.forward(X);
  EXPECT_TRUE(pred_a.isApprox(pred_b_after));

  // Удаляем за собой бинарник
  std::remove(filename.c_str());
}
