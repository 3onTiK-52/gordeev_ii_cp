#include <iostream>
// Проверяем подключение Eigen
#include <Eigen/Dense>
// Проверяем, что пути include настроены верно и файл находится
#include <nn/core/AnyMovable.h>

int main() {
  std::cout << "=== Neural Network Setup Successful! ===" << std::endl;

  // Создаем простую матрицу, чтобы убедиться, что Eigen слинковался
  Eigen::MatrixXf m(2, 2);
  m << 1.0f, 2.0f, 3.0f, 4.0f;

  std::cout << "Test Eigen Matrix:\n" << m << std::endl;
  std::cout << "C++ Standard: " << __cplusplus << std::endl;

  return 0;
}
