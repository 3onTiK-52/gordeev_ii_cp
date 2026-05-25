#pragma once
#include "layers/layer.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nn {

class Sequential {
private:
  std::vector<std::unique_ptr<Layer>> layers_;

public:
  void add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
  }

  // Проход вперед: передаем данные последовательно через все слои
  Matrix forward(Matrix input) {
    for (const auto &layer : layers_) {
      input = layer->forward(input);
    }
    return input;
  }

  // Обратный проход: передаем ошибку в обратном порядке
  Matrix backward(Matrix grad) {
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      grad = (*it)->backward(grad);
    }
    return grad;
  }

  // Обновление всех слоев
  void update(Optimizer &opt) {
    for (auto &layer : layers_) {
      layer->update(opt);
    }
  }

  // Сохранение всей модели в файл
  void save(const std::string &filepath) const {
    std::ofstream os(filepath, std::ios::binary);
    if (!os.is_open()) {
      throw std::runtime_error(
          "Не удалось открыть файл для сохранения модели: " + filepath);
    }

    for (const auto &layer : layers_) {
      layer->save(os);
    }
  }

  // Загрузка всей модели из файла
  void load(const std::string &filepath) {
    std::ifstream is(filepath, std::ios::binary);
    if (!is.is_open()) {
      throw std::runtime_error("Не удалось открыть файл для загрузки модели: " +
                               filepath);
    }

    for (auto &layer : layers_) {
      layer->load(is);
    }
  }
};

} // namespace nn
