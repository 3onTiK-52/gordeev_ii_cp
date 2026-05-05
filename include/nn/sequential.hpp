#pragma once
#include "layers/layer.hpp"
#include <memory>
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
};

} // namespace nn
