import re
import matplotlib.pyplot as plt
import argparse

def plot_training_logs(log_file_path):
    epochs = []
    accuracies = []

    # Регулярное выражение
    pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+[0-9.]+\s+\|\s+Train Accuracy:\s+([0-9.]+)%")

    # Читаем лог-файл
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    epochs.append(int(match.group(1)))
                    accuracies.append(float(match.group(2)))
    except FileNotFoundError:
        print(f"Ошибка: Файл '{log_file_path}' не найден.")
        return

    if not epochs:
        print("В файле не найдено данных об эпохах. Проверьте формат логов.")
        return

    # Настройка одиночного графика
    plt.figure(figsize=(10, 6))

    # Рисуем Точность (Accuracy)
    plt.plot(epochs, accuracies, color='tab:blue', marker='o', linewidth=2.5, markersize=8)
    
    # Оформление
    plt.xlabel('Эпоха (Epoch)', fontsize=12, fontweight='bold')
    plt.ylabel('Точность / Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Динамика точности обучения (Training Accuracy)', fontsize=16, pad=15)
    
    # Сетка и деления (чтобы каждая эпоха была подписана)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(epochs) 
    
    # Сохраняем и показываем
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    print("График успешно сохранен как 'learning_curve.png'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Построение графика точности из логов.")
    parser.add_argument("log_file", nargs="?", default="training_log.txt", help="Путь к файлу с логами")
    args = parser.parse_args()
    
    plot_training_logs(args.log_file)
