cmake_minimum_required(VERSION 3.22)

project(Gordeev_NN_CP CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)

# Собираем все исходники из папки src
file(GLOB_RECURSE NN_SOURCES "src/*.cpp")

# Создаем библиотеку нашей нейросети
add_library(nn_lib ${NN_SOURCES})

# Указываем, что все заголовочные файлы лежат в папке include
target_include_directories(nn_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Линкуем Eigen к нашей библиотеке
target_link_libraries(nn_lib PUBLIC Eigen3::Eigen)

# Создаем исполняемый файл из main.cpp
add_executable(nn_app app/main.cpp)

# Подключаем нашу библиотеку к исполняемому файлу
target_link_libraries(nn_app PRIVATE nn_lib)

# --- Блок тестирования ---
# Включаем встроенную функцию скачивания из сети
include(FetchContent)

# Указываем, откуда скачать фреймворк Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

# Скачиваем и делаем его доступным для нашего проекта
FetchContent_MakeAvailable(googletest)

# Включаем панель тестирования в CMake
enable_testing()

# Говорим CMake, что у нас появилась папка с тестами
add_subdirectory(tests)