# Arbitrary Style Transfer

Реализация алгоритма произвольной передачи стиля с использованием метода Adaptive Instance Normalization (AdaIN) на базе PyTorch. Этот проект позволяет переносить стиль с одного изображения на другое в режиме реального времени с использованием Streamlit для демонстрации.

> **Базовая идея:**  
> Алгоритм основан на работе [Huang, X. & Belongie, S. (2017). *Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization*. ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html).

![Входное изображение](https://github.com/ElijahHoff/arbitrary_style_transfer/blob/main/test_results/image_2.jpg)
![Стиль](https://github.com/ElijahHoff/arbitrary_style_transfer/blob/main/test_results/image_1.jpg)
![Пример результата]([path/to/example.png](https://github.com/ElijahHoff/arbitrary_style_transfer/blob/main/test_results/result_2_to_1.jpg))

---

## Оглавление

- [Обзор](#обзор)
- [Особенности](#особенности)
- [Требования](#требования)
- [Установка](#установка)
- [Использование](#использование)
- [Примеры работы](#примеры-работы)
- [Цитирование](#цитирование)
- [Лицензия](#лицензия)

---

## Обзор

Данный проект предоставляет инструмент для переноса стиля, позволяющий объединить содержимое одного изображения и стиль другого. Основой является механизм адаптивной нормировки признаков (AdaIN), который изменяет статистику (среднее и стандартное отклонение) признаков содержимого в соответствии с признаками стиля.

Приложение разработано с использованием Streamlit, что позволяет удобно демонстрировать работу алгоритма через веб-интерфейс.

---

## Особенности

- **Произвольный перенос стиля.** Можно использовать любые изображения для получения новых композиций.
- **Работа в реальном времени.** Оптимизирован для быстрого выполнения.
- **Удобный веб-интерфейс.** Приложение на базе Streamlit обеспечивает простое и интуитивно понятное управление.

---

## Требования

- **Python:** 3.7+
- **PyTorch:** 1.4+ (рекомендуется использовать последнюю стабильную версию)
- **Дополнительные библиотеки:**  
  - torchvision  
  - numpy  
  - Pillow (PIL)  
  - Streamlit

Все зависимости перечислены в файле [requirements.txt](requirements.txt).

---

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/ElijahHoff/arbitrary_style_transfer.git
   cd arbitrary_style_transfer
2. **Установите необходимые пакеты:**

   pip install -r requirements.txt
   
## Использование

  Запустите приложение с помощью Streamlit. Для этого выполните следующую команду в терминале:
  streamlit run streamlit_app.py

## Примеры работы
В папке test_results находятся примеры работы алгоритма, где показаны результаты переноса стиля на различных изображениях. Ознакомьтесь с ними для более подробного понимания возможностей проекта.

## Цитирование

Если вы используете данный код в своей работе, просим сослаться на следующую публикацию:
```bibtex
@inproceedings{huang2017arbitrary,
  title={Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization},
  author={Huang, Xun and Belongie, Serge},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}


