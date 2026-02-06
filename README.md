# clip_nada_reimplementation_project

Адаптация StyleGAN2 (FFHQ лица) под новые домены с помощью CLIP (Directional loss) + выбор обучаемых слоёв через Global Loss.

## Куда смотреть
- **Обучение (entrypoint):** [`train.py`](./train.py)
- **Лоссы:** [`losses.py`](./losses.py)
- **Выбор/разморозка слоёв:** [`freezing.py`](./freezing.py)
- **Конфиги экспериментов:** [`configs/`](./configs)
- 
## Ноутбуки:
- Ноутбук с запуском обучения: ['train_notebook.ipynb'](./train_notebook.ipynb)
- Ноутбук с картинками: ['results.ipynb'](./results.ipynb)

## Финальный отчёт
- [`REPORT.pdf`](./final_report.pdf)
