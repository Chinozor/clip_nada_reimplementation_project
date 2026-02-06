# clip_nada_reimplementation_project

Адаптация StyleGAN2 (FFHQ лица) под новые домены с помощью CLIP (Directional loss) + выбор обучаемых слоёв через Global Loss.

Я надеюсь, что все ноутбуки запускаются без каких-либо дополнительных действий, по крайней мере я постарался чтобы это было так)

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

## Картинки
<img width="967" height="897" alt="image" src="https://github.com/user-attachments/assets/efe2c083-6da4-4154-abf0-eaa90ffb7b47" />
Сгенерированные фото
<img width="974" height="991" alt="image" src="https://github.com/user-attachments/assets/769d03f0-da7c-41d4-af83-621f127679cb" />
Реальные фото
