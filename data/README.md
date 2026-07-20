# Данные

Сами файлы датасета в репозитории не лежат: RuMedBench не публикует лицензию, разрешающую
перевыкладывать данные, а записи медицинские. Здесь только ссылка и инструкция.

## Источник

**RuMedTop3** из бенчмарка **RuMedBench** — обезличенные записи приёмов, жалобы пациентов с
кодом диагноза МКБ-10.

https://github.com/sberbank-ai-lab/RuMedBench/tree/main/data/RuMedTop3

## Как получить

Скачать `train_v1.jsonl`, `dev_v1.jsonl`, `test_v1.jsonl` из репозитория выше и положить в эту
папку:

```
data/
├── train_v1.jsonl
├── dev_v1.jsonl
└── test_v1.jsonl
```

Перефразированный `test_para_v1.jsonl` генерируется из `test_v1.jsonl`:

```bash
python -m src.evaluation.robustness
```

Без данных обучение и тесты пропускаются, сервис на прод-модели (`MODEL=transformer`) работает —
веса LoRA-адаптера лежат в `models/rubioroberta_ft/`.

## Цитирование

```bibtex
@misc{blinov2022rumedbench,
  title  = {RuMedBench: A Russian Medical Language Understanding Benchmark},
  author = {Pavel Blinov and Arina Reshetnikova and Aleksandr Nesterov
            and Galina Zubkova and Vladimir Kokh},
  year   = {2022},
  eprint = {2201.06499}
}
```
