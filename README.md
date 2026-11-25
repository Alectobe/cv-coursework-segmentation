# Сегментация полипов на эндоскопических снимках (Kvasir-SEG)

Курсовой проект по компьютерному зрению, **уровень «на 4»**: готовая сеть (U-Net из
`segmentation_models_pytorch`) и её модификации, сравнение с исходной моделью, трекинг
экспериментов в ClearML, микросервис инференса, запуск обучения и сервиса в Docker Compose.

| | |
|---|---|
| **Задача** | бинарная сегментация: для каждого пикселя снимка колоноскопии определить, относится ли он к полипу |
| **Данные** | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/): 1000 снимков с масками полипов |
| **Исходная модель** | U-Net, энкодер ResNet-34 (ImageNet), функция потерь BCE ([smp 0.5.0](https://github.com/qubvel-org/segmentation_models.pytorch)) |
| **Модификации** | 1) Attention Gates на skip-соединениях; 2) функция потерь Dice + Focal; 3) блок ASPP в bottleneck |
| **Основная метрика** | mDice на валидации (Dice по каждому снимку, усреднённый); дополнительные: mIoU, precision, recall |
| **Трекер** | ClearML (облачный, app.clear.ml) |
| **Сервис** | FastAPI: `GET /health`, `POST /predict`, `POST /predict/mask`, `POST /predict/overlay` |

Отчёт с описанием метода, экспериментов и анализом ошибок: [REPORT.md](REPORT.md).

---

## Содержание

1. [Быстрый старт: сервис с готовыми весами](#1-быстрый-старт-сервис-с-готовыми-весами)
2. [API](#2-api)
3. [Требования к среде](#3-требования-к-среде)
4. [Данные](#4-данные)
5. [Модель и модификации](#5-модель-и-модификации)
6. [Полный цикл: подготовка данных, обучение, оценка](#6-полный-цикл-подготовка-данных-обучение-оценка)
7. [Трекер экспериментов (ClearML)](#7-трекер-экспериментов-clearml)
8. [Итоговая модель и связь с экспериментом](#8-итоговая-модель-и-связь-с-экспериментом)
9. [Структура репозитория](#9-структура-репозитория)
10. [Тесты](#10-тесты)
11. [Источники и заимствования](#11-источники-и-заимствования)

---

## 1. Быстрый старт: сервис с готовыми весами

Нужны только Docker и Docker Compose: ни Python, ни GPU, ни обучение не требуются.

```bash
git clone https://github.com/Alectobe/cv-coursework-segmentation.git
cd cv-coursework-segmentation
cp .env.example .env                 # ключи ClearML для сервиса не нужны

docker compose run --rm weights      # скачать итоговые веса в artifacts/final/model.pt
docker compose up -d api             # собрать образ и запустить сервис на http://localhost:8000
curl http://localhost:8000/health    # {"status":"ok", "model": {...}} - модель загружена
```

Сервис возвращает `200` на `/health` только после загрузки и прогрева модели, до этого `503`.
Docker-healthcheck контейнера использует тот же эндпоинт: `docker compose ps` покажет `healthy`.

Проверка на снимках, которые не участвовали в обучении (тестовая выборка):

```bash
docker compose run --rm examples     # нужны данные (раздел 6): 6 тестовых снимков -> examples/polyp_01..06.jpg
curl -F "file=@examples/polyp_03.jpg" http://localhost:8000/predict/overlay -o overlay.png
```

Остановка: `docker compose down`. Веса остаются в `artifacts/` на хосте.

> **Windows (PowerShell):** вместо `curl` пишите `curl.exe`, вместо `cp` - `copy`.

## 2. API

Интерактивная документация (Swagger UI): <http://localhost:8000/docs>. Там можно загрузить
снимок кнопкой «Try it out» и сразу увидеть ответ.

### `GET /health`

| Код | Когда | Тело |
|---|---|---|
| 200 | модель загружена и прогрета | `{"status": "ok", "model": {experiment, epoch, val_dice, clearml_task_id, clearml_task_url, checkpoint_sha256, ...}}` |
| 503 | весов нет или они не загрузились | `{"status": "unavailable", "detail": "<причина>"}` |

### `POST /predict`

Запрос: `multipart/form-data` с полем `file` (JPEG, PNG, BMP, TIFF, WebP; до 15 МБ).
Параметры query: `threshold` (0..1, по умолчанию 0.5 из чекпоинта), `include_overlay` (по умолчанию `true`).

```bash
curl -F "file=@examples/polyp_03.jpg" "http://localhost:8000/predict?threshold=0.5"
```

Ответ `200` (значения - пример):

```json
{
  "filename": "polyp_03.jpg",
  "width": 622,
  "height": 529,
  "threshold": 0.5,
  "polyp_found": true,
  "polyp_area_ratio": 0.117,
  "bbox": [212, 180, 401, 352],
  "mean_confidence": 0.93,
  "max_probability": 0.998,
  "mask_png_base64": "iVBORw0KGgo...",
  "overlay_png_base64": "iVBORw0KGgo...",
  "inference_ms": 180.4,
  "model": {"experiment": "...", "epoch": 0, "val_dice": 0.0, "image_size": 352, "default_threshold": 0.5,
            "clearml_task_id": "...", "clearml_task_url": "https://app.clear.ml/...", "checkpoint_sha256": "..."}
}
```

| Поле | Смысл |
|---|---|
| `mask_png_base64` | бинарная маска 0/255 **в исходном разрешении снимка**, PNG в base64 |
| `overlay_png_base64` | маска поверх снимка (зелёная заливка и контур), PNG в base64 |
| `polyp_found`, `polyp_area_ratio`, `bbox` | есть ли полип, доля площади кадра, рамка `[x_min, y_min, x_max, y_max]` |
| `mean_confidence`, `max_probability` | средняя вероятность внутри маски и максимальная по кадру |
| `model` | из какого эксперимента и чекпоинта получены веса |

### `POST /predict/mask` и `POST /predict/overlay`

Тот же запрос, но в ответе сразу PNG: так результат удобно посмотреть без `jq` и Python.

```bash
curl -F "file=@examples/polyp_03.jpg" http://localhost:8000/predict/mask -o mask.png
curl -F "file=@examples/polyp_03.jpg" http://localhost:8000/predict/overlay -o overlay.png
```

Если нужна маска из JSON-ответа (Linux/macOS с `jq`):

```bash
curl -s -F "file=@examples/polyp_03.jpg" http://localhost:8000/predict \
  | jq -r .mask_png_base64 | base64 -d > mask.png
```

### Ошибки

Все ошибки возвращаются в формате `{"detail": "<описание>"}`.

| Код | Когда |
|---|---|
| 400 | пустой файл; файл не является изображением или повреждён; сторона меньше 32 px |
| 413 | файл больше 15 МБ или изображение больше 40 Мп |
| 422 | нет поля `file`; `threshold` вне [0, 1] |
| 503 | модель не загружена (нет весов) |

```bash
curl -F "file=@README.md" http://localhost:8000/predict
# {"detail":"Файл не является корректным изображением (ожидается JPEG, PNG, BMP, TIFF или WebP)"}  -> HTTP 400
```

## 3. Требования к среде

**Сервис инференса:** Docker Engine 24+ (или Docker Desktop) с Docker Compose v2. GPU не нужен,
образ занимает около 1.7 ГБ.

**Обучение:**
- NVIDIA GPU (хватит 6 ГБ видеопамяти: batch 8, вход 352x352, смешанная точность);
- драйвер NVIDIA и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (Linux) или Docker Desktop с бэкендом WSL2 (Windows);
- версия CUDA в выводе `nvidia-smi` должна быть **не ниже 12.8**: образ по умолчанию собирается с
  `torch 2.8.0+cu128`, это RTX 20xx-50xx. Если драйвер старее или GPU поколения GTX 10xx, в `.env`
  укажите `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`;
- около 10 ГБ на диске под образ обучения (PyTorch с CUDA);
- доступ в интернет к `datasets.simula.no` (данные), `huggingface.co` (веса ResNet-34 ImageNet),
  `download.pytorch.org`, `pypi.nvidia.com` и `pypi.org` (сборка образа), `app.clear.ml` / `api.clear.ml`
  / `files.clear.ml` (трекер).

Проверка, что контейнер видит GPU:

```bash
docker compose run --rm --entrypoint python train -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Данные

**Источник:** Kvasir-SEG (Simula Research Laboratory; Jha et al., MMM 2020), официальный архив
`https://datasets.simula.no/downloads/kvasir-seg.zip` (46 МБ, SHA-256 `03b30e21...78f7`, проверяется
при загрузке). Лицензия разрешает использование в исследовательских и учебных целях, поэтому снимки
в репозитории не хранятся: в git лежит только **состав выборок**.

**Формат разметки:** для каждого снимка `images/<id>.jpg` есть маска `masks/<id>.jpg` того же размера
(белый - полип, чёрный - фон). Маски хранятся в JPEG, поэтому при загрузке бинаризуются по порогу 127.
В архиве есть и `kavsir_bboxes.json` с рамками, в проекте он не используется.

**Характеристики** (`data/splits/dataset_stats.json`):
- 1000 снимков, на каждом есть хотя бы один полип (пустых масок нет);
- 333 разных разрешения: от 332x352 до 1920x1072, в основном около 620x530;
- полип занимает от 0.5% до 81% кадра, медиана 11.4%: сильный дисбаланс «фон / полип».

**Разбиение** (`scripts/prepare_data.py`, seed 42), стратифицировано по размеру полипа:

| Выборка | Снимков | Маленькие (<5%) | Средние (5-20%) | Крупные (>20%) | Медиана площади |
|---|---|---|---|---|---|
| train | 800 | 160 | 428 | 212 | 11.8% |
| val | 100 | 20 | 53 | 27 | 9.9% |
| test | 100 | 20 | 53 | 27 | 11.6% |

**Дубликаты.** Скрипт ищет почти-дубликаты перцептивным хэшем (pHash 16x16 по центральным 60% кадра;
по всему кадру хэш бесполезен из-за одинаковой рамки эндоскопа). Найдено 4 пары: 3 побайтовые копии и
1 пара соседних кадров одной сцены (`data/splits/duplicates.csv`). Каждая пара целиком попадает в одну
выборку, поэтому пересечения train/val/test нет. Проверка: `python scripts/prepare_data.py` без `--force`
и `pytest tests/test_splits.py`.

**Зачем три выборки:** train - обучение; val - выбор лучшей эпохи и сравнение вариантов (все решения
принимаются только по ней); test - однократная итоговая оценка выбранной модели.

## 5. Модель и модификации

Исходная модель - готовый `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet")`.
Модификации встраиваются в его декодер (`src/polypseg/models/factory.py`); энкодер, данные, аугментации,
оптимизатор, расписание lr, число эпох и seed у всех вариантов **одинаковые**. Конфиги отличаются только
секциями `model` / `loss` (проверка: `diff configs/baseline.yaml configs/attn_gates.yaml`).

| Конфиг | Тип модификации (по заданию) | Что изменено | Цель | Параметров |
|---|---|---|---|---|
| `baseline.yaml` | - | исходный U-Net, BCE | точка отсчёта | 24.44 млн |
| `attn_gates.yaml` | способ объединения признаков | skip-признаки перед конкатенацией умножаются на карту внимания, посчитанную по сигналу с более грубого уровня декодера ([Attention U-Net](https://arxiv.org/abs/1804.03999)) | подавить фон, похожий на полип по текстуре (складки, блики), уменьшить ложные срабатывания | 24.80 млн |
| `dice_focal.yaml` | другая функция потерь | Dice + Focal (γ=2) вместо BCE, собственная реализация | ослабить доминирование пикселей фона, сфокусироваться на границах и сложных пикселях | 24.44 млн |
| `aspp.yaml` | архитектурный блок | ASPP (DeepLabv3) вместо пустого центрального блока декодера ([DeepLabv3](https://arxiv.org/abs/1706.05587)) | многомасштабный контекст: размер полипа меняется в 170 раз | 26.67 млн |
| `attn_dice_focal.yaml` | комбинация 1 + 2 | Attention Gates + Dice + Focal | проверить, складываются ли эффекты (запускать, если обе модификации улучшили val mDice) | 24.80 млн |

Общие параметры обучения (`configs/*.yaml`, секция `train`): вход 352x352, AdamW (lr 1e-4,
weight decay 1e-4), cosine-расписание до 1e-6, batch 8, 50 эпох, смешанная точность (AMP),
порог 0.5, seed 42. Аугментации (только train): отражения, повороты на 90°, лёгкое аффинное
преобразование, ColorJitter. Лучшая эпоха выбирается по val mDice.

## 6. Полный цикл: подготовка данных, обучение, оценка

Все команды выполняются из корня репозитория. Результаты сохраняются на хосте в смонтированных
каталогах (`data/`, `artifacts/`, `reports/`, `examples/`) и переживают удаление контейнеров.

```bash
cp .env.example .env                          # вписать ключи ClearML (раздел 7)
docker compose build train                    # образ обучения (один раз, ~10 ГБ)

# 1. Данные: скачать Kvasir-SEG и проверить зафиксированное разбиение
docker compose run --rm prepare

# 2. Обучение исходной модели и трёх модификаций (каждый запуск = отдельный эксперимент в ClearML)
docker compose run --rm train --config configs/baseline.yaml
docker compose run --rm train --config configs/attn_gates.yaml
docker compose run --rm train --config configs/dice_focal.yaml
docker compose run --rm train --config configs/aspp.yaml
# если и attn_gates, и dice_focal улучшили val mDice:
docker compose run --rm train --config configs/attn_dice_focal.yaml

# 3. Сравнение вариантов -> reports/experiments.md, reports/figures/
docker compose run --rm compare

# 4. Выбор итоговой модели по val mDice -> artifacts/final/model.pt + model_card.json
docker compose run --rm select

# 5. Итоговая оценка на test -> reports/test_eval/ (+ задача типа testing в ClearML)
docker compose run --rm evaluate

# 6. Сервис с полученными весами
docker compose up -d --build api
```

Что где появляется:

| Путь | Содержимое |
|---|---|
| `artifacts/experiments/<имя>/` | `best.pt` (веса лучшей эпохи), `history.csv` (метрики по эпохам), `curves.png`, `config.yaml`, `summary.json` (в т.ч. ID задачи и модели в ClearML) |
| `artifacts/final/` | `model.pt` - веса для сервиса, `model_card.json` - откуда они |
| `reports/experiments.md` | таблица сравнения вариантов с исходной моделью |
| `reports/test_eval/` | метрики на test, `per_image.csv`, лучшие и худшие примеры, Dice по размеру полипа |
| `artifacts/hf_cache/` | кэш весов ImageNet (скачиваются один раз) |

Дополнительно:
- оценка любого чекпоинта на val в исходном разрешении:
  `docker compose run --rm evaluate --checkpoint artifacts/experiments/aspp/best.pt --split val --out-dir artifacts/eval/aspp_val`;
- повтор с другим seed (оценка разброса): `docker compose run --rm train --config configs/baseline.yaml --seed 1`;
- быстрая проверка кода без GPU, интернета и трекера:
  `docker compose run --rm train --config configs/baseline.yaml --epochs 1 --limit 16 --no-tracker --no-pretrained --device cpu`
  (такие запуски помечаются в `summary.json` и не участвуют в сравнении и выборе модели).

Ориентировочное время обучения одного варианта (50 эпох) - 15-30 минут на GPU уровня RTX 3060
(оценка, зависит от GPU и диска).

**Публикация весов** (чтобы `docker compose run --rm weights` работал у проверяющего):

```bash
gh release create v1.0 artifacts/final/model.pt artifacts/final/model_card.json \
  --title "Итоговые веса" --notes "Эксперимент и ID задачи ClearML - в model_card.json"
```

или через веб-интерфейс GitHub: Releases -> Draft a new release, тег `v1.0`, файл `model.pt`.
Адрес по умолчанию в `.env.example`:
`https://github.com/Alectobe/cv-coursework-segmentation/releases/download/v1.0/model.pt`.
Вместо GitHub Release веса можно взять из реестра моделей ClearML: `CLEARML_MODEL_ID=<id>` в `.env`
(ID - в `model_card.json` или во вкладке Models эксперимента).

## 7. Трекер экспериментов (ClearML)

**Подключение:**
1. Зарегистрироваться на <https://app.clear.ml> (бесплатный тариф).
2. Settings -> Workspace -> **Create new credentials**.
3. Скопировать `access_key` и `secret_key` в `.env`:
   ```
   CLEARML_API_ACCESS_KEY=...
   CLEARML_API_SECRET_KEY=...
   ```
   Хосты `api/app/files.clear.ml` уже прописаны в `.env.example`.

Без ключей обучение не запустится (сразу будет понятная ошибка): трекер обязателен.
Флаг `--no-tracker` нужен только для проверки кода.

**Что сохраняется для каждого запуска** (проект `Kvasir-SEG polyp segmentation`):

| В ClearML | Что |
|---|---|
| название, описание, теги | имя конфига, цель модификации (`experiment.description`) |
| CONFIGURATION -> Configuration objects -> `config` | конфиг целиком |
| CONFIGURATION -> Hyperparameters | секции `data`, `model`, `loss`, `train`, `dataset`, `model_stats` |
| Hyperparameters -> `dataset` | источник данных, SHA-256 архива, размеры выборок, SHA-256 файлов разбиения, способ дедупликации |
| SCALARS | loss, Dice, IoU (train и val), precision и recall (val), lr - по эпохам |
| PLOTS / DEBUG SAMPLES | кривые обучения |
| ARTIFACTS | `splits` (файлы разбиения), `history`, `summary` |
| MODELS (Output) | веса лучшей эпохи (`<имя>_best`) |

Итоговая оценка на test - отдельная задача `<имя>_eval_test` (тип *testing*), дочерняя к задаче
обучения: в ней test-метрики (single values), примеры предсказаний и `per_image.csv`.

**Просмотр:** app.clear.ml -> Projects -> *Kvasir-SEG polyp segmentation*. Чтобы сравнить варианты,
отметьте задачи `baseline`, `attn_gates`, `dice_focal`, `aspp` -> **Compare**: графики val Dice
наложатся друг на друга, а различия в конфигурации будут подсвечены.

## 8. Итоговая модель и связь с экспериментом

Связь «веса в сервисе <-> эксперимент в трекере» прослеживается тремя способами:
1. `GET /health` и поле `model` в ответе `/predict`: имя эксперимента, `clearml_task_id`,
   `clearml_task_url` и SHA-256 чекпоинта. ID задачи записывается в чекпоинт во время обучения.
2. `artifacts/final/model_card.json` (публикуется вместе с весами в Release): эксперимент, эпоха,
   val-метрики, ID задачи и модели в ClearML, SHA-256 файла.
3. В ClearML у задачи обучения во вкладке Models лежит тот же файл весов.

**Веса, используемые сервисом:**

| Эксперимент | ClearML task | ClearML model ID | val mDice | test mDice | Файл |
|---|---|---|---|---|---|
| заполняется после финального обучения из `artifacts/final/model_card.json` | | | | | [Release v1.0](https://github.com/Alectobe/cv-coursework-segmentation/releases/tag/v1.0) |

## 9. Структура репозитория

```
├── api/                        # HTTP-сервис (FastAPI)
│   ├── main.py                 #   эндпоинты, валидация входа, коды ошибок
│   ├── predictor.py            #   загрузка весов, предсказание, кодирование PNG
│   └── schemas.py              #   схемы ответов (Swagger)
├── configs/                    # по одному полному конфигу на эксперимент
├── data/
│   ├── raw/                    #   Kvasir-SEG (не в git)
│   └── splits/                 #   В GIT: train/val/test.txt, duplicates.csv, манифест, статистика
├── docker/                     # train.Dockerfile (GPU), api.Dockerfile (CPU)
├── docker-compose.yml          # prepare, train, evaluate, compare, select, examples, api, weights, tests
├── reports/                    # таблицы и графики для отчёта
├── requirements/               # зафиксированные версии (включая транзитивные)
├── scripts/
│   ├── download_data.py        #   скачивание и проверка SHA-256 архива
│   ├── prepare_data.py         #   проверка данных, дубликаты, стратифицированное разбиение
│   ├── compare_experiments.py  #   таблица сравнения вариантов
│   ├── select_final.py         #   выбор итоговой модели по val mDice
│   ├── download_weights.py     #   получение итоговых весов (Release или ClearML)
│   ├── export_examples.py      #   тестовые снимки для демонстрации
│   └── certs/                  #   промежуточный сертификат Sectigo (см. ниже)
├── src/polypseg/               # общий пакет: обучение, оценка, сервис
│   ├── data/                   #   датасет, препроцессинг, аугментации
│   ├── models/                 #   factory.py (сборка вариантов), attention_gate.py, aspp.py
│   ├── losses.py               #   BCE, Dice, Focal, Dice+Focal
│   ├── metrics.py              #   Dice, IoU, precision, recall по снимкам
│   ├── train.py / evaluate.py  #   обучение и оценка
│   ├── inference.py            #   загрузка чекпоинта и предсказание (общий для evaluate и api)
│   └── tracking.py             #   обёртка над ClearML
├── tests/                      # pytest: модели, лоссы и метрики, разбиение, API
├── README.md
└── REPORT.md
```

**Про `scripts/certs/`.** Сервер `datasets.simula.no` отдаёт неполную цепочку TLS-сертификатов.
Браузер сам докачивает недостающий промежуточный сертификат, а Python - нет, и загрузка падает с
`CERTIFICATE_VERIFY_FAILED`. Поэтому публичный промежуточный сертификат *Sectigo Public Server
Authentication CA DV R36* лежит в репозитории и добавляется к доверенным корням. Проверка TLS не
отключается, а архив дополнительно сверяется по SHA-256. Если загрузка всё же не удалась, скачайте
архив браузером и выполните `docker compose run --rm prepare sh -c "python scripts/download_data.py --zip data/raw/kvasir-seg.zip && python scripts/prepare_data.py"`, предварительно положив его в `data/raw/`.

## 10. Тесты

```bash
docker compose run --rm tests        # в контейнере сервиса, ничего не ставит на хост
```

29 тестов: форма выхода всех вариантов модели, что модификации только добавляют слои к исходному
U-Net, Attention Gate и ASPP, значения метрик в пограничных случаях, лоссы, корректность
зафиксированного разбиения (без пересечений, дубликаты в одной выборке, хэши совпадают с манифестом),
API (200 и все 4xx/503).

## 11. Источники и заимствования

| Что | Источник | Как используется |
|---|---|---|
| U-Net, энкодер ResNet-34 | [segmentation_models_pytorch 0.5.0](https://github.com/qubvel-org/segmentation_models.pytorch); Ronneberger et al., [arXiv:1505.04597](https://arxiv.org/abs/1505.04597); He et al., [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | готовая реализация (исходная модель) |
| Attention Gate | Oktay et al., [arXiv:1804.03999](https://arxiv.org/abs/1804.03999) | **реализовано самостоятельно**: `models/attention_gate.py`, встраивание в декодер `models/factory.py` |
| ASPP | Chen et al., DeepLabv3, [arXiv:1706.05587](https://arxiv.org/abs/1706.05587) | **реализовано самостоятельно**: `models/aspp.py` (dilation уменьшены под карту 11x11) |
| Focal loss | Lin et al., [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | **реализовано самостоятельно**: `losses.py` |
| Dice loss | Milletari et al., V-Net, [arXiv:1606.04797](https://arxiv.org/abs/1606.04797) | **реализовано самостоятельно**: `losses.py` |
| Kvasir-SEG | Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020 | данные |
| Остальное | - | код данных, обучения, оценки, метрик, сервиса и Docker написан для проекта |
