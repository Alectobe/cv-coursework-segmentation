"""Тонкая обёртка над ClearML.

Весь код обучения и оценки обращается к трекеру только через этот класс.
При enabled=False вызовы ничего не делают: это нужно для smoke-тестов (--no-tracker).
Для настоящих экспериментов трекер обязателен, и без ключей запуск сразу падает
с понятной ошибкой, а не обучается «вслепую».
"""

import os
from pathlib import Path


def _credentials_available():
    has_env = bool(os.environ.get("CLEARML_API_ACCESS_KEY")) and bool(os.environ.get("CLEARML_API_SECRET_KEY"))
    has_conf = Path(os.environ.get("CLEARML_CONFIG_FILE", Path.home() / "clearml.conf")).exists()
    return has_env or has_conf


class Tracker:
    def __init__(self, enabled, project, task_name, task_type="training", tags=None, description=None):
        self.task = None
        self.logger = None
        if not enabled:
            print("[tracker] Трекер отключён (--no-tracker): метрики пишутся только локально")
            return
        if not _credentials_available():
            raise RuntimeError(
                "Не найдены ключи ClearML. Заполните CLEARML_API_ACCESS_KEY и CLEARML_API_SECRET_KEY в .env "
                "(см. README, раздел «Трекер экспериментов») или запустите с --no-tracker для проверки кода."
            )
        from clearml import Task

        types = {"training": Task.TaskTypes.training, "testing": Task.TaskTypes.testing}
        self.task = Task.init(
            project_name=project,
            task_name=task_name,
            task_type=types[task_type],
            tags=list(tags or []),
            output_uri=True,  # веса загружаются в файловое хранилище ClearML
            reuse_last_task_id=False,  # каждый запуск - отдельный эксперимент
            auto_connect_frameworks={"pytorch": False, "matplotlib": False, "tensorboard": False},
        )
        if description:
            self.task.set_comment(description)
        self.logger = self.task.get_logger()

    @property
    def enabled(self):
        return self.task is not None

    @property
    def task_id(self):
        return self.task.id if self.task else None

    @property
    def task_url(self):
        return self.task.get_output_log_web_page() if self.task else None

    def connect(self, obj, name):
        """Регистрирует словарь параметров (вкладка CONFIGURATION -> HYPERPARAMETERS)."""
        if self.task:
            self.task.connect(obj, name=name)
        return obj

    def connect_configuration(self, cfg, name):
        """Сохраняет конфиг целиком (вкладка CONFIGURATION -> CONFIGURATION OBJECTS)."""
        if self.task:
            self.task.connect_configuration(configuration=cfg, name=name)

    def set_parent(self, task_id):
        if self.task and task_id:
            self.task.set_parent(task_id)

    def scalar(self, title, series, value, iteration):
        if self.logger:
            self.logger.report_scalar(title=title, series=series, value=float(value), iteration=iteration)

    def single_value(self, name, value):
        if self.logger:
            self.logger.report_single_value(name=name, value=float(value))

    def image(self, title, series, path, iteration=0):
        if self.logger:
            self.logger.report_image(title=title, series=series, iteration=iteration, local_path=str(path))

    def artifact(self, name, obj):
        if self.task:
            self.task.upload_artifact(name=name, artifact_object=obj, wait_on_upload=True)

    def upload_model(self, path, name, tags=None, iteration=None):
        """Загружает веса как OutputModel и возвращает ID модели в ClearML."""
        if not self.task:
            return None
        from clearml import OutputModel

        try:
            model = OutputModel(task=self.task, name=name, framework="PyTorch", tags=list(tags or []))
            model.update_weights(
                weights_filename=str(path),
                auto_delete_file=False,
                iteration=iteration,
                async_enable=False,
            )
            return model.id
        except Exception as exc:  # noqa: BLE001 - веса не должны теряться из-за сбоя реестра моделей
            # Например, в офлайн-режиме ClearML OutputModel не поддерживается.
            # Тогда веса сохраняются в задаче как обычный артефакт.
            print(f"[tracker] не удалось зарегистрировать модель ({exc}); загружаю веса как артефакт задачи")
            self.artifact("best_weights", Path(path))
            return None

    def close(self):
        if self.task:
            self.task.flush(wait_for_uploads=True)
            self.task.close()
