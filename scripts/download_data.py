"""Скачивание и распаковка Kvasir-SEG в data/raw.

    python scripts/download_data.py                       # скачать с официального сайта
    python scripts/download_data.py --zip ~/kvasir-seg.zip  # взять архив, скачанный вручную

Про сертификат: сервер datasets.simula.no отдаёт неполную цепочку TLS
(нет промежуточного сертификата "Sectigo Public Server Authentication CA DV R36").
Браузер сам докачивает его, а Python - нет, и загрузка падает с CERTIFICATE_VERIFY_FAILED.
Поэтому публичный промежуточный сертификат лежит в scripts/certs и добавляется
к доверенным корням. Проверка TLS при этом не отключается, а целостность архива
дополнительно проверяется по SHA-256.
"""

import argparse
import hashlib
import shutil
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

from polypseg.data.source import ARCHIVE_NAME, DATASET_DIR, KVASIR_SHA256, KVASIR_URL

INTERMEDIATE_CERT = Path(__file__).resolve().parent / "certs" / "sectigo_public_server_auth_ca_dv_r36.pem"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ssl_context():
    try:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    ctx.load_verify_locations(cafile=str(INTERMEDIATE_CERT))
    return ctx


def download(url, dst):
    print(f"[data] скачиваю {url}")
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(url, context=ssl_context(), timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[data] {done / 2**20:6.1f} / {total / 2**20:.1f} МБ", end="", flush=True)
    print()
    tmp.replace(dst)


def safe_extract(archive, out_dir):
    """Распаковка с проверкой, что файлы не выходят за пределы out_dir."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (out_dir / member).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise RuntimeError(f"Подозрительный путь в архиве: {member}")
        zf.extractall(out_dir)


def main():
    p = argparse.ArgumentParser(description="Скачивание Kvasir-SEG")
    p.add_argument("--out", default="data/raw", help="каталог для архива и распакованных данных")
    p.add_argument("--zip", help="путь к уже скачанному архиву kvasir-seg.zip")
    p.add_argument("--url", default=KVASIR_URL)
    p.add_argument("--skip-sha256", action="store_true", help="не сверять SHA-256 (если архив на сайте обновился)")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / ARCHIVE_NAME
    dataset_dir = out_dir / DATASET_DIR

    if (dataset_dir / "images").is_dir() and (dataset_dir / "masks").is_dir():
        print(f"[data] данные уже распакованы в {dataset_dir}, пропускаю загрузку")
        return

    if args.zip:
        shutil.copyfile(args.zip, archive)
    elif not archive.exists():
        try:
            download(args.url, archive)
        except Exception as exc:  # noqa: BLE001 - нужно показать понятную инструкцию при любой сетевой ошибке
            print(f"[data] не удалось скачать архив: {exc}", file=sys.stderr)
            print(
                f"[data] скачайте {args.url} вручную (браузером) в {archive} и запустите скрипт снова",
                file=sys.stderr,
            )
            sys.exit(1)

    digest = sha256(archive)
    if digest != KVASIR_SHA256 and not args.skip_sha256:
        print(f"[data] SHA-256 архива {digest} не совпадает с ожидаемым {KVASIR_SHA256}", file=sys.stderr)
        print("[data] архив повреждён или изменён; удалите его и скачайте заново", file=sys.stderr)
        sys.exit(1)
    print(f"[data] SHA-256 архива: {digest}")

    safe_extract(archive, out_dir)
    n_images = len(list((dataset_dir / "images").glob("*.jpg")))
    n_masks = len(list((dataset_dir / "masks").glob("*.jpg")))
    print(f"[data] распаковано в {dataset_dir}: {n_images} снимков, {n_masks} масок")


if __name__ == "__main__":
    main()
