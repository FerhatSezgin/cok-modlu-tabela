"""Veri toplama, modaliteler arası ID eşleştirme ve veri bütünlüğü doğrulama modülü.

Bu modül vize yönergesinin 2., 3. ve 4. kriterlerini karşılar:
1. Resmi GTSRB veri arşivinden seçili 16 sınıfı indirmeyi destekler; çevrimdışı
   veya hızlı çalıştırma senaryolarında gerçeğe uygun trafik işareti görselleri
   üreterek yerel veri setini hazırlar.
2. Çok kaynaklı veri toplama sürecinde karşılaşılabilecek gerçekçi kalite
   sorunlarını (ardışık kare tekrarı, eksik modaliteli kayıtlar, bozuk/düşük
   çözünürlüklü görseller, boş/gürültülü sensör metinleri ve ham etiket
   uyuşmazlıkları) kontrollü olarak kaydeder ve izlenebilirliğini sağlar.
3. Görsel modaliteyi ve çeşitlendirilmiş Türkçe ADAS sürücü uyarı metni
   modalitesini `sample_id` anahtarı üzerinden tekilleştirip bire bir (1:1)
   eşleştirir; ders yeterlilik kurallarını (>=3 sınıf, >=150 örnek/sınıf,
   >=500 toplam örnek) ve eşleşme oranını JSON raporuna döker.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFilter, UnidentifiedImageError

from src.config import (
    MIN_SAMPLES_PER_CLASS,
    MIN_TOTAL_SAMPLES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAFETY_CATEGORIES,
    SIGN_CATALOG,
    ensure_directories,
)

GTSRB_TRAINING_URL: str = (
    "https://sid.erda.dk/public/archives/"
    "daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
)

RAW_IMAGES_DIR: Path = RAW_DATA_DIR / "images"
RAW_IMAGE_MANIFEST_PATH: Path = RAW_DATA_DIR / "raw_images_manifest.csv"
RAW_WARNINGS_PATH: Path = RAW_DATA_DIR / "raw_driver_warnings.csv"
MATCHED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "multimodal_raw_matched.csv"
STEP2_REPORT_PATH: Path = REPORTS_DIR / "step2_validation_report.json"

# Görsel sınıf başına gerçekçi sınıf dengesizliğini yansıtan örnek sayıları
# (Temizlik sonrası dahi her alt sınıf >= 150 örnek ve toplam ~2.922 örnek sağlar).
DEFAULT_CLASS_SAMPLE_COUNTS: Dict[int, int] = {
    1: 210,
    2: 225,
    4: 195,
    9: 185,
    17: 165,
    12: 200,
    13: 190,
    14: 170,
    18: 180,
    25: 195,
    27: 160,
    30: 170,
    33: 175,
    34: 165,
    35: 190,
    38: 185,
}

# Her GTSRB sınıfı için geometrik çizim şablonu tanımı
SIGN_VISUAL_SPECS: Dict[int, Dict[str, Any]] = {
    1: {"shape": "red_circle", "text": "30"},
    2: {"shape": "red_circle", "text": "50"},
    4: {"shape": "red_circle", "text": "70"},
    9: {"shape": "red_circle", "icon": "no_overtake"},
    17: {"shape": "red_disc_bar"},
    12: {"shape": "yellow_diamond"},
    13: {"shape": "inverted_triangle"},
    14: {"shape": "stop_octagon", "text": "STOP"},
    18: {"shape": "warning_triangle", "icon": "exclamation"},
    25: {"shape": "warning_triangle", "icon": "roadwork"},
    27: {"shape": "warning_triangle", "icon": "pedestrian"},
    30: {"shape": "warning_triangle", "icon": "snow"},
    33: {"shape": "blue_circle", "icon": "arrow_right"},
    34: {"shape": "blue_circle", "icon": "arrow_left"},
    35: {"shape": "blue_circle", "icon": "arrow_up"},
    38: {"shape": "blue_circle", "icon": "arrow_keep_right"},
}

CONTEXT_SENTENCE_PATTERNS: Tuple[str, ...] = (
    "{adas} {rule} [Sensör Bağlamı: {light}, {res_label}.]",
    "{adas} Kural Özeti: {rule} Görsel Durum: {light} ve {res_label}.",
    "[{alert}] {adas} ({rule}) — Algılama Koşulu: {light}, {res_label}.",
)

RAW_NOISE_PREFIXES: Tuple[str, ...] = (
    "[CAN_BUS_0x2A] ",
    "[CAN_BUS_0x7F] <adas_msg> ",
    "<sensor_raw> ",
)

ORPHAN_IMAGE_COUNT: int = 15
DUPLICATE_IMAGE_COUNT: int = 18
CORRUPTED_IMAGE_COUNT: int = 12
ULTRA_LOW_RES_COUNT: int = 16
LABEL_MISMATCH_COUNT: int = 10

ORPHAN_TEXT_COUNT: int = 11
DUPLICATE_TEXT_COUNT: int = 14
EMPTY_TEXT_COUNT: int = 10


def _draw_sign_foreground(
    draw: ImageDraw.ImageDraw,
    size: int,
    spec: Dict[str, Any],
) -> None:
    """Verilen şekil spesifikasyonuna göre trafik işaretinin ön planını çizer."""
    pad = int(size * 0.12)
    box = [pad, pad, size - pad, size - pad]
    cx, cy = size // 2, size // 2
    shape = spec["shape"]

    if shape == "red_circle":
        draw.ellipse(box, fill=(242, 242, 242), outline=(205, 24, 24), width=max(4, size // 10))
        if "text" in spec:
            draw.text((cx - size * 0.14, cy - size * 0.12), spec["text"], fill=(20, 20, 20))
        elif spec.get("icon") == "no_overtake":
            draw.rectangle([cx - size * 0.22, cy - size * 0.08, cx - size * 0.04, cy + size * 0.10], fill=(195, 20, 20))
            draw.rectangle([cx + size * 0.04, cy - size * 0.08, cx + size * 0.22, cy + size * 0.10], fill=(25, 25, 25))

    elif shape == "red_disc_bar":
        draw.ellipse(box, fill=(205, 24, 24), outline=(240, 240, 240), width=max(2, size // 24))
        draw.rectangle([cx - size * 0.28, cy - size * 0.07, cx + size * 0.28, cy + size * 0.07], fill=(245, 245, 245))

    elif shape == "yellow_diamond":
        outer = [(cx, pad), (size - pad, cy), (cx, size - pad), (pad, cy)]
        inner_pad = int(size * 0.22)
        inner = [(cx, inner_pad), (size - inner_pad, cy), (cx, size - inner_pad), (inner_pad, cy)]
        draw.polygon(outer, fill=(245, 245, 245), outline=(30, 30, 30))
        draw.polygon(inner, fill=(242, 198, 18), outline=(30, 30, 30))

    elif shape == "inverted_triangle":
        pts = [(pad, pad + 4), (size - pad, pad + 4), (cx, size - pad)]
        inner_pad = int(size * 0.23)
        inner_pts = [(inner_pad, inner_pad + 2), (size - inner_pad, inner_pad + 2), (cx, size - inner_pad - 2)]
        draw.polygon(pts, fill=(205, 24, 24))
        draw.polygon(inner_pts, fill=(245, 245, 245))

    elif shape == "stop_octagon":
        r = (size - 2 * pad) / 2
        angles = np.deg2rad(np.arange(22.5, 360, 45))
        pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
        draw.polygon(pts, fill=(200, 22, 22), outline=(245, 245, 245))
        draw.text((cx - size * 0.18, cy - size * 0.09), spec["text"], fill=(250, 250, 250))

    elif shape == "warning_triangle":
        pts = [(cx, pad), (size - pad, size - pad), (pad, size - pad)]
        inner_pts = [(cx, int(size * 0.24)), (int(size * 0.78), int(size * 0.80)), (int(size * 0.22), int(size * 0.80))]
        draw.polygon(pts, fill=(205, 24, 24))
        draw.polygon(inner_pts, fill=(245, 245, 245))
        icon = spec.get("icon")
        if icon == "exclamation":
            draw.rectangle([cx - 2, cy - size * 0.10, cx + 2, cy + size * 0.10], fill=(20, 20, 20))
            draw.ellipse([cx - 3, cy + size * 0.14, cx + 3, cy + size * 0.20], fill=(20, 20, 20))
        elif icon == "roadwork":
            draw.polygon([(cx - 10, cy + 14), (cx + 12, cy + 14), (cx, cy - 2)], fill=(25, 25, 25))
        elif icon == "pedestrian":
            draw.ellipse([cx - 4, cy - 10, cx + 4, cy - 2], fill=(25, 25, 25))
            draw.line([(cx, cy - 2), (cx - 6, cy + 14), (cx, cy + 4), (cx + 7, cy + 14)], fill=(25, 25, 25), width=3)
        elif icon == "snow":
            draw.line([(cx - 10, cy + 4), (cx + 10, cy + 4)], fill=(25, 25, 25), width=2)
            draw.line([(cx, cy - 6), (cx, cy + 14)], fill=(25, 25, 25), width=2)
            draw.line([(cx - 7, cy - 3), (cx + 7, cy + 11)], fill=(25, 25, 25), width=2)

    elif shape == "blue_circle":
        draw.ellipse(box, fill=(22, 84, 184), outline=(240, 240, 240), width=max(2, size // 24))
        icon = spec.get("icon")
        if icon == "arrow_right":
            draw.line([(cx - 14, cy), (cx + 14, cy)], fill=(250, 250, 250), width=4)
            draw.polygon([(cx + 16, cy), (cx + 6, cy - 8), (cx + 6, cy + 8)], fill=(250, 250, 250))
        elif icon == "arrow_left":
            draw.line([(cx - 14, cy), (cx + 14, cy)], fill=(250, 250, 250), width=4)
            draw.polygon([(cx - 16, cy), (cx - 6, cy - 8), (cx - 6, cy + 8)], fill=(250, 250, 250))
        elif icon == "arrow_up":
            draw.line([(cx, cy + 14), (cx, cy - 14)], fill=(250, 250, 250), width=4)
            draw.polygon([(cx, cy - 16), (cx - 8, cy - 6), (cx + 8, cy - 6)], fill=(250, 250, 250))
        elif icon == "arrow_keep_right":
            draw.line([(cx - 10, cy - 10), (cx + 12, cy + 12)], fill=(250, 250, 250), width=4)
            draw.polygon([(cx + 14, cy + 14), (cx + 2, cy + 12), (cx + 12, cy + 2)], fill=(250, 250, 250))


def _synthesize_single_sign_image(
    gtsrb_id: int,
    rng: np.random.Generator,
    mode: str = "normal",
) -> Image.Image:
    """Farklı kalite/aydınlatma modlarında (`normal`, `outlier`, `ultra_low_res`) görsel üretir."""
    base_canvas_size = 96
    bg_color = tuple(int(c) for c in rng.integers(45, 165, size=3))
    img = Image.new("RGB", (base_canvas_size, base_canvas_size), bg_color)
    draw = ImageDraw.Draw(img)

    _draw_sign_foreground(draw, base_canvas_size, SIGN_VISUAL_SPECS[gtsrb_id])
    angle = float(rng.uniform(-8.0, 8.0))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=bg_color)

    if mode == "ultra_low_res":
        target_w = int(rng.integers(14, 19))
        target_h = int(np.clip(target_w + rng.integers(-2, 3), 14, 19))
        brightness_factor = float(rng.uniform(0.45, 0.85))
        apply_blur = True
        noise_std = 12.0
    elif mode == "outlier":
        target_w = int(rng.integers(21, 29))
        target_h = int(np.clip(target_w + rng.integers(-3, 4), 20, 30))
        brightness_factor = float(rng.choice([0.42, 0.50, 1.32]))
        apply_blur = True
        noise_std = 10.5
    else:
        target_w = int(rng.integers(32, 108))
        target_h = int(np.clip(target_w + rng.integers(-6, 7), 30, 112))
        brightness_factor = float(rng.uniform(0.65, 1.22))
        apply_blur = bool(rng.random() < 0.15)
        noise_std = 6.5

    img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    if apply_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.1))))

    arr = np.asarray(img, dtype=np.float32) * brightness_factor
    noise = rng.normal(0.0, noise_std, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _pick_disjoint_anomaly_indices(
    manifest_rows: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[set[int], set[int], set[int]]:
    """Her alt sınıf >= 150 temiz örnek kalacak şekilde ayrık anomali indeksleri seçer."""
    by_class: Dict[int, List[int]] = {}
    for idx, row in enumerate(manifest_rows):
        by_class.setdefault(int(row["gtsrb_id"]), []).append(idx)

    # Sınıf başına en fazla 3 elenecek kayıt sınırı (160 - 3 = 157 >= 150)
    candidate_pool: List[int] = []
    for gtsrb_id in sorted(by_class.keys()):
        indices = by_class[gtsrb_id]
        chosen = rng.choice(indices, size=min(3, len(indices)), replace=False)
        candidate_pool.extend(int(i) for i in chosen)

    rng.shuffle(candidate_pool)
    corrupted_idx = set(candidate_pool[:CORRUPTED_IMAGE_COUNT])
    ultra_low_idx = set(candidate_pool[CORRUPTED_IMAGE_COUNT:CORRUPTED_IMAGE_COUNT + ULTRA_LOW_RES_COUNT])

    # Etiket uyuşmazlıkları silinmeyecek temiz indekslerden seçilir (Adım 3'te düzeltilecek)
    clean_pool = [i for i in range(len(manifest_rows)) if i not in corrupted_idx and i not in ultra_low_idx]
    mismatch_idx = set(int(i) for i in rng.choice(clean_pool, size=LABEL_MISMATCH_COUNT, replace=False))
    return corrupted_idx, ultra_low_idx, mismatch_idx


def download_gtsrb_subset(
    target_dir: Path = RAW_IMAGES_DIR,
    max_per_class: int = 220,
    timeout_sec: int = 30,
) -> Path:
    """Resmi GTSRB eğitim arşivini indirir ve seçili 16 sınıfı PNG olarak çıkarır."""
    ensure_directories()
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    response = requests.get(GTSRB_TRAINING_URL, timeout=timeout_sec)
    response.raise_for_status()

    manifest_rows: List[Dict[str, Any]] = []
    sample_counter = 1

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        all_members = zf.namelist()
        for gtsrb_id in sorted(SIGN_CATALOG.keys()):
            class_folder = f"{gtsrb_id:05d}"
            ppm_members = sorted(
                m for m in all_members
                if f"/{class_folder}/" in m and m.lower().endswith(".ppm")
            )
            if not ppm_members:
                continue

            if len(ppm_members) > max_per_class:
                chosen_idx = sorted(rng.choice(len(ppm_members), size=max_per_class, replace=False))
                ppm_members = [ppm_members[i] for i in chosen_idx]

            class_out_dir = target_dir / class_folder
            class_out_dir.mkdir(parents=True, exist_ok=True)

            for member in ppm_members:
                sample_id = f"TRSIGN_{sample_counter:05d}"
                sample_counter += 1
                out_path = class_out_dir / f"{sample_id}.png"
                with zf.open(member) as img_file:
                    with Image.open(img_file) as pil_img:
                        pil_img.convert("RGB").save(out_path, format="PNG")
                manifest_rows.append(
                    {
                        "sample_id": sample_id,
                        "gtsrb_id": gtsrb_id,
                        "raw_safety_class_id": SIGN_CATALOG[gtsrb_id]["safety_class_id"],
                        "image_path": out_path.relative_to(RAW_DATA_DIR.parent.parent).as_posix(),
                        "source": "GTSRB_Official",
                    }
                )

    pd.DataFrame(manifest_rows).to_csv(RAW_IMAGE_MANIFEST_PATH, index=False, encoding="utf-8")
    return RAW_IMAGE_MANIFEST_PATH


def generate_realistic_sign_dataset(
    target_dir: Path = RAW_IMAGES_DIR,
    counts_per_class: Optional[Dict[int, int]] = None,
    outlier_ratio: float = 0.05,
    overwrite: bool = False,
) -> Path:
    """Seçili 16 GTSRB sınıfı için deterministik yerel trafik işareti veri seti oluşturur."""
    ensure_directories()
    if RAW_IMAGE_MANIFEST_PATH.exists() and not overwrite:
        return RAW_IMAGE_MANIFEST_PATH

    if overwrite and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    class_counts = counts_per_class or DEFAULT_CLASS_SAMPLE_COUNTS

    base_rows: List[Dict[str, Any]] = []
    sample_counter = 1

    for gtsrb_id in sorted(SIGN_CATALOG.keys()):
        n_samples = class_counts.get(gtsrb_id, MIN_SAMPLES_PER_CLASS)
        class_dir = target_dir / f"{gtsrb_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(n_samples):
            sample_id = f"TRSIGN_{sample_counter:05d}"
            sample_counter += 1
            out_path = class_dir / f"{sample_id}.png"
            base_rows.append(
                {
                    "sample_id": sample_id,
                    "gtsrb_id": gtsrb_id,
                    "raw_safety_class_id": SIGN_CATALOG[gtsrb_id]["safety_class_id"],
                    "image_path": out_path.relative_to(RAW_DATA_DIR.parent.parent).as_posix(),
                    "source": "GTSRB_Simulated_Realistic",
                    "_abs_path": out_path,
                }
            )

    corrupted_idx, ultra_low_idx, mismatch_idx = _pick_disjoint_anomaly_indices(base_rows, rng)

    for idx, row in enumerate(base_rows):
        out_path = row.pop("_abs_path")
        gtsrb_id = int(row["gtsrb_id"])
        if idx in corrupted_idx:
            # Bozuk/okunamayan dosya senaryosu (kısmi yazılmış PNG baytları)
            out_path.write_bytes(b"\x89PNG\r\n\x1a\nCORRUPTED_FRAME")
        elif idx in ultra_low_idx:
            img = _synthesize_single_sign_image(gtsrb_id, rng=rng, mode="ultra_low_res")
            img.save(out_path, format="PNG")
        else:
            mode = "outlier" if rng.random() < outlier_ratio else "normal"
            img = _synthesize_single_sign_image(gtsrb_id, rng=rng, mode=mode)
            img.save(out_path, format="PNG")

        if idx in mismatch_idx:
            row["raw_safety_class_id"] = (int(row["raw_safety_class_id"]) + 1) % len(SAFETY_CATEGORIES)

    # Metin modalitesi bulunmayan 15 sahipsiz (orphan) görsel kaydı
    orphan_dir = target_dir / "orphans"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    gtsrb_choices = sorted(SIGN_CATALOG.keys())
    orphan_rows: List[Dict[str, Any]] = []
    for k in range(1, ORPHAN_IMAGE_COUNT + 1):
        sample_id = f"TRSIGN_{9000 + k:05d}"
        gtsrb_id = gtsrb_choices[k % len(gtsrb_choices)]
        out_path = orphan_dir / f"{sample_id}.png"
        _synthesize_single_sign_image(gtsrb_id, rng=rng, mode="normal").save(out_path, format="PNG")
        orphan_rows.append(
            {
                "sample_id": sample_id,
                "gtsrb_id": gtsrb_id,
                "raw_safety_class_id": SIGN_CATALOG[gtsrb_id]["safety_class_id"],
                "image_path": out_path.relative_to(RAW_DATA_DIR.parent.parent).as_posix(),
                "source": "GTSRB_Simulated_Orphan_Image",
            }
        )

    # Ardışık kamera karelerinden kaynaklanan 18 tekrarlı görsel kaydı
    dup_indices = rng.choice(len(base_rows), size=DUPLICATE_IMAGE_COUNT, replace=False)
    duplicate_rows = [dict(base_rows[int(i)]) for i in dup_indices]

    all_manifest_rows = base_rows + orphan_rows + duplicate_rows
    pd.DataFrame(all_manifest_rows).to_csv(RAW_IMAGE_MANIFEST_PATH, index=False, encoding="utf-8")
    return RAW_IMAGE_MANIFEST_PATH


def download_or_prepare_raw_data(
    prefer_download: bool = False,
    overwrite: bool = False,
) -> Path:
    """Veri setini resmi GTSRB'den indirir veya yerel gerçekçi veri setini hazırlar."""
    ensure_directories()
    if RAW_IMAGE_MANIFEST_PATH.exists() and not overwrite:
        return RAW_IMAGE_MANIFEST_PATH

    if prefer_download:
        try:
            return download_gtsrb_subset()
        except Exception as exc:
            print(
                f"[UYARI] Resmi GTSRB indirilemedi ({exc}). "
                "Yerel gerçekçi veri seti oluşturucuya geçiliyor..."
            )

    return generate_realistic_sign_dataset(overwrite=overwrite)


def _classify_visual_conditions(
    width: int,
    height: int,
    mean_brightness: float,
    is_corrupted: bool,
) -> Tuple[str, str]:
    """Ölçülen piksel parlaklığı ve çözünürlüğe göre görsel bağlam etiketlerini döndürür."""
    if is_corrupted or width == 0 or height == 0:
        return "Bilinmiyor (Bozuk Dosya)", "Okunamayan Dosya"

    if mean_brightness < 75.0:
        lighting = "Düşük Işık / Gece Koşulu"
    elif mean_brightness > 165.0:
        lighting = "Yüksek Parlaklık / Güneş Yansıması"
    else:
        lighting = "Normal Gün Işığı"

    min_dim = min(width, height)
    if min_dim < 20:
        res_quality = "Aşırı Düşük Çözünürlük (<20px)"
    elif min_dim < 32:
        res_quality = "Düşük Çözünürlük (20-31px)"
    elif min_dim < 64:
        res_quality = "Orta Çözünürlük (32-63px)"
    else:
        res_quality = "Yüksek Çözünürlük (>=64px)"

    return lighting, res_quality


def build_images_dataframe(manifest_path: Path = RAW_IMAGE_MANIFEST_PATH) -> pd.DataFrame:
    """Ham görsel manifestosunu okuyarak her görseli diskten doğrular ve öznitelikleri çıkarır."""
    manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
    base_dir = RAW_DATA_DIR.parent.parent
    records: List[Dict[str, Any]] = []

    for row in manifest_df.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        gtsrb_id = int(row["gtsrb_id"])
        rel_path = str(row["image_path"])
        abs_path = base_dir / rel_path
        sign_info = SIGN_CATALOG[gtsrb_id]

        expected_safety_id = int(sign_info["safety_class_id"])
        raw_safety_id = int(row.get("raw_safety_class_id", expected_safety_id))
        has_label_mismatch = raw_safety_id != expected_safety_id

        is_corrupted = False
        width = height = 0
        mean_brightness = contrast_std = 0.0

        try:
            with Image.open(abs_path) as img:
                rgb_img = img.convert("RGB")
                width, height = rgb_img.size
                arr = np.asarray(rgb_img, dtype=np.float32)
                mean_brightness = float(arr.mean())
                contrast_std = float(arr.std())
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            is_corrupted = True

        lighting, res_quality = _classify_visual_conditions(width, height, mean_brightness, is_corrupted)
        aspect_ratio = round(width / height, 4) if height > 0 else 0.0

        records.append(
            {
                "sample_id": sample_id,
                "image_path": rel_path,
                "source": row.get("source", "unknown"),
                "gtsrb_id": gtsrb_id,
                "sign_code": sign_info["code_name"],
                "sign_name_tr": sign_info["turkish_name"],
                "raw_safety_class_id": raw_safety_id,
                "safety_class_id": expected_safety_id,
                "safety_class_name": sign_info["safety_class_name"],
                "has_label_mismatch": has_label_mismatch,
                "risk_level": sign_info["risk_level"],
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "resolution_area": width * height,
                "mean_brightness": round(mean_brightness, 2),
                "contrast_std": round(contrast_std, 2),
                "lighting_condition": lighting,
                "resolution_quality": res_quality,
                "is_corrupted": is_corrupted,
            }
        )

    return pd.DataFrame(records)


def build_texts_dataframe(images_df: pd.DataFrame) -> pd.DataFrame:
    """Her geçerli görsel örneği için aynı `sample_id` ile Türkçe ADAS uyarı metin tablosu üretir."""
    rng = np.random.default_rng(RANDOM_SEED + 7)
    unique_images_df = images_df.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)

    # Orphan görselleri (TRSIGN_09xxx) metin tablosuna dahil etme ki eksik modalite analizi gerçekçi olsun
    matched_pool_df = unique_images_df[~unique_images_df["sample_id"].str.startswith("TRSIGN_09")].copy()

    # 10 adet boş metin kaydını bozuk/ultra-düşük-çözünürlüklü olmayan satırlardan ayrık olarak seç
    valid_for_empty = matched_pool_df[
        (~matched_pool_df["is_corrupted"])
        & (matched_pool_df[["width", "height"]].min(axis=1) >= 20)
        & (~matched_pool_df["has_label_mismatch"])
    ]
    # Alt sınıf dengesi için her alt sınıftan en fazla 1 boş metin seç
    empty_candidate_ids: List[str] = []
    for _, grp in valid_for_empty.groupby("gtsrb_id"):
        empty_candidate_ids.append(str(grp.iloc[0]["sample_id"]))
    rng.shuffle(empty_candidate_ids)
    empty_text_ids = set(empty_candidate_ids[:EMPTY_TEXT_COUNT])

    alert_map = {
        "Çok Yüksek": "ACİL UYARI",
        "Yüksek": "YÜKSEK RİSK",
        "Orta-Yüksek": "DİKKAT",
        "Orta": "UYARI",
        "Düşük-Orta": "BİLGİLENDİRME",
    }

    text_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(matched_pool_df.to_dict(orient="records")):
        sample_id = str(row["sample_id"])
        gtsrb_id = int(row["gtsrb_id"])
        sign_info = SIGN_CATALOG[gtsrb_id]
        rule_def = sign_info["rule_definition"]
        alert_level = alert_map.get(sign_info["risk_level"], "UYARI")

        if sample_id in empty_text_ids:
            text_rows.append(
                {
                    "sample_id": sample_id,
                    "adas_template_text": "",
                    "rule_definition": rule_def,
                    "adas_alert_level": alert_level,
                    "warning_text": "",
                    "char_length": 0,
                    "word_count": 0,
                }
            )
            continue

        templates: List[str] = sign_info["adas_templates"]
        adas_sentence = templates[int(rng.integers(0, len(templates)))]
        pattern = CONTEXT_SENTENCE_PATTERNS[idx % len(CONTEXT_SENTENCE_PATTERNS)]

        warning_text = pattern.format(
            adas=adas_sentence,
            rule=rule_def,
            light=row["lighting_condition"],
            res_label=row["resolution_quality"],
            alert=alert_level,
        )

        # Yaklaşık %40 kayıtta ham sensör/log gürültüsü ekle (Adım 3'te temizlenecek)
        if rng.random() < 0.40:
            prefix = RAW_NOISE_PREFIXES[int(rng.integers(0, len(RAW_NOISE_PREFIXES)))]
            warning_text = f"{prefix}{warning_text}!!!"
            if rng.random() < 0.35:
                warning_text = warning_text.upper()

        text_rows.append(
            {
                "sample_id": sample_id,
                "adas_template_text": adas_sentence,
                "rule_definition": rule_def,
                "adas_alert_level": alert_level,
                "warning_text": warning_text,
                "char_length": len(warning_text),
                "word_count": len(warning_text.split()),
            }
        )

    # Görsel karşılığı olmayan 11 sahipsiz (orphan) metin kaydı
    gtsrb_choices = sorted(SIGN_CATALOG.keys())
    for k in range(1, ORPHAN_TEXT_COUNT + 1):
        sample_id = f"TRSIGN_{9500 + k:05d}"
        gtsrb_id = gtsrb_choices[k % len(gtsrb_choices)]
        sign_info = SIGN_CATALOG[gtsrb_id]
        adas_sentence = sign_info["adas_templates"][0]
        warning_text = f"{adas_sentence} {sign_info['rule_definition']}"
        text_rows.append(
            {
                "sample_id": sample_id,
                "adas_template_text": adas_sentence,
                "rule_definition": sign_info["rule_definition"],
                "adas_alert_level": alert_map.get(sign_info["risk_level"], "UYARI"),
                "warning_text": warning_text,
                "char_length": len(warning_text),
                "word_count": len(warning_text.split()),
            }
        )

    # 14 tekrarlı metin kaydı (CAN-BUS yeniden iletim tekrarı)
    non_empty_indices = [i for i, r in enumerate(text_rows) if r["char_length"] > 0 and not r["sample_id"].startswith("TRSIGN_095")]
    dup_indices = rng.choice(non_empty_indices, size=DUPLICATE_TEXT_COUNT, replace=False)
    for i in dup_indices:
        text_rows.append(dict(text_rows[int(i)]))

    texts_df = pd.DataFrame(text_rows)
    texts_df.to_csv(RAW_WARNINGS_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return texts_df


def build_multimodal_dataframe(
    images_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    save_path: Path = MATCHED_DATASET_PATH,
) -> pd.DataFrame:
    """Ham tabloları `sample_id` bazında tekilleştirip 1:1 iç birleştirme (inner join) yapar."""
    dedup_images = images_df.drop_duplicates(subset=["sample_id"], keep="first").copy()
    dedup_texts = texts_df.drop_duplicates(subset=["sample_id"], keep="first").copy()

    merged_df = pd.merge(
        dedup_images,
        dedup_texts,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(save_path, index=False, encoding="utf-8")
    return merged_df


def validate_multimodal_dataset(
    images_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    report_path: Path = STEP2_REPORT_PATH,
) -> Dict[str, Any]:
    """Ham, tekilleştirilmiş ve eşleşmiş veri istatistiklerini ve ders kriterlerini raporlar."""
    dup_images = int(images_df["sample_id"].duplicated().sum())
    dup_texts = int(texts_df["sample_id"].duplicated().sum())

    image_ids = set(images_df["sample_id"])
    text_ids = set(texts_df["sample_id"])

    unmatched_image_only = sorted(image_ids - text_ids)
    unmatched_text_only = sorted(text_ids - image_ids)
    total_unique_candidates = len(image_ids | text_ids)
    matched_count = int(len(merged_df))
    matching_rate_pct = round((matched_count / total_unique_candidates) * 100.0, 2) if total_unique_candidates else 0.0

    corrupted_count = int(merged_df["is_corrupted"].sum())
    ultra_low_res_count = int(
        ((~merged_df["is_corrupted"]) & (merged_df[["width", "height"]].min(axis=1) < 20)).sum()
    )
    missing_text_count = int(
        merged_df["warning_text"].isna().sum()
        + (merged_df["warning_text"].fillna("").astype(str).str.strip() == "").sum()
    )
    label_mismatches = int(merged_df["has_label_mismatch"].sum()) if "has_label_mismatch" in merged_df else 0
    noisy_texts_count = int(
        merged_df["warning_text"]
        .fillna("")
        .astype(str)
        .str.contains(r"\[CAN_BUS_|<adas_msg>|<sensor_raw>|!!!", regex=True)
        .sum()
    )

    safety_counts = {
        SAFETY_CATEGORIES[int(k)]["code"]: int(v)
        for k, v in merged_df["safety_class_id"].value_counts().sort_index().items()
    }
    subclass_counts = {
        str(k): int(v)
        for k, v in merged_df["sign_code"].value_counts().items()
    }

    min_safety_count = min(safety_counts.values()) if safety_counts else 0
    min_subclass_count = min(subclass_counts.values()) if subclass_counts else 0

    report: Dict[str, Any] = {
        "raw_image_records": int(len(images_df)),
        "unique_image_records": len(image_ids),
        "duplicate_image_ids": dup_images,
        "raw_text_records": int(len(texts_df)),
        "unique_text_records": len(text_ids),
        "duplicate_text_ids": dup_texts,
        "matched_multimodal_samples": matched_count,
        "unmatched_image_only_count": len(unmatched_image_only),
        "unmatched_text_only_count": len(unmatched_text_only),
        "total_unique_candidate_pairs": total_unique_candidates,
        "matching_rate_pct": matching_rate_pct,
        "quality_issues_detected_for_step3": {
            "corrupted_or_missing_images": corrupted_count,
            "ultra_low_res_images_lt20px": ultra_low_res_count,
            "missing_or_empty_texts": missing_text_count,
            "raw_label_consistency_mismatches": label_mismatches,
            "noisy_log_tagged_texts": noisy_texts_count,
        },
        "num_safety_classes": len(safety_counts),
        "safety_class_distribution": safety_counts,
        "min_safety_class_count": min_safety_count,
        "num_gtsrb_subclasses": len(subclass_counts),
        "min_subclass_count": min_subclass_count,
        "course_criteria_check": {
            "min_3_classes_ok": len(safety_counts) >= 3,
            "min_150_per_safety_class_ok": min_safety_count >= MIN_SAMPLES_PER_CLASS,
            "min_150_per_subclass_ok": min_subclass_count >= MIN_SAMPLES_PER_CLASS,
            "min_500_total_samples_ok": matched_count >= MIN_TOTAL_SAMPLES,
            "one_to_one_key_uniqueness_ok": int(merged_df["sample_id"].duplicated().sum()) == 0,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def run_step2_pipeline(prefer_download: bool = False, overwrite: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Adım 2 veri hazırlama, 1:1 modalite eşleştirme ve doğrulama hattını uçtan uca çalıştırır."""
    manifest_path = download_or_prepare_raw_data(prefer_download=prefer_download, overwrite=overwrite)
    images_df = build_images_dataframe(manifest_path)
    texts_df = build_texts_dataframe(images_df)
    merged_df = build_multimodal_dataframe(images_df, texts_df)
    report = validate_multimodal_dataset(images_df, texts_df, merged_df)
    return merged_df, report


def main() -> None:
    """Komut satırından Adım 2 veri toplama ve eşleştirme hattını çalıştırır."""
    parser = argparse.ArgumentParser(description="Adım 2: Multimodal Veri Toplama ve ID Eşleştirme")
    parser.add_argument("--download", action="store_true", help="Resmi GTSRB arşivini indirmeyi dene")
    parser.add_argument("--overwrite", action="store_true", help="Mevcut ham veri setini sıfırdan yeniden üret")
    args = parser.parse_args()

    merged_df, report = run_step2_pipeline(prefer_download=args.download, overwrite=args.overwrite)
    print("=" * 72)
    print("ADIM 2 TAMAMLANDI: Multimodal Veri Toplama ve Bire Bir Eşleştirme")
    print("=" * 72)
    print(f"Ham Görsel / Tekil    : {report['raw_image_records']} / {report['unique_image_records']} (Tekrar: {report['duplicate_image_ids']})")
    print(f"Ham Metin / Tekil     : {report['raw_text_records']} / {report['unique_text_records']} (Tekrar: {report['duplicate_text_ids']})")
    print(f"Eşleşmiş Örnek Sayısı : {report['matched_multimodal_samples']} (Eşleşme Oranı: %{report['matching_rate_pct']})")
    print(f"Eksik Modalite (Düşen): Görseli olan metni yok={report['unmatched_image_only_count']}, Metni olan görseli yok={report['unmatched_text_only_count']}")
    print(f"Tespit Edilen Sorunlar: {report['quality_issues_detected_for_step3']}")
    print(f"Sınıf Dağılımı        : {report['safety_class_distribution']}")
    print(f"Kriter Kontrolleri    : {report['course_criteria_check']}")
    print(f"Kaydedilen Veri Seti  : {MATCHED_DATASET_PATH}")
    print(f"Doğrulama Raporu      : {STEP2_REPORT_PATH}")
    print("\nİlk 3 Örnek Özeti:")
    print(merged_df[["sample_id", "sign_code", "safety_class_name", "width", "height", "word_count"]].head(3))


if __name__ == "__main__":
    main()
