"""Veri toplama, modaliteler arası ID eşleştirme ve veri bütünlüğü doğrulama modülü.

Bu modül vize aşamasının 2., 3. ve 4. isterlerini karşılar:
1. Resmi GTSRB veri arşivinden seçili 16 sınıfı indirmeyi destekler; çevrimdışı
   veya hızlı çalıştırma senaryolarında gerçeğe uygun trafik işareti görselleri
   üreterek yerel veri setini hazırlar.
2. Görsel modaliteyi (RGB görüntüler ve ölçülen görsel kalite metrikleri) ve
   metin modalitesini (çeşitlendirilmiş Türkçe ADAS sürücü uyarı metinleri,
   kural tanımları ve ışık/çözünürlük bağlamı) `sample_id` anahtarı üzerinden
   bire bir (1:1) eşleştirir.
3. Eksik modalite, bozuk dosya, tekrarlı ID, etiket tutarlılığı ve dersin
   minimum veri yeterlilik kurallarını (>=3 sınıf, >=150 örnek/sınıf, >=500
   toplam örnek) denetleyerek JSON doğrulama raporu üretir.
"""

from __future__ import annotations

import csv
import io
import json
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

# Görsel sınıf başına gerçekçi sınıf dengesizliğini yansıtan varsayılan örnek sayıları
# (Her alt sınıf >= 150 örnek kuralını sağlar; toplam ~2.960 örnek üretir).
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

# Her GTSRB sınıfı için geometrik çizim şablonu tanımı (tekrarlayan if/elif bloklarını önler)
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
            label = spec["text"]
            # Basit yüksek kontrastlı piksel-ölçekli rakam yerleşimi
            draw.text((cx - size * 0.14, cy - size * 0.12), label, fill=(20, 20, 20))
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
    is_outlier: bool = False,
) -> Image.Image:
    """Gerçekçi çözünürlük, aydınlatma ve gürültü koşullarıyla tek bir tabela görseli üretir."""
    base_canvas_size = 96
    bg_color = tuple(int(c) for c in rng.integers(45, 165, size=3))
    img = Image.new("RGB", (base_canvas_size, base_canvas_size), bg_color)
    draw = ImageDraw.Draw(img)

    _draw_sign_foreground(draw, base_canvas_size, SIGN_VISUAL_SPECS[gtsrb_id])

    # Hafif perspektif / dönüş açısı varyasyonu
    angle = float(rng.uniform(-8.0, 8.0))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=bg_color)

    # Aykırı örneklerde (düşük çözünürlük / gece / aşırı parlaklık) kalite bozulması uygula
    if is_outlier:
        target_w = int(rng.integers(20, 29))
        target_h = int(np.clip(target_w + rng.integers(-3, 4), 20, 30))
        brightness_factor = float(rng.choice([0.42, 0.50, 1.32]))
        apply_blur = True
    else:
        target_w = int(rng.integers(32, 108))
        target_h = int(np.clip(target_w + rng.integers(-6, 7), 30, 112))
        brightness_factor = float(rng.uniform(0.65, 1.22))
        apply_blur = bool(rng.random() < 0.15)

    img = img.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    if apply_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.1))))

    arr = np.asarray(img, dtype=np.float32) * brightness_factor
    noise = rng.normal(0.0, 6.5 if not is_outlier else 11.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


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
                chosen_idx = sorted(
                    rng.choice(len(ppm_members), size=max_per_class, replace=False)
                )
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

    target_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    class_counts = counts_per_class or DEFAULT_CLASS_SAMPLE_COUNTS

    manifest_rows: List[Dict[str, Any]] = []
    sample_counter = 1

    for gtsrb_id in sorted(SIGN_CATALOG.keys()):
        n_samples = class_counts.get(gtsrb_id, MIN_SAMPLES_PER_CLASS)
        class_dir = target_dir / f"{gtsrb_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(n_samples):
            sample_id = f"TRSIGN_{sample_counter:05d}"
            sample_counter += 1
            is_outlier = bool(rng.random() < outlier_ratio)
            img = _synthesize_single_sign_image(gtsrb_id, rng=rng, is_outlier=is_outlier)

            out_path = class_dir / f"{sample_id}.png"
            img.save(out_path, format="PNG")

            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "gtsrb_id": gtsrb_id,
                    "image_path": out_path.relative_to(RAW_DATA_DIR.parent.parent).as_posix(),
                    "source": "GTSRB_Simulated_Realistic",
                }
            )

    pd.DataFrame(manifest_rows).to_csv(RAW_IMAGE_MANIFEST_PATH, index=False, encoding="utf-8")
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
) -> Tuple[str, str]:
    """Ölçülen piksel parlaklığı ve çözünürlüğe göre görsel bağlam etiketlerini döndürür."""
    if mean_brightness < 75.0:
        lighting = "Düşük Işık / Gece Koşulu"
    elif mean_brightness > 165.0:
        lighting = "Yüksek Parlaklık / Güneş Yansıması"
    else:
        lighting = "Normal Gün Işığı"

    min_dim = min(width, height)
    if min_dim < 32:
        res_quality = "Düşük Çözünürlük (<32px)"
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

        lighting, res_quality = _classify_visual_conditions(width, height, mean_brightness)
        aspect_ratio = round(width / height, 4) if height > 0 else 0.0

        records.append(
            {
                "sample_id": sample_id,
                "image_path": rel_path,
                "source": row.get("source", "unknown"),
                "gtsrb_id": gtsrb_id,
                "sign_code": sign_info["code_name"],
                "sign_name_tr": sign_info["turkish_name"],
                "safety_class_id": sign_info["safety_class_id"],
                "safety_class_name": sign_info["safety_class_name"],
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
    """Her görsel örneği için aynı `sample_id` ile çeşitlendirilmiş Türkçe ADAS uyarı metni üretir."""
    rng = np.random.default_rng(RANDOM_SEED)
    text_rows: List[Dict[str, Any]] = []

    alert_map = {
        "Çok Yüksek": "ACİL UYARI",
        "Yüksek": "YÜKSEK RİSK",
        "Orta-Yüksek": "DİKKAT",
        "Orta": "UYARI",
        "Düşük-Orta": "BİLGİLENDİRME",
    }

    for idx, row in enumerate(images_df.to_dict(orient="records")):
        sample_id = row["sample_id"]
        gtsrb_id = int(row["gtsrb_id"])
        sign_info = SIGN_CATALOG[gtsrb_id]

        templates: List[str] = sign_info["adas_templates"]
        adas_sentence = templates[int(rng.integers(0, len(templates)))]
        rule_def = sign_info["rule_definition"]
        alert_level = alert_map.get(sign_info["risk_level"], "UYARI")
        pattern = CONTEXT_SENTENCE_PATTERNS[idx % len(CONTEXT_SENTENCE_PATTERNS)]

        warning_text = pattern.format(
            adas=adas_sentence,
            rule=rule_def,
            light=row["lighting_condition"],
            res_label=row["resolution_quality"],
            alert=alert_level,
        )

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

    texts_df = pd.DataFrame(text_rows)
    texts_df.to_csv(RAW_WARNINGS_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return texts_df


def build_multimodal_dataframe(
    images_df: pd.DataFrame,
    texts_df: pd.DataFrame,
    save_path: Path = MATCHED_DATASET_PATH,
) -> pd.DataFrame:
    """Görsel ve metin tablolarını `sample_id` üzerinden 1:1 birleştirir ve diske kaydeder."""
    merged_df = pd.merge(
        images_df,
        texts_df,
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
    """Modalite eşleştirmesi, etiket tutarlılığı ve ders kriterlerini denetler."""
    image_ids = set(images_df["sample_id"])
    text_ids = set(texts_df["sample_id"])

    unmatched_in_texts = sorted(image_ids - text_ids)
    unmatched_in_images = sorted(text_ids - image_ids)

    corrupted_count = int(images_df["is_corrupted"].sum()) if "is_corrupted" in images_df else 0
    missing_text_count = int(
        merged_df["warning_text"].isna().sum()
        + (merged_df["warning_text"].astype(str).str.strip() == "").sum()
    )

    # Etiket bütünlüğü kontrolü: her satırın safety_class_id ve code değeri SIGN_CATALOG ile uyuşmalı
    label_mismatches = 0
    for row in merged_df[["gtsrb_id", "safety_class_id", "safety_class_name"]].to_dict("records"):
        expected = SIGN_CATALOG.get(int(row["gtsrb_id"]))
        if (
            expected is None
            or expected["safety_class_id"] != int(row["safety_class_id"])
            or expected["safety_class_name"] != row["safety_class_name"]
        ):
            label_mismatches += 1

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
    total_samples = int(len(merged_df))

    report: Dict[str, Any] = {
        "total_image_records": int(len(images_df)),
        "total_text_records": int(len(texts_df)),
        "matched_multimodal_samples": total_samples,
        "duplicate_image_ids": int(images_df["sample_id"].duplicated().sum()),
        "duplicate_text_ids": int(texts_df["sample_id"].duplicated().sum()),
        "unmatched_image_ids_count": len(unmatched_in_texts),
        "unmatched_text_ids_count": len(unmatched_in_images),
        "corrupted_or_missing_images": corrupted_count,
        "missing_or_empty_texts": missing_text_count,
        "label_consistency_mismatches": label_mismatches,
        "num_safety_classes": len(safety_counts),
        "safety_class_distribution": safety_counts,
        "min_safety_class_count": min_safety_count,
        "num_gtsrb_subclasses": len(subclass_counts),
        "min_subclass_count": min_subclass_count,
        "course_criteria_check": {
            "min_3_classes_ok": len(safety_counts) >= 3,
            "min_150_per_safety_class_ok": min_safety_count >= MIN_SAMPLES_PER_CLASS,
            "min_150_per_subclass_ok": min_subclass_count >= MIN_SAMPLES_PER_CLASS,
            "min_500_total_samples_ok": total_samples >= MIN_TOTAL_SAMPLES,
            "one_to_one_matching_ok": (
                len(unmatched_in_texts) == 0
                and len(unmatched_in_images) == 0
                and corrupted_count == 0
                and missing_text_count == 0
                and label_mismatches == 0
            ),
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
    merged_df, report = run_step2_pipeline(prefer_download=False, overwrite=False)
    print("=" * 72)
    print("ADIM 2 TAMAMLANDI: Multimodal Veri Toplama ve Bire Bir Eşleştirme")
    print("=" * 72)
    print(f"Toplam Eşleşmiş Örnek : {report['matched_multimodal_samples']}")
    print(f"Ana Sınıf Sayısı      : {report['num_safety_classes']}")
    print(f"Alt GTSRB Sınıf Sayısı: {report['num_gtsrb_subclasses']}")
    print(f"Sınıf Dağılımı        : {report['safety_class_distribution']}")
    print(f"Kriter Kontrolleri    : {report['course_criteria_check']}")
    print(f"Kaydedilen Veri Seti  : {MATCHED_DATASET_PATH}")
    print(f"Doğrulama Raporu      : {STEP2_REPORT_PATH}")
    print("\nİlk 3 Örnek Özeti:")
    print(merged_df[["sample_id", "sign_code", "safety_class_name", "width", "height", "word_count"]].head(3))


if __name__ == "__main__":
    main()
