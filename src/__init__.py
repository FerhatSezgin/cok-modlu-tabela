"""Trafik İşareti Multimodal Sürüş Güvenliği ve Kural İhlali Sınıflandırması paketi."""

from importlib import import_module
from typing import Any

from src.config import (
    BASE_DIR,
    DATA_DIR,
    FIGURES_DIR,
    IMAGE_SIZE,
    MIN_SAMPLES_PER_CLASS,
    MIN_TOTAL_SAMPLES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAFETY_CATEGORIES,
    SIGN_CATALOG,
    SPLIT_RATIOS,
)

_LAZY_EXPORTS = {
    "download_or_prepare_raw_data": ("src.data_loader", "download_or_prepare_raw_data"),
    "build_images_dataframe": ("src.data_loader", "build_images_dataframe"),
    "build_texts_dataframe": ("src.data_loader", "build_texts_dataframe"),
    "build_multimodal_dataframe": ("src.data_loader", "build_multimodal_dataframe"),
    "validate_multimodal_dataset": ("src.data_loader", "validate_multimodal_dataset"),
    "run_step2_pipeline": ("src.data_loader", "run_step2_pipeline"),
    "clean_multimodal_dataset": ("src.preprocessing", "clean_multimodal_dataset"),
    "preprocess_single_image": ("src.preprocessing", "preprocess_single_image"),
    "clean_turkish_text": ("src.preprocessing", "clean_turkish_text"),
    "run_step3_pipeline": ("src.preprocessing", "run_step3_pipeline"),
    "compute_statistical_summary": ("src.eda", "compute_statistical_summary"),
    "run_step4_eda_pipeline": ("src.eda", "run_step4_eda_pipeline"),
    "create_stratified_splits": ("src.splitter", "create_stratified_splits"),
    "prepare_model_ready_tensors": ("src.splitter", "prepare_model_ready_tensors"),
    "run_step4_split_pipeline": ("src.splitter", "run_step4_split_pipeline"),
}


def __getattr__(name: str) -> Any:
    """Alt modülleri ihtiyaç anında yükleyerek `python -m src.<modul>` uyarısını önler."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "FIGURES_DIR",
    "REPORTS_DIR",
    "RANDOM_SEED",
    "IMAGE_SIZE",
    "MIN_SAMPLES_PER_CLASS",
    "MIN_TOTAL_SAMPLES",
    "SPLIT_RATIOS",
    "SAFETY_CATEGORIES",
    "SIGN_CATALOG",
    *_LAZY_EXPORTS.keys(),
]
