"""Model availability and model status handlers."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

import logging
import os

from api_types import LinkModelsResponse, ModelFileStatus, ModelInfo, ModelsStatusResponse, TextEncoderStatus
from handlers.base import StateHandlerBase, with_state_lock
from runtime_config.model_download_specs import MODEL_FILE_ORDER, resolve_required_model_types
from state.app_state_types import AppState, AvailableFiles

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig


class ModelsHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock)
        self._config = config

    @staticmethod
    def _path_size(path: Path, is_folder: bool) -> int:
        if not is_folder:
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _scan_available_files(self) -> AvailableFiles:
        files: AvailableFiles = {}
        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            path = self._config.model_path(model_type)
            if spec.is_folder:
                ready = path.exists() and any(path.iterdir()) if path.exists() else False
                files[model_type] = path if ready else None
            else:
                files[model_type] = path if path.exists() else None
        return files

    @with_state_lock
    def refresh_available_files(self) -> AvailableFiles:
        self.state.available_files = self._scan_available_files()
        return self.state.available_files.copy()

    def get_text_encoder_status(self) -> TextEncoderStatus:
        files = self.refresh_available_files()
        text_encoder_path = files["text_encoder"]
        exists = text_encoder_path is not None
        text_spec = self._config.spec_for("text_encoder")
        size_bytes = self._path_size(text_encoder_path, is_folder=True) if exists else 0
        expected = text_spec.expected_size_bytes

        return TextEncoderStatus(
            downloaded=exists,
            size_bytes=size_bytes if exists else expected,
            size_gb=round((size_bytes if exists else expected) / (1024**3), 1),
            expected_size_gb=round(expected / (1024**3), 1),
        )

    def get_models_list(self) -> list[ModelInfo]:
        pro_steps = self.state.app_settings.pro_model.steps
        pro_upscaler = self.state.app_settings.pro_model.use_upscaler
        return [
            ModelInfo(id="fast", name="Fast (Distilled)", description="8 steps + 2x upscaler"),
            ModelInfo(
                id="pro",
                name="Pro (Full)",
                description=f"{pro_steps} steps" + (" + 2x upscaler" if pro_upscaler else " (native resolution)"),
            ),
        ]

    def get_models_status(self, has_api_key: bool | None = None) -> ModelsStatusResponse:
        files = self.refresh_available_files()
        settings = self.state.app_settings.model_copy(deep=True)

        if has_api_key is None:
            has_api_key = bool(settings.ltx_api_key)

        models: list[ModelFileStatus] = []
        total_size = 0
        downloaded_size = 0
        required_types = resolve_required_model_types(
            self._config.required_model_types,
            has_api_key,
            settings.use_local_text_encoder,
        )

        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            path = files[model_type]
            exists = path is not None
            actual_size = self._path_size(path, is_folder=spec.is_folder) if exists else 0
            required = model_type in required_types
            if required:
                total_size += spec.expected_size_bytes
                if exists:
                    downloaded_size += actual_size

            description = spec.description
            optional_reason: str | None = None
            if model_type == "text_encoder":
                description += " (optional with API key)" if has_api_key else ""
                optional_reason = "Uses LTX API for text encoding" if has_api_key else None

            models.append(
                ModelFileStatus(
                    name=spec.name,
                    description=description,
                    downloaded=exists,
                    size=actual_size if exists else spec.expected_size_bytes,
                    expected_size=spec.expected_size_bytes,
                    required=required,
                    is_folder=spec.is_folder,
                    optional_reason=optional_reason if model_type == "text_encoder" else None,
                )
            )

        all_downloaded = all(model.downloaded for model in models if model.required)

        return ModelsStatusResponse(
            models=models,
            all_downloaded=all_downloaded,
            total_size=total_size,
            downloaded_size=downloaded_size,
            total_size_gb=round(total_size / (1024**3), 1),
            downloaded_size_gb=round(downloaded_size / (1024**3), 1),
            models_path=str(self._config.models_dir),
            has_api_key=has_api_key,
            text_encoder_status=self.get_text_encoder_status(),
            use_local_text_encoder=settings.use_local_text_encoder,
        )

    def link_models_from_directory(self, source_dir: str) -> LinkModelsResponse:
        """Scan source_dir for known model files/folders and symlink them into models_dir."""
        src = Path(source_dir)
        if not src.is_dir():
            return LinkModelsResponse(linked=[], notFound=[m for m in MODEL_FILE_ORDER])

        logger = logging.getLogger(__name__)
        linked: list[str] = []
        not_found: list[str] = []

        for model_type in MODEL_FILE_ORDER:
            spec = self._config.spec_for(model_type)
            candidate = src / spec.relative_path
            dest = self._config.model_path(model_type)

            if not candidate.exists():
                not_found.append(model_type)
                continue

            if dest.exists() or dest.is_symlink():
                logger.info("Model %s already exists at %s, skipping", model_type, dest)
                linked.append(model_type)
                continue

            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(candidate.resolve()), str(dest))
                logger.info("Symlinked %s -> %s", dest, candidate)
                linked.append(model_type)
            except OSError:
                # Symlink not supported (e.g. cross-device on Windows) — skip
                logger.warning("Failed to symlink %s, skipping", model_type, exc_info=True)
                not_found.append(model_type)

        self.refresh_available_files()
        return LinkModelsResponse(linked=linked, notFound=not_found)
