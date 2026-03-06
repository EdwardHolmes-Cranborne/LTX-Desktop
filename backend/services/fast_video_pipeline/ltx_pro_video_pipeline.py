"""LTX pro video pipeline wrapper using TI2VidTwoStagesPipeline."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8, is_mps_device


class LTXProVideoPipeline:
    pipeline_kind: Final = "pro"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        distill_lora_path: str,
        device: torch.device,
        user_loras: list[tuple[str, float]] | None = None,
    ) -> "LTXProVideoPipeline":
        return LTXProVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            distill_lora_path=distill_lora_path,
            device=device,
            user_loras=user_loras,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        distill_lora_path: str,
        device: torch.device,
        user_loras: list[tuple[str, float]] | None = None,
    ) -> None:
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        distill_lora = [LoraPathStrengthAndSDOps(distill_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]

        loras: list[LoraPathStrengthAndSDOps] = []
        if user_loras:
            for path, strength in user_loras:
                loras.append(LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP))

        # On MPS, torch.cuda.synchronize() in TI2VidTwoStagesPipeline will crash.
        # Patch it to no-op before creating the pipeline.
        if is_mps_device(device) and not torch.cuda.is_available():
            torch.cuda.synchronize = lambda *args, **kwargs: None  # type: ignore[assignment]

        self.pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distill_lora,
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=loras,
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None,
        )
        self._num_inference_steps = 30

    @property
    def num_inference_steps(self) -> int:
        return self._num_inference_steps

    @num_inference_steps.setter
    def num_inference_steps(self, value: int) -> None:
        self._num_inference_steps = value

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        video_guider = MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[28],
        )
        audio_guider = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=1.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            skip_step=0,
            stg_blocks=[28],
        )

        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
            images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
            tiling_config=tiling_config,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int | None = None,
    ) -> None:
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT

        neg = negative_prompt or DEFAULT_NEGATIVE_PROMPT
        steps = num_inference_steps or self._num_inference_steps
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            negative_prompt=neg,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=steps,
            images=images,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                negative_prompt="",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                num_inference_steps=4,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        transformer = self.pipeline.stage_1_model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),  # type: ignore[reportUnknownMemberType]
        )
        setattr(self.pipeline.stage_1_model_ledger, "transformer", lambda: compiled)
