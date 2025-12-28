"""
OuteTTS 1.0-1B Predictor for Replicate.

Runs the OuteAI/Llama-OuteTTS-1.0-1B model for text-to-speech synthesis
with support for 23 languages including Persian/Farsi.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from cog import BasePredictor, Input, Path as CogPath

# Model will be downloaded during build via script/download_weights
MODEL_ID = "OuteAI/Llama-OuteTTS-1.0-1B"
CACHE_DIR = "./weights"

# OuteTTS 1.0-1B only includes one default speaker
DEFAULT_SPEAKER = "EN-FEMALE-1-NEUTRAL"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory."""
        import outetts

        # Use HuggingFace backend with cached weights
        config = outetts.ModelConfig.auto_config(
            model=outetts.Models.VERSION_1_0_SIZE_1B,
            backend=outetts.Backend.HF,
        )

        self.interface = outetts.Interface(config=config)

        # Pre-load default speaker
        self.default_speaker = self.interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    def predict(
        self,
        text: str = Input(
            description="Text to synthesize. Supports 23 languages including Persian/Farsi with native script.",
            default="سلام، حال شما چطور است؟",
        ),
        speaker: str = Input(
            description="Speaker voice ID. Currently only EN-FEMALE-1-NEUTRAL is available.",
            default="EN-FEMALE-1-NEUTRAL",
        ),
        temperature: float = Input(
            description="Sampling temperature. Lower = more deterministic, higher = more varied.",
            default=0.4,
            ge=0.1,
            le=1.5,
        ),
    ) -> CogPath:
        """Generate speech from text."""
        import outetts

        # Use the only available default speaker
        if speaker.upper() != "EN-FEMALE-1-NEUTRAL":
            raise ValueError(f"Speaker {speaker!r} not available. Only EN-FEMALE-1-NEUTRAL is supported.")
        spk = self.default_speaker

        # Generate audio
        generation = self.interface.generate(
            config=outetts.GenerationConfig(
                text=text,
                generation_type=outetts.GenerationType.CHUNKED,
                speaker=spk,
                sampler_config=outetts.SamplerConfig(temperature=temperature),
            )
        )

        # Save to temp file
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        generation.save(str(output_path))

        return CogPath(output_path)
