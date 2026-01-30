import logging
import numpy as np
import kokoro_onnx.config

_LOGGER = logging.getLogger(__name__)

class KokoroEngine:
    def __init__(self, model_path: str, voices_path: str):
        self.model_path = model_path
        self.voices_path = voices_path
        self.tts = None
        self.sample_rate = kokoro_onnx.config.SAMPLE_RATE
        self.available_voices = []

    def load(self):
        _LOGGER.info("Loading Kokoro library and models...")
        from kokoro_onnx import Kokoro
        self.tts = Kokoro(self.model_path, self.voices_path)
        self.available_voices = list(self.tts.voices.keys())
        _LOGGER.info(f"Kokoro engine ready. Sample rate: {self.sample_rate}Hz")

    async def synthesize_stream(self, text: str, voice_name: str):
        """Асинхронно генерирует аудио-чанки для одного предложения."""
        if self.tts is None:
            raise RuntimeError("Engine is not loaded!")
            
        lang = "en-us"
        if voice_name.startswith("b"): lang = "en-gb"
        elif voice_name.startswith("i"): lang = "it"
        elif voice_name.startswith("j"): lang = "jp"
        elif voice_name.startswith("z"): lang = "cn"
        elif voice_name.startswith("e"): lang = "es"
        elif voice_name.startswith("f"): lang = "fr"
        elif voice_name.startswith("h"): lang = "hi"

        audio_stream = self.tts.create_stream(
            text,
            voice=voice_name,
            speed=1.0,
            lang=lang
        )
        
        async for audio_chunk, _ in audio_stream:
            yield (audio_chunk * 32767).astype(np.int16).tobytes()