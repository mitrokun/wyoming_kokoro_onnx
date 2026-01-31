import argparse
import asyncio
import logging
from functools import partial
import os
import urllib.request

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from .kokoro_engine import KokoroEngine
from .handler import KokoroEventHandler

_LOGGER = logging.getLogger(__name__)
__version__ = "1.0.0"

# Ссылки на модели
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

def download_file(url, destination):
    """Скачивает файл, если его нет."""
    if os.path.exists(destination):
        _LOGGER.debug(f"File {destination} already exists, skipping download.")
        return
    _LOGGER.info(f"Downloading {destination} from {url}...")
    urllib.request.urlretrieve(url, destination)
    _LOGGER.info(f"Done downloading {destination}")

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10210", help="Server URI")
    parser.add_argument("--model", default="kokoro-v1.0.onnx", help="Path to ONNX model")
    parser.add_argument("--voices-bin", dest="voices", default="voices-v1.0.bin", help="Path to voices bin")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming support")
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument("--log-format", default="%(asctime)s %(levelname)s:%(name)s:%(message)s", help="Log format")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)

    try:
        download_file(MODEL_URL, args.model)
        download_file(VOICES_URL, args.voices)
    except Exception as e:
        _LOGGER.error(f"Failed to download models: {e}")
        return

    _LOGGER.info("Initializing Kokoro Engine...")
    engine = KokoroEngine(model_path=args.model, voices_path=args.voices)
    
    await asyncio.to_thread(engine.load)

    wyoming_voices = []
    for voice_id in engine.available_voices:
        lang_map = {
            "a": "en-us", "b": "en-gb", "i": "it", "j": "ja", 
            "z": "zh", "e": "es", "f": "fr", "h": "hi"
        }
        lang_code = lang_map.get(voice_id[0], "en-us")
        
        wyoming_voices.append(
            TtsVoice(
                name=voice_id,
                description=f"{voice_id}",
                attribution=Attribution(name="Hexgrad", url="https://huggingface.co/hexgrad/Kokoro-82M"),
                installed=True,
                version=__version__,
                languages=[lang_code],
            )
        )

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="Kokoro",
                description="Kokoro ONNX TTS",
                attribution=Attribution(name="Hexgrad", url="https://huggingface.co/hexgrad/Kokoro-82M"),
                installed=True,
                version=__version__,
                supports_synthesize_streaming=args.streaming,
                voices=wyoming_voices,
            )
        ],
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"Server ready at {args.uri}")

    await server.run(
        partial(
            KokoroEventHandler,
            wyoming_info,
            args,
            engine,
        )
    )

def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run()
