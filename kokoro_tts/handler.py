import argparse
import asyncio
import logging
import time

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .kokoro_engine import KokoroEngine

_LOGGER = logging.getLogger(__name__)

class KokoroEventHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, cli_args: argparse.Namespace, engine: KokoroEngine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.cli_args = cli_args
        self.engine = engine
        
        self.sbd = SentenceBoundaryDetector()
        self._current_voice = "af_heart"
        
        # Очередь и воркер
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._audio_started = False
        
        # Флаг для предотвращения дублирования
        self._is_streaming_session = False 

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        # --- 1. Одиночный запрос (Synthesize) ---
        if Synthesize.is_type(event.type):
            # ГЛАВНОЕ: Если мы уже обработали это через стриминг чанков, игнорируем
            if self._is_streaming_session:
                _LOGGER.debug("Ignoring Synthesize event (already handled via streaming)")
                return True
            
            synthesize = Synthesize.from_event(event)
            _LOGGER.info(f"Single synthesis request: {synthesize.text[:50]}...")
            
            self._current_voice = synthesize.voice.name if synthesize.voice else "af_heart"
            self.sbd = SentenceBoundaryDetector() # Сброс буфера
            
            await self._start_worker()
            for s in self.sbd.add_chunk(synthesize.text):
                await self._queue.put(s)
            
            rem = self.sbd.finish()
            if rem: await self._queue.put(rem)
            
            await self._stop_worker()
            return True

        # --- 2. Начало стриминга (SynthesizeStart) ---
        if SynthesizeStart.is_type(event.type):
            self._is_streaming_session = True # Входим в режим стриминга
            
            start = SynthesizeStart.from_event(event)
            self._current_voice = start.voice.name if start.voice else "af_heart"
            _LOGGER.info(f"Text stream STARTED (voice: {self._current_voice})")
            
            self.sbd = SentenceBoundaryDetector() # Свежий детектор для новой сессии
            self._audio_started = False
            await self._start_worker()
            return True

        # --- 3. Чанки текста ---
        if SynthesizeChunk.is_type(event.type):
            if not self._is_streaming_session:
                return True
                
            chunk = SynthesizeChunk.from_event(event)
            for sentence in self.sbd.add_chunk(chunk.text):
                _LOGGER.debug(f"Queuing sentence: {sentence}")
                await self._queue.put(sentence)
            return True

        # --- 4. Конец стриминга ---
        if SynthesizeStop.is_type(event.type):
            if not self._is_streaming_session:
                return True

            _LOGGER.info("Text stream STOP received")
            rem = self.sbd.finish()
            if rem:
                await self._queue.put(rem)
            
            await self._stop_worker()
            
            return True

        return True

    async def _start_worker(self):
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._synthesize_worker())

    async def _stop_worker(self):
        if self._worker_task:
            await self._queue.put(None)
            await self._worker_task
            self._worker_task = None

    async def _synthesize_worker(self):
        """Фоновый синтезатор: берет предложения из очереди."""
        _LOGGER.debug("Synthesis worker started")
        try:
            while True:
                text = await self._queue.get()
                if text is None:
                    break
                
                clean_text = text.strip()
                if not clean_text:
                    continue

                _LOGGER.info(f"Synthesizing: '{clean_text}'")
                start_t = time.perf_counter()
                
                first_chunk = True
                async for pcm_bytes in self.engine.synthesize_stream(clean_text, self._current_voice):
                    if first_chunk:
                        _LOGGER.debug(f"TTFA: {(time.perf_counter()-start_t)*1000:.1f}ms")
                        if not self._audio_started:
                            await self.write_event(AudioStart(rate=self.engine.sample_rate, width=2, channels=1).event())
                            self._audio_started = True
                        first_chunk = False
                    
                    await self.write_event(AudioChunk(audio=pcm_bytes, rate=self.engine.sample_rate, width=2, channels=1).event())

            if self._audio_started:
                await self.write_event(AudioStop().event())
                self._audio_started = False
            
            await self.write_event(SynthesizeStopped().event())

        except Exception as e:
            _LOGGER.exception("Error in synthesis worker")
        finally:
            _LOGGER.debug("Synthesis worker stopped")