# wyoming [kokoro_onnx](https://github.com/thewh1teagle/kokoro-onnx)
Since popular Kokoro implementations for Home Assistant still don't support streaming (synthesis begins when enough chunks have accumulated for one sentence), I'll create my own server version. I'll also experiment with [uv](https://docs.astral.sh/uv/getting-started/installation/).
```
git clone https://github.com/mitrokun/wyoming_kokoro_onnx
cd wyoming_kokoro_onnx
```
Localize the entire cache in the working directory.
```
UV_CACHE_DIR=.uv_cache uvx --from . kokoro-tts
```
If this doesn't matter to you, you can use
```
uvx --from . kokoro-tts
```

The model will be downloaded on the first launch, the server will start on port `10210`.

---
Overall, there are some interesting English voices here, but for other languages ​​I'd look for a different engine. For Spanish, Portuguese, and French, I'd prefer [supertonic](https://github.com/mitrokun/wyoming_supertonic). It has a much better quality/performance ratio.

