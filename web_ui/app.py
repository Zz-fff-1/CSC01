"""多模态纠错 Web UI：默认 Transformers（LlamaFactory ChatModel）；可选 Ollama。"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_utils import load_inference_config  # noqa: E402
from lf_chat_infer import LfChatInference, apply_no_lora_ocr_asr, build_text_eval_config  # noqa: E402

BACKEND = os.getenv("CORRECTION_BACKEND", "transformers").lower().strip()
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

TEXT_YAML = _ROOT / "train_config/text_correction_inference.yaml"
OCR_YAML = _ROOT / "train_config/ocr_vl_inference.yaml"
ASR_YAML = _ROOT / "train_config/asr_correction_inference.yaml"

TEXT_SYSTEM = (
    "你是一个中文文本纠错助手。用户会提供可能含有错误的中文文本，请纠正其中的错误，只输出纠错后的正确文本。"
)
OCR_SYSTEM = "你是一个OCR文字识别助手。请识别图片中的中文文字，按顺序输出。"
OCR_USER = "<image>请识别这张图片中的所有中文文字。"
ASR_SYSTEM = (
    "你是一个ASR文本纠错助手。用户会提供一段语音和对应的ASR识别文本，请根据语音内容纠正ASR文本中的错误，只输出纠错后的正确文本。"
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class _TransformersRunnerHolder:
    """按 (mode, use_lora) 缓存 ChatModel（HuggingFace 后端）。"""

    def __init__(self) -> None:
        self._key: str | None = None
        self._runner: LfChatInference | None = None

    def get(self, mode: str, use_lora: bool) -> LfChatInference:
        key = f"{mode}|{int(use_lora)}"
        if self._key == key and self._runner is not None:
            return self._runner
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        self._key = key
        raw = load_inference_config(str(_yaml_for_mode(mode)))
        if mode == "text":
            cfg = build_text_eval_config(raw, no_lora=not use_lora)
        else:
            cfg = apply_no_lora_ocr_asr(raw, no_lora=not use_lora)
        self._runner = LfChatInference(cfg)
        return self._runner


def _yaml_for_mode(mode: str) -> Path:
    return {"text": TEXT_YAML, "ocr": OCR_YAML, "asr": ASR_YAML}[mode]


_tf_holder = _TransformersRunnerHolder()


def _parse_use_lora(s: str) -> bool:
    return str(s).lower() in ("1", "true", "yes", "on")


def _parse_max_tokens(s: str, default: int = 512) -> int:
    try:
        v = int(s)
        return max(1, min(v, 8192))
    except (TypeError, ValueError):
        return default


def _ollama_chat_sync(model: str, messages: list[dict[str, Any]], options: dict[str, Any] | None = None) -> str:
    url = f"{OLLAMA_BASE}/api/chat"
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if options:
        body["options"] = options
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {err_body or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"无法连接 Ollama ({OLLAMA_BASE}): {e.reason}") from e

    msg = payload.get("message") or {}
    return (msg.get("content") or "").strip()


async def _ollama_chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    images_b64: list[str] | None = None,
    num_predict: int | None = None,
) -> str:
    msgs: list[dict[str, Any]] = [dict(m) for m in messages]
    if images_b64:
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                u = dict(msgs[i])
                u["images"] = images_b64
                msgs[i] = u
                break
    opts = {}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    return await asyncio.to_thread(_ollama_chat_sync, model, msgs, opts or None)


app = FastAPI(title="Multi-model Chinese Spelling Correction")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "backend": BACKEND, "ollama_base": OLLAMA_BASE if BACKEND == "ollama" else None}


@app.get("/api/config")
def api_config():
    rel = lambda p: str(p.relative_to(_ROOT))  # noqa: E731
    return {
        "backend": BACKEND,
        "ollama_base": OLLAMA_BASE,
        "default_models": {
            "text": os.getenv("OLLAMA_MODEL_TEXT", "").strip(),
            "ocr": os.getenv("OLLAMA_MODEL_OCR", "").strip(),
            "asr": os.getenv("OLLAMA_MODEL_ASR", "").strip(),
        },
        "inference_configs": {
            "text": rel(TEXT_YAML),
            "ocr": rel(OCR_YAML),
            "asr": rel(ASR_YAML),
        },
    }


@app.post("/api/correct")
async def correct(
    mode: str = Form(...),
    model: str = Form(""),
    text: str = Form(""),
    max_tokens: str = Form("512"),
    use_lora: str = Form("true"),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
):
    mode = mode.lower().strip()
    if mode not in ("text", "ocr", "asr"):
        raise HTTPException(status_code=400, detail="mode 须为 text | ocr | asr")
    n_tok = _parse_max_tokens(max_tokens)

    if BACKEND == "ollama":
        m = model.strip()
        if not m:
            raise HTTPException(status_code=400, detail="Ollama 模式下请填写模型名，例如 qwen2.5:7b")
        try:
            if mode == "text":
                if not text.strip():
                    raise HTTPException(status_code=400, detail="请输入待纠错文本")
                messages = [
                    {"role": "system", "content": TEXT_SYSTEM},
                    {
                        "role": "user",
                        "content": f"请对以下中文文本进行纠错，只输出纠错后的正确文本：\n{text.strip()}",
                    },
                ]
                out = await _ollama_chat(m, messages, num_predict=n_tok)
                return {"ok": True, "result": out, "mode": mode, "backend": "ollama", "model": m}

            if mode == "ocr":
                if image is None or not image.filename:
                    raise HTTPException(status_code=400, detail="请上传图片")
                raw = await image.read()
                b64 = base64.b64encode(raw).decode("ascii")
                messages = [
                    {"role": "system", "content": OCR_SYSTEM},
                    {"role": "user", "content": "请识别这张图片中的所有中文文字，按顺序输出。"},
                ]
                out = await _ollama_chat(m, messages, images_b64=[b64], num_predict=min(n_tok, 1024))
                return {"ok": True, "result": out, "mode": mode, "backend": "ollama", "model": m}

            if not text.strip():
                raise HTTPException(status_code=400, detail="请填写 ASR 待纠错文本")
            messages = [
                {"role": "system", "content": ASR_SYSTEM},
                {
                    "role": "user",
                    "content": f"以下为语音识别文本稿，请纠错后只输出正确全文：\n{text.strip()}",
                },
            ]
            out = await _ollama_chat(m, messages, num_predict=n_tok)
            return {"ok": True, "result": out, "mode": mode, "backend": "ollama", "model": m}

        except HTTPException:
            raise
        except Exception as e:
            return JSONResponse(status_code=502, content={"ok": False, "error": str(e)})

    # ---------- Transformers（默认）----------
    use_lora_b = _parse_use_lora(use_lora)
    try:
        runner = _tf_holder.get(mode, use_lora_b)
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"模型加载失败: {e}"})

    try:
        if mode == "text":
            if not text.strip():
                raise HTTPException(status_code=400, detail="请输入待纠错文本")
            messages = [
                {"role": "system", "content": TEXT_SYSTEM},
                {
                    "role": "user",
                    "content": f"请对以下中文文本进行纠错，只输出纠错后的正确文本：\n{text.strip()}",
                },
            ]
            out = runner.predict(messages, max_new_tokens=n_tok, do_sample=False)
            return {"ok": True, "result": out, "mode": mode, "backend": "transformers", "use_lora": use_lora_b}

        if mode == "ocr":
            if image is None or not image.filename:
                raise HTTPException(status_code=400, detail="请上传图片")
            suffix = Path(image.filename).suffix or ".png"
            fd, path = tempfile.mkstemp(suffix=suffix, prefix="ocr_")
            os.close(fd)
            try:
                data = await image.read()
                with open(path, "wb") as f:
                    f.write(data)
                messages = [
                    {"role": "system", "content": OCR_SYSTEM},
                    {"role": "user", "content": OCR_USER},
                ]
                out = runner.predict(
                    messages,
                    images=[path],
                    max_new_tokens=min(n_tok, 512),
                    do_sample=False,
                )
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            return {"ok": True, "result": out, "mode": mode, "backend": "transformers", "use_lora": use_lora_b}

        if audio is None or not audio.filename:
            raise HTTPException(status_code=400, detail="请上传音频")
        if not text.strip():
            raise HTTPException(status_code=400, detail="请填写 ASR 待纠错文本")

        suffix = Path(audio.filename).suffix or ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="asr_")
        os.close(fd)
        try:
            data = await audio.read()
            with open(path, "wb") as f:
                f.write(data)
            messages = [
                {"role": "system", "content": ASR_SYSTEM},
                {"role": "user", "content": f"<audio>根据语音对ASR文本纠错:\nASR文本: {text.strip()}"},
            ]
            out = runner.predict(messages, audios=[path], max_new_tokens=n_tok, do_sample=False)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return {"ok": True, "result": out, "mode": mode, "backend": "transformers", "use_lora": use_lora_b}

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
