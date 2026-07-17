"""Discover chat/analysis models from OpenAI, Gemini, and local LM Studio.

Offline-friendly: LM Studio is always probed first. With no cloud API keys,
a running LM Studio server is enough to populate the Backend list.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Iterable

from ui.settings_dialog import get_gemini_api_key, get_openai_api_key

log = logging.getLogger(__name__)

# Cloud defaults only when those providers are used without a successful API list
FALLBACK_OPENAI = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o4-mini",
]

FALLBACK_GEMINI = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

# Shown only if LM Studio is not reachable *and* no cloud models either
FALLBACK_LMSTUDIO = [
    "LMStudio/local-model",
]

_OPENAI_INCLUDE = re.compile(
    r"^(gpt-4|gpt-5|gpt-3\.5|o1|o3|o4|chatgpt)",
    re.I,
)
_OPENAI_EXCLUDE = re.compile(
    r"(embed|whisper|tts|audio|realtime|transcribe|image|dall-e|moderation|search|instruct)",
    re.I,
)

_GEMINI_EXCLUDE = re.compile(
    r"(embed|aqa|gecko|vision|imagen|tts|robotics|computer-use|exp-|"
    r"image|native-audio|audio|preview-\d|robotics|-live|omni|customtools)",
    re.I,
)

# Embedding / utility models sometimes appear in LM Studio — hide those
_LMSTUDIO_EXCLUDE = re.compile(
    r"(embed|embedding|whisper|tts|clip|nomic-embed)",
    re.I,
)


def _uniq_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        x = (x or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def lmstudio_base_url() -> str:
    return (
        os.environ.get("LMSTUDIO_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "http://127.0.0.1:1234/v1"
    ).rstrip("/")


def fetch_openai_models(api_key: str | None = None) -> tuple[list[str], str, bool]:
    """Return (model_ids, source_note, is_live)."""
    key = (api_key or get_openai_api_key() or "").strip()
    if not key:
        return [], "OpenAI skipped (no API key)", False

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, timeout=15.0)
        models = client.models.list()
        ids: list[str] = []
        for m in models.data:
            mid = getattr(m, "id", None) or ""
            if not mid:
                continue
            if not _OPENAI_INCLUDE.search(mid):
                continue
            if _OPENAI_EXCLUDE.search(mid):
                continue
            ids.append(mid)
        ids = sorted(set(ids), key=lambda s: (0 if "gpt-4o" in s else 1, s))
        if not ids:
            return list(FALLBACK_OPENAI), "OpenAI API: no chat models — defaults", False
        return ids, f"OpenAI API ({len(ids)})", True
    except Exception as e:
        log.warning("OpenAI model list failed: %s", e)
        return list(FALLBACK_OPENAI), f"OpenAI defaults (list failed)", False


def fetch_gemini_models(api_key: str | None = None) -> tuple[list[str], str, bool]:
    """Return (model_ids, source_note, is_live)."""
    key = (api_key or get_gemini_api_key() or "").strip()
    if not key:
        return [], "Gemini skipped (no API key)", False

    try:
        from google import genai

        client = genai.Client(api_key=key)
        ids: list[str] = []
        for m in client.models.list():
            name = getattr(m, "name", None) or getattr(m, "model", None) or ""
            mid = str(name).split("/")[-1].strip()
            if not mid.lower().startswith("gemini"):
                continue
            if _GEMINI_EXCLUDE.search(mid):
                continue
            if not re.search(r"(flash|pro|lite)", mid, re.I):
                continue
            methods = getattr(m, "supported_actions", None) or getattr(
                m, "supported_generation_methods", None
            )
            if methods:
                methods_l = [str(x).lower() for x in methods]
                if methods_l and not any(
                    "generate" in x or "content" in x for x in methods_l
                ):
                    continue
            ids.append(mid)

        def _rank(s: str) -> tuple:
            sl = s.lower()
            return (
                0 if "2.5" in sl else 1 if "2.0" in sl else 2,
                0 if "flash" in sl else 1 if "pro" in sl else 2,
                s,
            )

        ids = sorted(set(ids), key=_rank)
        if not ids:
            return list(FALLBACK_GEMINI), "Gemini API: empty — defaults", False
        return ids, f"Gemini API ({len(ids)})", True
    except Exception as e:
        log.warning("Gemini model list failed: %s", e)
        return list(FALLBACK_GEMINI), "Gemini defaults (list failed)", False


def _lmstudio_list_via_http(base: str, timeout: float = 3.0) -> list[str]:
    """GET {base}/models — works offline without cloud; no OpenAI SDK required."""
    url = f"{base}/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {os.environ.get('LMSTUDIO_API_KEY', 'lm-studio')}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for m in data:
        if not isinstance(m, dict):
            continue
        mid = (m.get("id") or m.get("name") or "").strip()
        if not mid:
            continue
        if _LMSTUDIO_EXCLUDE.search(mid):
            continue
        ids.append(f"LMStudio/{mid}")
    return ids


def fetch_lmstudio_models(timeout: float = 3.0) -> tuple[list[str], str, bool]:
    """List models from a local LM Studio OpenAI-compatible server.

    Works fully offline. Tries raw HTTP first, then the OpenAI client.
    """
    base = lmstudio_base_url()
    api_key = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

    # 1) Lightweight HTTP (preferred offline path)
    try:
        ids = _lmstudio_list_via_http(base, timeout=timeout)
        if ids:
            ids = sorted(set(ids), key=lambda s: s.lower())
            return ids, f"LM Studio @ {base} ({len(ids)} models)", True
    except urllib.error.URLError as e:
        log.debug("LM Studio HTTP list failed: %s", e)
    except Exception as e:
        log.debug("LM Studio HTTP list error: %s", e)

    # 2) OpenAI SDK against local base_url
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base, api_key=api_key, timeout=timeout)
        models = client.models.list()
        ids = []
        for m in models.data:
            mid = getattr(m, "id", None) or ""
            if mid and not _LMSTUDIO_EXCLUDE.search(mid):
                ids.append(f"LMStudio/{mid}")
        if ids:
            ids = sorted(set(ids), key=lambda s: s.lower())
            return ids, f"LM Studio @ {base} ({len(ids)} models)", True
    except Exception as e:
        log.debug("LM Studio SDK list failed: %s", e)

    return (
        [],
        f"LM Studio offline (start server at {base}, load a model, enable local server)",
        False,
    )


def build_backend_list(
    *,
    include_openai: bool = True,
    include_gemini: bool = True,
    include_lmstudio: bool = True,
) -> tuple[list[str], str]:
    """Full combo list + human-readable source summary.

    Order when offline-friendly:
      1) Live LM Studio models (always probed first)
      2) Live Gemini / OpenAI if keys present
      3) Cloud fallbacks only if that cloud provider has a key but list failed
      4) Placeholder LMStudio/local-model only if nothing else at all
    """
    notes: list[str] = []
    items: list[str] = []
    has_live_local = False
    has_live_cloud = False

    # --- Local first (works offline) ---
    if include_lmstudio:
        lm, note, live = fetch_lmstudio_models()
        notes.append(note)
        if live and lm:
            items.extend(lm)
            has_live_local = True

    # --- Cloud (only if keys configured) ---
    if include_gemini:
        gem, note, live = fetch_gemini_models()
        notes.append(note)
        if gem:
            items.extend(gem)
            if live:
                has_live_cloud = True

    if include_openai:
        oai, note, live = fetch_openai_models()
        notes.append(note)
        if oai:
            items.extend(oai)
            if live:
                has_live_cloud = True

    items = _uniq_preserve(items)

    # Nothing at all → tiny offline starter so the combo is never empty
    if not items:
        items = list(FALLBACK_LMSTUDIO)
        if include_gemini and not get_gemini_api_key():
            items = list(FALLBACK_GEMINI) + items
        if include_openai and not get_openai_api_key():
            items = list(FALLBACK_OPENAI) + items
        items = _uniq_preserve(items)
        notes.append("offline starter list (no LM Studio / no live cloud)")

    # Status prefix for the UI strip
    if has_live_local and not has_live_cloud:
        prefix = "Offline OK — "
    elif has_live_local and has_live_cloud:
        prefix = "Local + cloud — "
    elif has_live_cloud:
        prefix = "Cloud — "
    else:
        prefix = ""

    return items, prefix + " | ".join(notes)


def provider_for_model(model_id: str) -> str:
    """Map a combo entry to a provider tag used by call paths."""
    m = (model_id or "").lower()
    if m.startswith("lmstudio") or "lmstudio/" in m:
        return "lmstudio"
    if "gemini" in m:
        return "gemini"
    return "openai"


def model_id_for_api(combo_text: str) -> str:
    """Strip LMStudio/ prefix for the actual API model field."""
    t = (combo_text or "").strip()
    if t.lower().startswith("lmstudio/"):
        return t.split("/", 1)[1]
    return t
