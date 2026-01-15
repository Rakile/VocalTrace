from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import json
import sys
from openai import OpenAI
from langdetect import detect, LangDetectException
import argparse
import os


# ---------------------------------------------------------------------------
# 1) Preclass-prompt (körs på gpt-5-nano)
# ---------------------------------------------------------------------------

PRECLASS_PROMPT = """
You are a classifier that analyzes conversation transcripts and decides what level of model is needed for analysis.

Return ONLY a JSON object with exactly these keys:

{{
  "emotional_intensity": 1-5,
  "relationship_complexity": 1-5,
  "conflict_level": 1-5,
  "topic_type": "familj | vårdnad | försäljning | support | tekniskt | business | smalltalk | konflikt | okänt",
  "requires_deep_psychology": true/false,
  "recommendation": "nano | mini | gpt5.1",
  "notes": "short natural language explanation (max 2 sentences)"
}}

Guidelines:

- emotional_intensity: 1 = neutral, 5 = very strong emotions (grief, anger, fear, guilt, shame, etc.)
- relationship_complexity: 1 = simple (e.g. seller–customer), 5 = complex (multiple family relations, roles, power dynamics)
- conflict_level: 1 = no conflict, 5 = strong conflict/custody battle/explicit confrontation
- requires_deep_psychology: true if the conversation involves family, children, custody, therapy, trauma, strong relational conflicts or sensitive personal psychology.
- recommendation:
  - "nano" for simple, neutral, technical or purely informational conversations.
  - "mini" for customer service, sales, business, and lighter relational content.
  - "gpt5.1" for family, custody, strong relational conflicts, children, couples, or emotionally sensitive topics.

Now classify this transcript:

[TRANSCRIPT START]
{transcript}
[TRANSCRIPT END]
"""


@dataclass
class PreclassResult:
    emotional_intensity: int
    relationship_complexity: int
    conflict_level: int
    topic_type: str
    requires_deep_psychology: bool
    recommendation: str
    raw: Dict[str, Any]


# ---------------------------------------------------------------------------
# 2) ModelSelector
# ---------------------------------------------------------------------------

class ModelSelector:
    """
    Generic model selector.

    You provide:
      - a `call_llm` function:
          call_llm(model_name: str, prompt: str) -> str  (raw text from model)

    This class:
      - runs a cheap pre-scan on nano
      - parses result
      - applies fallback rules
      - returns the chosen main model name (e.g. "gpt-5-nano", "gpt-5-mini", "gpt-5.1")
    """

    def __init__(
        self,
        call_llm: Callable[[str, str], str],
        nano_model: str = "gpt-5-nano",
        mini_model: str = "gpt-5-mini",
        deep_model: str = "gpt-5.1",
    ) -> None:
        self.call_llm = call_llm
        self.nano_model = nano_model
        self.mini_model = mini_model
        self.deep_model = deep_model

    # ---------- public API ----------
    def select_model_mockup(self, transcript: str) -> Dict[str, Any]:
        reason = (
            f"Chose grok-4-1-reasoning based on: "
            f"emotional_intensity=0, "
            f"relationship_complexity=0, "
            f"conflict_level=0, "
            f"topic_type=0, "
            f"requires_deep_psychology=yes&no, "
            f"model_recommendation=xxx"
        )

        return {
            "chosen_model": "grok-4-1-reasoning",
            "preclass": "Xxx",
            "reason": reason,
        }
    def select_model(self, transcript: str) -> Dict[str, Any]:
        """
        Run preclassification on nano, then decide which main model to use.

        Returns:
            {
              "chosen_model": "...",
              "preclass": PreclassResult,
              "reason": "short human explanation"
            }
        """
        pre = self._run_preclass(transcript)
        chosen = self._decide(pre)

        reason = (
            f"Chose {chosen} based on: "
            f"emotional_intensity={pre.emotional_intensity}, "
            f"relationship_complexity={pre.relationship_complexity}, "
            f"conflict_level={pre.conflict_level}, "
            f"topic_type={pre.topic_type}, "
            f"requires_deep_psychology={pre.requires_deep_psychology}, "
            f"model_recommendation={pre.recommendation!r}"
        )

        return {
            "chosen_model": chosen,
            "preclass": pre,
            "reason": reason,
        }

    # ---------- internal helpers ----------

    def _run_preclass(self, transcript: str) -> PreclassResult:
        prompt = PRECLASS_PROMPT.format(transcript=transcript)
        raw_text = self.call_llm(self.nano_model, prompt)

        # Try to extract JSON; be robust to extra text
        try:
            json_str = self._extract_json(raw_text)
            data = json.loads(json_str)
        except Exception as e:
            raise RuntimeError(f"Failed to parse preclass JSON: {e}\nRaw text:\n{raw_text}")

        return PreclassResult(
            emotional_intensity=int(data.get("emotional_intensity", 3)),
            relationship_complexity=int(data.get("relationship_complexity", 3)),
            conflict_level=int(data.get("conflict_level", 3)),
            topic_type=str(data.get("topic_type", "okänt")),
            requires_deep_psychology=bool(data.get("requires_deep_psychology", False)),
            recommendation=str(data.get("recommendation", "mini")),
            raw=data,
        )

    def _decide(self, pre: PreclassResult) -> str:
        """
        Core decision logic. Tweak thresholds as you like.
        """

        # Hard override: deep psychology flag
        if pre.requires_deep_psychology:
            return self.deep_model

        # Strong emotional/relational content → deep model
        if pre.emotional_intensity >= 4 or pre.relationship_complexity >= 4:
            return self.deep_model

        # Family / custody / conflict topics → deep model
        if pre.topic_type in ["familj", "vårdnad", "konflikt"]:
            return self.deep_model

        # For explicitly simple/neutral topics → nano
        if pre.topic_type in ["tekniskt", "support", "smalltalk"] and pre.conflict_level <= 2:
            return self.nano_model

        # Sales / business / support → mini is usually sweet spot
        if pre.topic_type in ["försäljning", "business"] and pre.conflict_level <= 3:
            return self.mini_model

        # Fallback to model’s own recommendation if consistent
        if pre.recommendation == "nano":
            return self.nano_model
        if pre.recommendation == "mini":
            return self.mini_model
        if pre.recommendation == "gpt5.1":
            return self.deep_model

        # Ultimate fallback
        return self.mini_model

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Try to find a JSON object in the model output.
        If the whole string is JSON, returns it as-is.
        If there is extra text, tries to find the first '{' and last '}'.
        """
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return text

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

        raise ValueError("Could not extract JSON from model output")


# ---------------------------------------------------------------------------
# 3) Analys-prompt (huvudmodellen – mini eller gpt-5.1)
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE_EN = """
You are a professional conversation analyst. You must analyze the transcript below
and return the result in strict JSON format.

📌 HARD REQUIREMENTS – MUST BE FOLLOWED:

1. No text outside the JSON. No comments. No explanations.
2. JSON must be 100% valid.
3. Never use real names, even if they appear in the transcript.
4. Identify speakers as: "person_1", "person_2", "person_3", etc.
5. Assign roles when they are identifiable (e.g., "host", "guest", "customer", "seller", "manager").
6. Never use direct quotes from the transcript.
7. Always separate:
   - **Facts:** What actually happens or is said.
   - **Interpretation:** Your analysis of meaning, intentions, emotions, relationships, or implications.
8. Do not output real email addresses or other PII.
   If such details appear in the transcript: write "anonymized email address".
9. No content may be omitted – if something is missing, use an empty string or empty list.
10. Output MUST follow exactly the JSON structure below.

📄 JSON STRUCTURE (USE EXACTLY THIS):

{{
  "conversation_type": "",
  "participants": [
    {{
      "id": "",
      "role": "",
      "facts": "",
      "interpretation": ""
    }}
  ],
  "overview": {{
    "facts": "",
    "interpretation": ""
  }},
  "key_facts": [
    {{
      "facts": "",
      "interpretation": ""
    }}
  ],
  "relational_dynamics": [
    {{
      "relation": "",
      "facts": "",
      "interpretation": ""
    }}
  ],
  "emotional_aspects": {{
    "direct_expressions": [
      {{
        "person": "",
        "facts": "",
        "interpretation": ""
      }}
    ],
    "possible_underlying": [
      {{
        "facts": "",
        "interpretation": ""
      }}
    ]
  }},
  "themes": [
    {{
      "theme": "",
      "description": "",
      "example": ""
    }}
  ],
  "tone_and_progression": "",
  "summary_short": "",
  "recommended_ai_persona": "A 1-sentence instruction for an AI assistant answering questions about this specific text. Example: 'You are a forensic accountant analyzing financial discrepancies.'"
}}

---

🎯 GOALS OF THE ANALYSIS:
- Adapt the analysis to the type of conversation (customer support, family, meeting, sales, interview, conflict, casual talk, podcast, etc.).
- Identify the most important facts and interaction patterns.
- Identify relationship dynamics and emotional signals.
- Provide a neutral, concise, professional analysis.

---

Here is the transcript:

[TRANSCRIPT START]
{transcript}
[TRANSCRIPT END]
"""

ANALYSIS_PROMPT_TEMPLATE_SV = """
Du är en professionell samtalsanalytiker. Du ska analysera transkriptionen nedan 
och returnera resultatet i strikt JSON-format.

📌 HÅRDA KRAV – MÅSTE FÖLJAS:

1. Ingen text utanför JSON. Inga kommentarer. Ingen förklaring.
2. JSON måste vara 100% giltigt.
3. Använd ALDRIG riktiga namn, även om de finns i transkriptionen.
4. Identifiera talare som: "person_1", "person_2", "person_3", osv.
5. Ange roll om den är tydlig (t.ex. "pappa", "barn", "kund", "säljare", "chef").
6. Inga direkta citat från transkriptet.
7. Skilj alltid mellan:
   - **Fakta:** Det som faktiskt händer eller sägs.
   - **Tolkning:** Din analys av innebörd, relation, intention eller känsla.
8. Inga riktiga e-postadresser eller annan PII får återges.
   Om de förekommer i transkriptet: skriv "anonymiserad e-postadress".
9. Inget innehåll får utelämnas – om något saknas, använd en tom sträng eller tom lista.
10. Output måste följa exakt JSON-strukturen nedan.

📄 JSON-struktur (ANVÄND EXAKT DENNA):

{{
  "conversation_type": "",  
  "participants": [
    {{
      "id": "",
      "roll": "",
      "fakta": "",
      "tolkning": ""
    }}
  ],
  "overview": {{
    "fakta": "",
    "tolkning": ""
  }},
  "key_facts": [
    {{
      "fakta": "",
      "tolkning": ""
    }}
  ],
  "relational_dynamics": [
    {{
      "relation": "",
      "fakta": "",
      "tolkning": ""
    }}
  ],
  "emotional_aspects": {{
    "direct_expressions": [
      {{
        "person": "",
        "fakta": "",
        "tolkning": ""
      }}
    ],
    "possible_underlying": [
      {{
        "fakta": "",
        "tolkning": ""
      }}
    ]
  }},
  "themes": [
    {{
      "theme": "",
      "description": "",
      "example": ""
    }}
  ],
 "summary_short": "",
 "rekommenderad_ai_persona": "En instruktion på 1 mening för en AI-assistent som svarar på frågor om just denna text. Exempel: 'Du är en forensisk expert som analyserar vittnesmål i en rättegång.'"
}}

---

🎯 MÅL MED ANALYSEN:
- Anpassa analysen efter vilken typ av samtal det är (kundtjänst, familj, möte, försäljning, intervju, konflikt, vardagssamtal, podcast etc).
- Identifiera de viktigaste fakta och mönster i interaktionen.
- Identifiera relationella dynamiker och känslomässiga spår.
- Ge en neutral, kortfattad, professionell analys.

---

Här följer transkriptionen:

[TRANSKRIPTION BÖRJAR]
{transcript}
[TRANSKRIPTION SLUT]
"""


# ---------------------------------------------------------------------------
# 4) Hög-nivå funktion: kör hela pipelinen
# ---------------------------------------------------------------------------

@dataclass
class FullAnalysisResult:
    chosen_model: str
    preclass: Optional[PreclassResult]
    preclass_reason: str
    analysis_json: Dict[str, Any]
    analysis_raw_text: str


def _extract_json(text: str) -> str:
    """
    Reuse JSON-extraktion även för analys-steget.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise ValueError("Could not extract JSON from model output")


def analyze_conversation(
    transcript: str,
    call_llm: Callable[[str, str], str],
    nano_model: str = "gpt-5-nano",
    mini_model: str = "gpt-5-mini",
    deep_model: str = "gpt-5.1",
    analysis_language: str = "auto",
    backend: str = "openai",  # NEW
) -> FullAnalysisResult:
    """
    Kör hela kedjan:
      1) pre-scan på nano
      2) välj huvudmodell
      3) kör analys-prompt
      4) returnera strukturerat resultat
    """

    # 1) Välj modell
    selector = ModelSelector(
        call_llm=call_llm,
        nano_model=nano_model,
        mini_model=mini_model,
        deep_model=deep_model,
    )
    if backend.lower().startswith("lmstudio"):
        chosen_model = backend.lower().replace("lmstudio", "openai")
        pre = None
        reason = f"Using LMStudio backend '{backend}'; skipping GPT model selection."
    elif backend.lower().startswith("openai") or "gpt" in backend.lower(): # backend.lower() in ("gpt", "chatgpt"):
        print("[i] Preclass: calling gpt-5-nano (to decide what model should be used)")
        sel = selector.select_model(transcript)
        #sel = selector.select_model_mockup(transcript)

        chosen_model: str = sel["chosen_model"]
        pre: PreclassResult = sel["preclass"]
        reason: str = sel["reason"]
    else:
        # NEW: non-OpenAI backend (e.g. Gemini) – skip GPT model selection
        pre = None
        if backend.lower().startswith("gemini"):
            chosen_model = backend
            reason = f"Using Gemini backend ({chosen_model}); skipping GPT model selection."
        else:
            # fallback: whatever backend string you passed in
            chosen_model = backend
            reason = f"Using fixed backend '{backend}'; skipping GPT model selection."

    # Decide which language to use for the analysis prompt
    if analysis_language == "auto":
        lang = detect_language_from_transcript(transcript)
        print(f"[i] Detected transcript language (text): {lang}")
    else:
        lang = analysis_language

    # 2) Bygg analys-prompt
    if lang.lower().startswith("sv"):
        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE_SV.format(transcript=transcript)
    else:
        analysis_prompt = ANALYSIS_PROMPT_TEMPLATE_EN.format(transcript=transcript)



    #analysis_prompt = ANALYSIS_PROMPT_TEMPLATE.format(transcript=transcript)

    # 3) Kör huvudmodellen
    print(f"[i] Chosen model: {chosen_model}")
    print(f"[i] Main analysis: calling {chosen_model} to analyze transcript")
    raw_analysis = call_llm(chosen_model, analysis_prompt)

    # 4) Försök parsa JSON
    analysis_json_str = _extract_json(raw_analysis)
    analysis_data = json.loads(analysis_json_str)

    return FullAnalysisResult(
        chosen_model=chosen_model,
        preclass=pre,
        preclass_reason=reason,
        analysis_json=analysis_data,
        analysis_raw_text=raw_analysis,
    )


# ---------------------------------------------------------------------------
# 5) Exempel på hur du kopplar in din LLM-klient
# ---------------------------------------------------------------------------

# Detta är bara en stub – byt ut mot din riktiga klient.
def call_llm_openai_stub(model_name: str, prompt: str) -> str:
    """
    EXEMPEL – byt ut mot din riktiga implementation.

    T.ex. om du använder OpenAI:
    ------------------------------
    from openai import OpenAI
    client = OpenAI()

    def call_llm_openai(model_name: str, prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    """
    raise NotImplementedError("Replace call_llm_openai_stub with real implementation")

def call_llm_openai(model_name: str, prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.responses.create(
        model=model_name,
        input=prompt,
        #service_tier="flex"
    )

    # Plocka ut texten – kontrollera ev. struktur i ditt eget projekt
    parts = []
    for item in response.output:
        if hasattr(item, "content"):
            if item.content:
                for c in item.content:
                    if hasattr(c, "type"):
                            if c.type == "output_text":
                                parts.append(c.text)
    return "\n".join(parts)

base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
api_key = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
def call_llm_lmstudio(model_name: str, prompt: str) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.responses.create(
        model=model_name,
        input=prompt,
        #service_tier="flex"
    )

    # Plocka ut texten – kontrollera ev. struktur i ditt eget projekt
    parts = []
    for item in response.output:
        if hasattr(item, "content"):
            if item.content:
                for c in item.content:
                    if hasattr(c, "type"):
                            if c.type == "output_text":
                                parts.append(c.text)
    return "\n".join(parts)

def call_llm_grokai(model_name: str, prompt: str) -> str:
    client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.getenv("GROK_API_KEY"))
    response = client.responses.create(
        model=model_name,
        input=prompt,
    )

    # Plocka ut texten – kontrollera ev. struktur i ditt eget projekt
    parts = []
    for item in response.output:
        if hasattr(item, "content"):
            if item.content:
                for c in item.content:
                    if hasattr(c, "type"):
                            if c.type == "output_text":
                                parts.append(c.text)
    return "\n".join(parts)

import os


# In conversation_analysis_pipeline.py

def call_llm_gemini_25_pro(model_name: str, prompt: str) -> str:
    """
    Dynamic Gemini Wrapper with Error Handling.
    """
    from google import genai
    from google.genai import types, errors

    api_key = os.getenv("GEMINI_TRANSCRIBE_ANALYSIS_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    # Only apply thinking budget if it's a "Pro" or "Thinking" model
    # (Flash usually ignores it, but safer to skip)
    use_thinking = "pro" in model_name.lower() or "thinking" in model_name.lower()

    config = None
    if use_thinking:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024)
        )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )

        if not response.text:
            return "Error: Model returned empty response."

        return response.text

    except errors.ClientError as e:
        # Catch the 429 Resource Exhausted
        if e.code == 429 or "RESOURCE_EXHAUSTED" in str(e):
            raise RuntimeError(
                f"Gemini API Quota Exceeded for '{model_name}'.\n\n"
                "The limit for this model is currently 0 or exhausted.\n"
                "SUGGESTION: Switch the Backend to a 'Flash' model (e.g. gemini-2.5-flash) "
                "which has significantly higher free tier limits."
            )
        else:
            raise RuntimeError(f"Gemini API Error: {e.message}")

    except Exception as e:
        raise RuntimeError(f"Unexpected Error: {str(e)}")

def normalize_analysis_schema(data: dict) -> dict:
    """
    Normalize analysis JSON from either Swedish-key or English-key schema
    into a canonical English-key schema.

    Expects top-level keys like:
      conversation_type, participants, overview, key_facts,
      relational_dynamics, emotional_aspects, themes, tone_and_progression, summary_short
    """

    out = dict(data)  # shallow copy so we don't mutate input

    # --- Participants ---
    norm_participants = []
    for p in data.get("participants", []):
        pid  = p.get("id")
        role = p.get("role") or p.get("roll")
        facts = p.get("facts") or p.get("fakta")
        interp = p.get("interpretation") or p.get("tolkning")

        norm_participants.append({
            "id": pid or "",
            "role": role or "",
            "facts": facts or "",
            "interpretation": interp or "",
        })
    out["participants"] = norm_participants

    # --- Overview ---
    ov = data.get("overview", {}) or {}
    out["overview"] = {
        "facts": ov.get("facts") or ov.get("fakta") or "",
        "interpretation": ov.get("interpretation") or ov.get("tolkning") or "",
    }

    # --- Key facts ---
    norm_key_facts = []
    for k in data.get("key_facts", []):
        facts = k.get("facts") or k.get("fakta")
        interp = k.get("interpretation") or k.get("tolkning")
        norm_key_facts.append({
            "facts": facts or "",
            "interpretation": interp or "",
        })
    out["key_facts"] = norm_key_facts

    # --- Relational dynamics ---
    norm_rel = []
    for r in data.get("relational_dynamics", []):
        relation = r.get("relation", "")
        facts = r.get("facts") or r.get("fakta")
        interp = r.get("interpretation") or r.get("tolkning")
        norm_rel.append({
            "relation": relation,
            "facts": facts or "",
            "interpretation": interp or "",
        })
    out["relational_dynamics"] = norm_rel

    # --- Emotional aspects ---
    emo = data.get("emotional_aspects", {}) or {}
    norm_emo = {"direct_expressions": [], "possible_underlying": []}

    for e in emo.get("direct_expressions", []):
        person = e.get("person", "")
        facts = e.get("facts") or e.get("fakta")
        interp = e.get("interpretation") or e.get("tolkning")
        norm_emo["direct_expressions"].append({
            "person": person,
            "facts": facts or "",
            "interpretation": interp or "",
        })

    for e in emo.get("possible_underlying", []):
        facts = e.get("facts") or e.get("fakta")
        interp = e.get("interpretation") or e.get("tolkning")
        norm_emo["possible_underlying"].append({
            "facts": facts or "",
            "interpretation": interp or "",
        })

    out["emotional_aspects"] = norm_emo

    # Themes, tone_and_progression, summary_short are already identical across schemas
    out["themes"] = data.get("themes", [])
    out["tone_and_progression"] = data.get("tone_and_progression", "")
    out["summary_short"] = data.get("summary_short", "")
    out["recommended_ai_persona"] = data.get("recommended_ai_persona") or \
                                    data.get("rekommenderad_ai_persona") or \
                                    "You are a helpful and objective assistant analyzing a conversation transcript."
    return out

def detect_language_from_transcript(text: str) -> str:
    """
    Detect language ('sv', 'en', or 'unknown') from transcript text.
    Uses only a sample of the text for speed.
    """
    sample = (text or "").strip()
    if not sample:
        return "unknown"

    # Use the first ~2000 characters as a sample
    sample = sample[:2000]

    try:
        lang = detect(sample)
    except LangDetectException:
        return "unknown"

    lang = lang.lower()
    if lang.startswith("sv"):
        return "sv"
    if lang.startswith("en"):
        return "en"
    return lang  # could be 'da', 'no', etc.

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analyze a conversation transcript and output JSON analysis."
    )
    parser.add_argument(
        "--analysis-language",
        default="auto",
        choices=["sv", "en", "auto"],
        help="Language for analysis instructions/output (sv, en, or auto to detect from transcript)."
    )
    parser.add_argument(
        "--backend",
        default="gemini",
        choices=["gemini", "chatgpt"],
        help="LLM backend to use."
    )

    parser.add_argument("transcript_filename", help="Path to transcript .txt file")
    args = parser.parse_args()

    if not args.transcript_filename:
        print("[!] Error: path to transcript required.")
        sys.exit(1)

    transcript_path = args.transcript_filename
    if not os.path.exists(transcript_path):
        print(f"[!] Error: file not found: {transcript_path}")
        sys.exit(1)

    base_name, _ = os.path.splitext(transcript_path)
    out_filename = f"{base_name}_analyzed.json"

    print("\n--- Conversation Analysis (Standalone) ---")
    print(f"[i] Loading transcript from: {transcript_path}")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    print("[i] Running preclassification + model selection...")
    if args.backend == "gemini":
        result = analyze_conversation(transcript, call_llm_gemini_25_pro, analysis_language=args.analysis_language, backend="gemini-2.5-pro")
    else:
        result = analyze_conversation(transcript, call_llm_openai, analysis_language=args.analysis_language, backend="gpt-5.1")
    raw_analysis = result.analysis_json
    normalized_analysis = normalize_analysis_schema(raw_analysis)

    print(f"[i] Preclass-reason: {result.preclass_reason}")
    print(f"[i] Writing JSON analysis to: {out_filename}")

    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chosen_model": result.chosen_model,
                "preclass_reason": result.preclass_reason,
                "analysis_raw": result.analysis_raw_text,  # string version
                "analysis_normalized": normalized_analysis  # canonical dict
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[✓] Analys klar. Sparad till: {out_filename}")

