import json
import os
from typing import Any

from fastapi import HTTPException
from openai import OpenAI

from model.in_memmory_db import BUILDING_STATE


# Simple in-memory per-user chat history
CHAT_HISTORY: dict[str, list[dict]] = {}  # { user_id: [ {"role": "user"/"assistant", "content": "..."}, ... ] }
MAX_HISTORY_TURNS = 6  # keep last N messages (not pairs) to include in the prompt


# Don't create the OpenAI client at import time because that will read
# OPENAI_API_KEY and crash imports if the env var isn't set. Create lazily.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Raise an HTTPException only when the AI is actually used; keep imports safe.
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in environment")
    _client = OpenAI(api_key=api_key)
    return _client


SYSTEM_PROMPT = """
You are LocalLoop, an AI helper for a single apartment building. Your job is to help neighbors borrow, lend, share, or responsibly get rid of items, and to answer simple questions about borrowings, events, and building impact.

STRICT OUTPUT RULES
- ALWAYS return exactly one valid JSON object and nothing else.
- The JSON object MUST contain these keys:
  - intent (string): one of the intents listed below
  - reply (string): a short, helpful human reply in the user's language (English or simple Slovak)
  - action (object|null): an action to apply to the backend state or null when no change is needed
  - confidence (number, optional): 0.0-1.0 confidence score

INTENTS (choose the single best intent)
- borrow_item — user asks to borrow something (examples: "Can I borrow a drill tomorrow?", "potrebujem požičať vŕtačku zajtra").
- return_item — user reports they returned an item ("I returned the drill", "vrátil som vŕtačku").
- register_item / offer_item — user offers an item to neighbors ("I have a loudspeaker I can lend", "mám repro ktoré môžem požičať").
- register_disposal_intent / get_rid_of_items — user wants to get rid of items/donate ("I want to get rid of old clothes", "chcem sa zbaviť kníh").
- ask_item_availability — user asks if an item exists or who owns it ("Who has a ladder?", "Is there a drill available?").
- ask_my_borrowings — "What have I borrowed?"
- ask_borrowed_from_me — "What did people borrow from me?"
- ask_events — "Any events coming up?"
- ask_impact — "How much CO2 and waste did we save so far?"
- small_talk — greetings, thanks, or unrelated chit-chat

ACTION OBJECTS
- action must be an object: {"action_type": string, "metadata": object|null}
- Allowed action_type values and metadata shapes:
  - create_borrow: {"item_id": string, "lender_id": string, "suggested_start": ISO datetime, "suggested_due": ISO datetime}
  - mark_returned: {"borrowing_id": string}
  - register_disposal_intent: either {
        "items": [ {"name": string, "description"?: string, "tags"?: [string,...], "owner_id"?: string } , ... ]
    }
    or the simpler fallback {"categories": [string,...]} (use items when details available)
  - register_item: {"name": string, "description"?: string, "tags"?: [string,...], "owner_id"?: string, "status"?: "available"|"borrowed"}
  - noop: metadata null or {}

WHEN TO RETURN ACTIONS
- Disposal flow (important change): When the user says they want to get rid of things, the assistant MUST follow this flow:
  1) If the user included full details for specific items (name and optional description/tags/owner), the assistant MAY return intent "register_disposal_intent" with metadata.items filled (an array of item objects). But BEFORE returning an action that changes state, the assistant MUST include in `reply` a clear structured summary for each item with fields:
      - Name: <name>
      - Description: <desc>
      - Tags: [tag1, tag2]
      - Owner: <owner_id or inferred>
      - Status to store: for_disposal
     and then ask the user to confirm ("Confirm? yes/no"). Set action to null if confirmation not explicit.
  2) If the user DID NOT provide details (e.g., they said "I want to get rid of an old sofa"), the assistant MUST ask a short clarifying question requesting the minimal details needed to create an item record: name (if ambiguous), short description, tags/categories, and whether they are the owner. Example question: "Do you want to list this as 'old sofa' to donate? Can you give a short description and confirm you're the owner?"
  3) When the user replies with the requested details or confirms, then return intent "register_disposal_intent" and action with metadata.items filled (array) containing the provided details. The backend will then create item-shaped disposal entries.
- For register_item (lending) keep the same confirm/summary behavior as before.
- For borrow_item/return_item keep existing actions and clarifying behavior.

USING BUILDING STATE
- Use the building_state provided (items, borrowings, events, disposal_intents, impact, user_id) to decide availability, owners, and to craft replies.
- When producing summaries, include the fields the backend will store (name, description, tags, owner_id, status) in a concise bulleted list.
- Prefer short, actionable replies. Use user's language (English or simple Slovak) and mirror phrasing when appropriate.

OUTPUT EXAMPLE (disposal summary)
{"intent":"register_disposal_intent","reply":"Summary for disposal:\n- Name: Old sofa\n- Description: 3-seater, fabric, good condition\n- Tags: [furniture,sofa]\n- Owner: peter\nStatus to store: for_disposal\nConfirm?","action":null,"confidence":0.85}
"""


def _safe_json_from_response(resp_obj: Any) -> Any:
    """Try several common shapes from the Responses API and return a parsed JSON object.

    This function explicitly handles the case where the SDK returns an object whose
    `output[0].content[0].text` is a JSON string (the exact shape you pasted).
    It also handles double-encoded JSON strings and a few fallback shapes.
    """
    try:
        # helper: attempt safe json decode and double-decode
        def _try_json_decode(s: str):
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            # If parsing produced a string (double-encoded JSON), try again
            if isinstance(parsed, str):
                try:
                    parsed2 = json.loads(parsed)
                    if isinstance(parsed2, (dict, list)):
                        return parsed2
                except Exception:
                    return None
            if isinstance(parsed, (dict, list)):
                return parsed
            return None

        # 1) Direct SDK object shape: resp.output[0].content[0].text
        try:
            outputs = getattr(resp_obj, "output", None)
            if outputs and len(outputs) > 0:
                first_out = outputs[0]
                content = getattr(first_out, "content", None) or (first_out.get("content") if isinstance(first_out, dict) else None)
                if content and len(content) > 0:
                    first_block = content[0]
                    # prefer block.text if present
                    block_text = None
                    if hasattr(first_block, "text"):
                        block_text = getattr(first_block, "text")
                    elif isinstance(first_block, dict) and "text" in first_block:
                        block_text = first_block.get("text")
                    if isinstance(block_text, str):
                        decoded = _try_json_decode(block_text.strip())
                        if decoded is not None:
                            return decoded
                    # try block.value if SDK already produced parsed value
                    block_value = getattr(first_block, "value", None) if hasattr(first_block, "value") else (first_block.get("value") if isinstance(first_block, dict) else None)
                    if isinstance(block_value, (dict, list)):
                        return block_value
        except Exception:
            # fallthrough to other methods
            pass

        # 2) handle dict-like resp_obj._obj produced by some SDK internals
        data = getattr(resp_obj, "_obj", None) or (resp_obj if isinstance(resp_obj, dict) else None)
        if isinstance(data, dict) and "output" in data:
            for out in data.get("output", []):
                content = out.get("content")
                if not content:
                    continue
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block and isinstance(block["text"], str):
                            decoded = _try_json_decode(block["text"].strip())
                            if decoded is not None:
                                return decoded
                        if "value" in block and isinstance(block["value"], (dict, list)):
                            return block["value"]

        # 3) Fallback: try to extract JSON substring from string representation
        text = str(resp_obj)
        # Find a balanced JSON substring starting at first '{' or '[' using a simple stack
        def _find_balanced(s: str, open_ch: str, close_ch: str, start_idx: int) -> int:
            depth = 0
            for i in range(start_idx, len(s)):
                ch = s[i]
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return i
            return -1

        first_curly = text.find('{')
        first_brack = text.find('[')
        candidates = []
        if first_curly != -1:
            end = _find_balanced(text, '{', '}', first_curly)
            if end != -1:
                candidates.append(text[first_curly:end+1])
        if first_brack != -1:
            end = _find_balanced(text, '[', ']', first_brack)
            if end != -1:
                candidates.append(text[first_brack:end+1])

        for candidate in candidates:
            parsed = _try_json_decode(candidate)
            if parsed is not None:
                return parsed

        # If nothing worked, raise so caller can return a helpful error
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON; raw: {resp_obj}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {e}; raw: {resp_obj}")


def run_localloop_brain(user_id: str, message: str):
    # Only send minimal building info needed (items, borrowings, events)
    building_context = {
        "user_id": user_id,
        "items": BUILDING_STATE.get("items", []),
        "borrowings": BUILDING_STATE.get("borrowings", []),
        "events": BUILDING_STATE.get("events", []),
        "impact": BUILDING_STATE.get("impact", {}),
        "disposal_intents": BUILDING_STATE.get("disposal_intents", []),
    }

    # Do not append user's message to history yet (we'll append after a successful AI response)

    # Instantiate client lazily (and fail with a clear error if OPENAI_API_KEY missing)
    client = _get_client()

    # Build the responses API call payload. We MUST use `input=[...]` and response_format json_object
    try:
        # Build a single combined text input: system prompt + building state + recent history + current user message
        history_lines = []
        for m in CHAT_HISTORY.get(user_id, []):
            role = m.get("role", "user")
            text = m.get("content", "")
            if not text:
                continue
            history_lines.append(f"{role.capitalize()}: {text}")

        combined_parts = [SYSTEM_PROMPT, f"Building state: {json.dumps(building_context)}"]
        user_lang = _detect_user_language(message, CHAT_HISTORY.get(user_id, []))
        # map language codes to friendly names the model understands
        lang_names = {
            "en": "English",
            "sk": "Slovak",
            "cs": "Czech",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "pl": "Polish",
            "ru": "Russian",
        }
        lang_label = lang_names.get(user_lang, "English")

        combined_parts = [
            SYSTEM_PROMPT,
            f"Building state: {json.dumps(building_context)}",
            f"Instruction: Please reply in {lang_label}. Use that language for the 'reply' field and mirror user's language when possible.",
        ]
        if history_lines:
            combined_parts.append("Conversation history:")
            combined_parts.append("\n".join(history_lines))
        combined_parts.append(f"User: {message}")
        combined_input = "\n\n".join(combined_parts)

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=combined_input,
            temperature=0.5,
            max_output_tokens=800,
            # response_format={"type": "json_object"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI request failed: {e}")

    # Parse response safely
    parsed = _safe_json_from_response(resp)

    # Validate shape
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=500, detail=f"AI returned non-object JSON: {parsed}")

    # Ensure required keys
    intent = parsed.get("intent")
    reply = parsed.get("reply")
    action = parsed.get("action")
    confidence = parsed.get("confidence")

    if intent is None or reply is None:
        raise HTTPException(status_code=500, detail=f"AI response missing keys: {parsed}")

    # Normalize action shape: allow both {action_type, metadata} or earlier shape
    if action and not isinstance(action, dict):
        raise HTTPException(status_code=500, detail=f"AI action must be an object or null: {action}")

    # Append the user's message and assistant reply to history, then trim
    CHAT_HISTORY.setdefault(user_id, []).append({"role": "user", "content": message})
    CHAT_HISTORY.setdefault(user_id, []).append({"role": "assistant", "content": reply})
    if len(CHAT_HISTORY[user_id]) > MAX_HISTORY_TURNS:
        CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_HISTORY_TURNS:]

    # Provide a default confidence if not present
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None

    # Return structured result
    return {"intent": intent, "reply": reply, "action": action, "confidence": confidence}


def _detect_user_language(message: str, history: list[dict]) -> str:
    # Merge message + recent history for context
    text = ((message or "") + " " + " ".join((h.get("content", "") or "") for h in (history or []))).strip()
    if not text:
        return "en"

    # Try to use the `langdetect` library if available (more robust)
    try:
        from langdetect import detect, DetectorFactory

        # make detection deterministic
        DetectorFactory.seed = 0
        lang_code = detect(text)
        # normalise a few variants and prefer two-letter codes we support
        lang_code = (lang_code or "").lower()
        simple_map = {
            "cs": "cs", "sk": "sk", "es": "es", "fr": "fr", "de": "de", "it": "it",
            "pt": "pt", "pl": "pl", "ru": "ru", "en": "en"
        }
        if lang_code in simple_map:
            return simple_map[lang_code]
        # langdetect may return variants like 'zh-cn' etc. fall back to prefix
        if len(lang_code) >= 2:
            prefix = lang_code.split("-")[0]
            return simple_map.get(prefix, prefix)
    except Exception:
        # fall back to lightweight heuristic below
        pass

    # lightweight heuristic fallback (token + script checks)
    lower = text.lower()
    # quick Cyrillic check
    for ch in lower:
        if '\u0400' <= ch <= '\u04FF':
            return "ru"

    lang_tokens = {
        "sk": ["potreb", "mám", "mam", "ďak", "dak", "prosím", "prosim", "požičať", "vrátil", "vratil"],
        "cs": ["potřeb", "mám", "mě", "děku", "prosím", "díky"],
        "es": ["quiero", "por favor", "gracias", "hola", "tengo"],
        "fr": ["bonjour", "merci", "s'il vous plaît", "salut", "j'ai"],
        "de": ["hallo", "danke", "bitte", "möchte", "kannst"],
        # "it": ["ciao", "grazie", "per favore", "ho"],
        "pt": ["obrigado", "por favor", "preciso", "tenho"],
        "pl": ["cześć", "dziękuję", "proszę", "chcę"],
        "en": ["please", "thanks", "hello", "i have", "i want", "can i"],
    }

    for code, tokens in lang_tokens.items():
        for tok in tokens:
            if tok in lower:
                return code

    # fallback: if there are many non-ascii characters, assume non-English; else English
    non_ascii = sum(1 for ch in lower if ord(ch) > 127)
    if non_ascii > 3:
        return "en"
    return "en"
