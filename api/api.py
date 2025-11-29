from ai.ai import run_localloop_brain
from domain.actions import apply_action
from main import app
from model.in_memmory_db import BUILDING_STATE
from model.models import ReturnBorrowingRequest, ChatRequest
from fastapi import HTTPException
import re


def _extract_offered_item_name(message: str) -> str | None:
    if not message:
        return None
    m = message.lower()
    # common offer markers
    patterns = [r"i have\s+([a-z0-9\s'-]+)", r"i've got\s+([a-z0-9\s'-]+)", r"i can lend\s+([a-z0-9\s'-]+)", r"i can loan\s+([a-z0-9\s'-]+)", r"mám\s+([a-z0-9\s'-]+)"]
    for p in patterns:
        match = re.search(p, m)
        if match:
            name = match.group(1).strip()
            # cut off trailing clauses like 'to someone' or 'if needed'
            name = re.split(r"\s+(to|for|if|i|we)\b", name)[0].strip()
            # basic cleanup
            name = re.sub(r"[^a-z0-9\s'-]", '', name)
            if len(name) >= 2:
                return name
    return None


@app.get("/api/building-state")
def get_building_state():
    return {
        "building": BUILDING_STATE["building"],
        "residents": BUILDING_STATE["residents"],
        "items": BUILDING_STATE["items"],
        "impact": BUILDING_STATE["impact"],
    }

@app.get("/api/borrowings")
def get_borrowings(user_id: str):
    borrowed = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("borrower_id") == user_id]
    lent = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("lender_id") == user_id]
    return {"borrowed": borrowed, "lent": lent}

@app.get("/api/events")
def get_events():
    return {"events": BUILDING_STATE.get("events", [])}

@app.post("/api/chat")
def post_chat(req: ChatRequest):
    brain_result = run_localloop_brain(req.user_id, req.message)
    intent = brain_result.get("intent")
    reply = brain_result.get("reply")
    action = brain_result.get("action")
    confidence = brain_result.get("confidence")

    # Apply action to BUILDING_STATE if present
    if action:
        # Optionally, one could check confidence here; for now we apply all actions returned by the model.
        try:
            apply_action(req.user_id, intent, action)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to apply action: {e}")
    else:
        # If model didn't return an action, try a lightweight heuristic to detect offers like "I have a drill"
        offered_name = _extract_offered_item_name(req.message)
        # Auto-register offered items whenever we detect an offered name (even if model said small_talk)
        if offered_name:
             # register the item automatically
             register_action = {
                 "action_type": "register_item",
                 "metadata": {"name": offered_name, "description": "Offered via chat", "owner_id": req.user_id}
             }
             try:
                 res = apply_action(req.user_id, intent, register_action)
                 if res and res.get("result") == "item_registered":
                     item_id = res.get("item_id")
                     reply = (reply or "Thanks!") + f" I've registered '{offered_name}' as available (id: {item_id})."
             except Exception:
                 # don't fail the whole request if registration fails
                 pass

    return {"reply": reply, "intent": intent, "confidence": confidence}

@app.post("/api/borrowings/return")
def return_borrowing(req: ReturnBorrowingRequest):
    # Mark borrowing as returned
    borrowing_id = req.borrowing_id
    # find borrowing
    for b in BUILDING_STATE.get("borrowings", []):
        if b.get("id") == borrowing_id:
            b["status"] = "returned"
            # make item available again
            for item in BUILDING_STATE.get("items", []):
                if item.get("id") == b.get("item_id"):
                    item["status"] = "available"
            from model.in_memmory_db import persist
            persist()
            return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Borrowing not found")
