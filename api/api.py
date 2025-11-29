from ai.ai import run_localloop_brain
from domain.actions import apply_action
from main import app
from model.in_memmory_db import BUILDING_STATE
from model.models import ReturnBorrowingRequest, ChatRequest
from fastapi import HTTPException


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
    # If no action was returned, simply reply (AI may ask follow-up questions); do not auto-register items.

    return {"reply": reply, "intent": intent, "confidence": confidence}


@app.get("/api/items")
def get_items():
    return {"items": BUILDING_STATE.get("items", [])}

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
