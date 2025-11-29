from ai.ai import run_localloop_brain
from main import app
from model.in_memmory_db import BUILDING_STATE
from model.models import ReturnBorrowingRequest, ChatRequest


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
    # TODO: filter BUILDING_STATE["borrowings"] into borrowed/lent
    return {"borrowed": [], "lent": []}

@app.get("/api/events")
def get_events():
    return {"events": BUILDING_STATE["events"]}

@app.post("/api/chat")
def post_chat(req: ChatRequest):
    brain_result = run_localloop_brain(req.user_id, req.message)
    intent = brain_result.get("intent")
    reply = brain_result.get("reply")
    action = brain_result.get("action")

    # Apply action to BUILDING_STATE
    if action:
        apply_action(req.user_id, intent, action)

    return {"reply": reply}

@app.post("/api/borrowings/return")
def return_borrowing(req: ReturnBorrowingRequest):
    # TODO: mark borrowing as returned
    return {"status": "ok"}
