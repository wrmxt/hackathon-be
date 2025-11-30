from ai.ai import run_localloop_brain, CHAT_HISTORY, CHAT_PREFERRED_LANG
from domain.actions import apply_action, confirm_borrowing
from main import app
from model.in_memmory_db import BUILDING_STATE, persist
from typing import Optional
from model.models import ReturnBorrowingRequest, ChatRequest, ConfirmBorrowingRequest, UpdateItemRequest, RequestBorrowingRequest

from fastapi import HTTPException


@app.get("/api/building-state")
def get_building_state():
    return {
        "building": BUILDING_STATE["building"],
        "residents": BUILDING_STATE["residents"],
        "items": BUILDING_STATE["items"],
        "impact": BUILDING_STATE["impact"],
        "disposal_intents": BUILDING_STATE["disposal_intents"],
    }


@app.get("/api/borrowings")
def get_borrowings(user_id: str):
    borrowed = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("borrower_id") == user_id]
    lent = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("lender_id") == user_id]
    return {"borrowed": borrowed, "lent": lent}


@app.get("/api/borrowings/pending")
def get_pending_borrowings(user_id: str):

    pending = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("status") == "waiting_for_confirm"]
    as_lender = [b for b in pending if b.get("lender_id") == user_id]
    as_borrower = [b for b in pending if b.get("borrower_id") == user_id]
    return {"as_lender": as_lender, "as_borrower": as_borrower}


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

    action_result = None
    final_message = None
    # Helper: format a concise final confirmation message for the user (localized)
    def _format_final_message(result: dict, user_id: str, lang: str) -> str:
        # Support a few common message types; fallback to a simple English summary.
        r = result or {}
        res = r.get("result")
        if res == "borrow_waiting_confirmation":
            bid = r.get("borrowing_id")
            # lookup borrowing and item for extra detail
            borrowing = next((b for b in BUILDING_STATE.get("borrowings", []) if b.get("id") == bid), None)
            item = None
            owner_name = None
            if borrowing:
                item = next((it for it in BUILDING_STATE.get("items", []) if it.get("id") == borrowing.get("item_id")), None)
                owner = next((p for p in BUILDING_STATE.get("residents", []) if p.get("id") == borrowing.get("lender_id")), None)
                owner_name = owner.get("name") if owner else borrowing.get("lender_id")
            name = item.get("name") if item else "item"
            if lang == "sk":
                return f"Požiadavka bola vytvorená a poslaná vlastníkovi ({owner_name}) na potvrdenie. Položka: {name}. ID: {bid}."
            return f"Your borrowing request was created and sent to {owner_name} for confirmation. Item: {name}. Request ID: {bid}."

        if res == "item_registered":
            item_id = r.get("item_id")
            item = next((it for it in BUILDING_STATE.get("items", []) if it.get("id") == item_id), None)
            name = item.get("name") if item else item_id
            if lang == "sk":
                return f"Položka '{name}' bola uložená v zozname (id={item_id})."
            return f"Item '{name}' has been registered (id={item_id})."

        if res == "disposal_registered":
            created = r.get("created_items") or []
            names = ", ".join([it.get("name") for it in created]) if created else None
            if lang == "sk":
                return f"Zaznamenané predmety na zber: {names or '(nie je uvedené)'}."
            return f"Disposal intents registered for: {names or '(none)'}"

        if res == "marked_returned":
            bid = r.get("borrowing_id")
            if lang == "sk":
                return f"Vrátenie bolo zaregistrované (id: {bid}). Vďaka!"
            return f"Return recorded (id: {bid}). Thank you!"

        # error or unknown
        err = r.get("error") or r.get("message")
        if err:
            if lang == "sk":
                return f"Akcia nebola vykonaná: {err}"
            return f"Action not completed: {err}"
        # generic fallback
        if lang == "sk":
            return "Akcia bola vykonaná."
        return "Action completed."

    # Apply action to BUILDING_STATE if present
    if action:
        # Optionally, one could check confidence here; for now we apply all actions returned by the model.
        try:
            action_result = apply_action(req.user_id, intent, action)
            # If the action changed state (backend returned a truthy result), prepare final confirmation message
            if action_result:
                user_lang = CHAT_PREFERRED_LANG.get(req.user_id, "en")
                final_message = _format_final_message(action_result, req.user_id, user_lang)
                # Replace the user's conversation context with a single assistant final message.
                # This keeps the final confirmation visible in history but removes prior context.
                try:
                    CHAT_HISTORY.pop(req.user_id, None)
                except KeyError:
                    pass
                CHAT_HISTORY[req.user_id] = [{"role": "assistant", "content": final_message}]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to apply action: {e}")
    # If no action was returned, simply reply (AI may ask follow-up questions); do not auto-register items.

    return {"reply": reply, "intent": intent, "confidence": confidence, "action_result": action_result, "final_message": final_message}


@app.get("/api/items")
def get_items(user_id: Optional[str] = None, exclude_owner: bool = False):
    """Return items with optional owner metadata.

    Query params:
    - user_id: optional; if provided each item will have `is_owner` and `owner_name` fields.
    - exclude_owner: boolean; if true and user_id provided, items owned by the user are filtered out.
    """
    items = BUILDING_STATE.get("items", [])
    residents = {r.get("id"): r for r in BUILDING_STATE.get("residents", [])}

    enriched = []
    for item in items:
        it = dict(item)  # shallow copy
        owner_id = it.get("owner_id")
        it["is_owner"] = (user_id is not None and owner_id == user_id)
        owner = residents.get(owner_id)
        it["owner_name"] = owner.get("name") if owner else owner_id
        if exclude_owner and it["is_owner"]:
            continue
        enriched.append(it)

    return {"items": enriched}

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
            persist()
            return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Borrowing not found")


@app.post("/api/borrowings/confirm")
def confirm_borrowing_endpoint(req: ConfirmBorrowingRequest):
    try:
        borrowing = confirm_borrowing(req.owner_id, req.borrowing_id)
        return {"status": "ok", "borrowing": borrowing}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# PATCH /api/items/{item_id} - update own item (name, description, tags, status)
# Constraints:
# - Only owner can update.
# - Cannot change status to available if there is an active borrowing.
# - No updates allowed while item is borrowed (status=='borrowed').
@app.patch("/api/items/{item_id}")
def update_item(item_id: str, req: UpdateItemRequest):
    # locate item
    item = next((it for it in BUILDING_STATE.get("items", []) if it.get("id") == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.get("owner_id") != req.user_id:
        raise HTTPException(status_code=403, detail="You can only modify your own items")

    # check active borrowings for this item
    active_borrow = next((b for b in BUILDING_STATE.get("borrowings", []) if b.get("item_id") == item_id and b.get("status") in ("active", "waiting_for_confirm")), None)
    if active_borrow and req.status and req.status == "available":
        # allow? if waiting_for_confirm and owner wants to mark unavailable maybe; disallow switching to available mid-borrow
        raise HTTPException(status_code=400, detail="Cannot set status to available while borrowing is active or pending")
    if item.get("status") == "borrowed":
        raise HTTPException(status_code=400, detail="Cannot modify a borrowed item until it is returned")

    # apply updates (only provided fields)
    if req.name is not None:
        item["name"] = req.name.strip() or item["name"]
    if req.description is not None:
        item["description"] = req.description
    if req.tags is not None:
        if not isinstance(req.tags, list):
            raise HTTPException(status_code=400, detail="Tags must be a list of strings")
        item["tags"] = req.tags
    if req.status is not None:
        if req.status not in ("available", "archived", "unavailable"):
            raise HTTPException(status_code=400, detail="Invalid status")
        item["status"] = req.status

    persist()
    return {"status": "ok", "item": item}

# DELETE /api/items/{item_id}
# Constraints:
# - Only owner can delete.
# - Cannot delete if item is borrowed or has active/pending borrowings.
@app.delete("/api/items/{item_id}")
def delete_item(item_id: str, user_id: str):
    item = next((it for it in BUILDING_STATE.get("items", []) if it.get("id") == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="You can only delete your own items")

    # any borrowing referencing this item with status waiting/active/borrowed blocks deletion
    blocking_statuses = {"waiting_for_confirm", "active", "borrowed"}
    blocking = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("item_id") == item_id and b.get("status") in blocking_statuses]
    if blocking:
        raise HTTPException(status_code=400, detail="Cannot delete item with active or pending borrowings")

    # Remove the item
    BUILDING_STATE["items"] = [it for it in BUILDING_STATE.get("items", []) if it.get("id") != item_id]
    # Remove non-blocking borrowings tied to the item (returned, cancelled, or any other leftover)
    BUILDING_STATE["borrowings"] = [b for b in BUILDING_STATE.get("borrowings", []) if b.get("item_id") != item_id]

    persist()
    return {"status": "ok"}

@app.post("/api/borrowings/request")
def request_borrowing(req: RequestBorrowingRequest):
    """Create a borrow request for an item.

    Rules:
    - Borrower and item must exist.
    - Borrower cannot be the owner.
    - No existing waiting/active borrowing for this item (by anyone) and no duplicate waiting request by same borrower.
    - Auto-generate start/due timestamps (now and +1 day).
    - Set item.status = 'requested'.
    """
    from datetime import datetime, timedelta, timezone

    # Validate borrower exists
    borrower = next((r for r in BUILDING_STATE.get("residents", []) if r.get("id") == req.user_id), None)
    if not borrower:
        raise HTTPException(status_code=404, detail="Borrower not found")

    # Validate item exists
    item = next((it for it in BUILDING_STATE.get("items", []) if it.get("id") == req.item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    lender_id = item.get("owner_id")
    if req.user_id == lender_id:
        raise HTTPException(status_code=400, detail="Cannot borrow your own item")

    # Block if item already has active or pending borrowing (by anyone)
    blocking = next((b for b in BUILDING_STATE.get("borrowings", []) if b.get("item_id") == req.item_id and b.get("status") in ("active", "waiting_for_confirm")), None)
    if blocking:
        raise HTTPException(status_code=400, detail="Item already borrowed or requested")

    # Prevent duplicate waiting request by same borrower (extra safety though blocking covers it)
    dup_same_user = next((b for b in BUILDING_STATE.get("borrowings", []) if b.get("item_id") == req.item_id and b.get("borrower_id") == req.user_id and b.get("status") == "waiting_for_confirm"), None)
    if dup_same_user:
        raise HTTPException(status_code=400, detail="You have already requested this item")

    # Item must be in a requestable state
    if item.get("status") not in ("available", "unavailable"):
        raise HTTPException(status_code=400, detail="Item is not requestable right now")

    # Generate start/due
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start_iso = now.isoformat()
    due_iso = (now + timedelta(days=1)).isoformat()

    # Create borrowing record
    borrowing_id = f"borrowing-{len(BUILDING_STATE.get('borrowings', [])) + 1}"
    borrowing = {
        "id": borrowing_id,
        "item_id": req.item_id,
        "lender_id": lender_id,
        "borrower_id": req.user_id,
        "start": start_iso,
        "due": due_iso,
        "status": "waiting_for_confirm",
    }
    BUILDING_STATE.setdefault("borrowings", []).append(borrowing)

    # Update item status to requested
    item["status"] = "requested"

    persist()
    return {"status": "ok", "borrowing": borrowing}
