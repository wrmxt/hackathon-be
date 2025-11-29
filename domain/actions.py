import datetime
from typing import Optional

from model.in_memmory_db import BUILDING_STATE, persist
import constants


def _find_item(item_id: str):
    for item in BUILDING_STATE.get("items", []):
        if item.get("id") == item_id:
            return item
    return None


def _find_borrowing(borrowing_id: str):
    for b in BUILDING_STATE.get("borrowings", []):
        if b.get("id") == borrowing_id:
            return b
    return None


def _create_event(event_type: str, source: str, metadata: dict, impact_co2: float, impact_waste: float):
    eid = f"event-{len(BUILDING_STATE.get('events', [])) + 1}"
    event = {
        "id": eid,
        "event_type": event_type,
        "source": source,
        "metadata": metadata,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "impact": {"co2_saved_kg": impact_co2, "waste_avoided_kg": impact_waste},
    }
    BUILDING_STATE.setdefault("events", []).append(event)
    BUILDING_STATE["impact"]["co2_saved_kg"] += impact_co2
    BUILDING_STATE["impact"]["waste_avoided_kg"] += impact_waste
    BUILDING_STATE["impact"]["events_count"] += 1
    return event


def apply_action(user_id: str, intent: str, action: Optional[dict]):
    # action is expected like {"action_type":"create_borrow","metadata":{...}}
    if not action:
        persist()
        return None

    action_type = action.get("action_type")
    metadata = action.get("metadata") or {}

    # CREATE BORROW
    if action_type == "create_borrow":
        item_id = metadata.get("item_id")
        lender_id = metadata.get("lender_id")
        start = metadata.get("suggested_start")
        due = metadata.get("suggested_due")

        if not item_id or not lender_id or not start or not due:
            # invalid action, ignore
            persist()
            return None

        item = _find_item(item_id)
        if not item or item.get("status") != "available":
            # can't borrow
            persist()
            return None

        # Create borrowing in a pending state - owner must confirm
        borrowing_id = f"borrowing-{len(BUILDING_STATE.get('borrowings', [])) + 1}"
        BUILDING_STATE.setdefault("borrowings", []).append({
            "id": borrowing_id,
            "item_id": item_id,
            "lender_id": lender_id,
            "borrower_id": user_id,
            "start": start,
            "due": due,
            "status": "waiting_for_confirm",
        })

        persist()
        return {"result": "borrow_waiting_confirmation", "borrowing_id": borrowing_id}

    # MARK RETURNED
    if action_type == "mark_returned":
        borrowing_id = metadata.get("borrowing_id")
        if not borrowing_id:
            persist()
            return None
        borrowing = _find_borrowing(borrowing_id)
        if not borrowing:
            persist()
            return None
        if borrowing.get("status") == "returned":
            persist()
            return {"result": "already_returned", "borrowing_id": borrowing_id}

        borrowing["status"] = "returned"
        # mark item available again
        item = _find_item(borrowing.get("item_id"))
        if item:
            item["status"] = "available"

        persist()
        return {"result": "marked_returned", "borrowing_id": borrowing_id}

    # REGISTER DISPOSAL INTENT
    if action_type == "register_disposal_intent":
        items = metadata.get("items")
        categories = metadata.get("categories", [])

        if not items and not categories:
            persist()
            return None

        disposal_store = BUILDING_STATE.setdefault("disposal_intents", [])
        created_items = []
        created_events = []

        # If items provided, create disposal entries from them
        if items and isinstance(items, list):
            for it in items:
                name = it.get("name") or "unnamed"
                description = it.get("description") or ""
                tags = it.get("tags") or []
                owner = it.get("owner_id") or user_id
                disp_id = f"disposal-{len(disposal_store) + 1}"
                item_obj = {
                    "id": disp_id,
                    "name": name,
                    "description": description,
                    "tags": tags,
                    "owner_id": owner,
                    "status": "for_disposal",
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                disposal_store.append(item_obj)
                created_items.append(item_obj)

        # Fallback: if categories provided, create simple disposal entries per category
        if categories and isinstance(categories, list) and len(categories) > 0:
            for cat in categories:
                disp_id = f"disposal-{len(disposal_store) + 1}"
                item_obj = {
                    "id": disp_id,
                    "name": cat if isinstance(cat, str) else str(cat),
                    "description": "",
                    "tags": [cat] if isinstance(cat, str) else [],
                    "owner_id": user_id,
                    "status": "for_disposal",
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                disposal_store.append(item_obj)
                created_items.append(item_obj)

        # Check threshold per tag to create events
        # Build unique tags from created_items
        tags_to_check = set(tag for it in created_items for tag in (it.get("tags") or []))
        for tag in tags_to_check:
            count = sum(1 for it in disposal_store if tag in (it.get("tags") or []))
            if count >= constants.DISPOSAL_INTENT_THRESHOLD:
                estimated_items = constants.ESTIMATED_ITEMS_PER_INTENT * count
                co2 = constants.IMPACT["CO2_PER_EVENT_ITEM_KG"] * estimated_items
                waste = constants.IMPACT["WASTE_PER_EVENT_ITEM_KG"] * estimated_items
                metadata_event = {"category": tag, "intents_count": count}
                ev = _create_event("collection", "disposal_intent", metadata_event, co2, waste)
                created_events.append(ev)
                # remove disposal intents for that tag
                BUILDING_STATE["disposal_intents"] = [it for it in disposal_store if tag not in (it.get("tags") or [])]
                disposal_store = BUILDING_STATE["disposal_intents"]

        persist()
        return {"result": "disposal_registered", "created_items": created_items, "created_events": created_events}

    # REGISTER ITEM (new)
    if action_type == "register_item":
        # metadata: name (required), description (opt), tags (opt list), status (opt)
        name = metadata.get("name") or metadata.get("title")
        description = metadata.get("description") or metadata.get("desc") or ""
        tags = metadata.get("tags") or metadata.get("categories") or []
        status = metadata.get("status") or "available"
        owner_id = metadata.get("owner_id") or user_id

        if not name:
            persist()
            return None

        item_id = f"item-{len(BUILDING_STATE.get('items', [])) + 1}"
        item = {
            "id": item_id,
            "name": name,
            "description": description,
            "tags": tags,
            "owner_id": owner_id,
            "status": status,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        BUILDING_STATE.setdefault("items", []).append(item)
        # update simple impact counters (optional)
        BUILDING_STATE["impact"]["items_shared"] = BUILDING_STATE["impact"].get("items_shared", 0) + 1

        persist()
        return {"result": "item_registered", "item_id": item_id}


def confirm_borrowing(owner_id: str, borrowing_id: str):
    """Confirm a pending borrowing. Only the lender (owner) can confirm.

    This marks borrowing as 'active', sets item.status to 'borrowed', and updates impact.
    Returns the updated borrowing dict on success, otherwise raises ValueError.
    """
    borrowing = _find_borrowing(borrowing_id)
    if not borrowing:
        raise ValueError("Borrowing not found")
    if borrowing.get("lender_id") != owner_id:
        raise ValueError("Only the lender/owner can confirm this borrowing")
    if borrowing.get("status") != "waiting_for_confirm":
        raise ValueError("Borrowing is not waiting for confirmation")

    # mark as active
    borrowing["status"] = "active"
    # mark item as borrowed
    item = _find_item(borrowing.get("item_id"))
    if item:
        item["status"] = "borrowed"

    # update impact
    BUILDING_STATE["impact"]["borrows_count"] += 1
    BUILDING_STATE["impact"]["co2_saved_kg"] += constants.IMPACT["CO2_PER_BORROW_KG"]
    BUILDING_STATE["impact"]["waste_avoided_kg"] += constants.IMPACT["WASTE_PER_BORROW_KG"]

    persist()
    return borrowing
