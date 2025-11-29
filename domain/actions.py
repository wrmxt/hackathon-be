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

        borrowing_id = f"borrowing-{len(BUILDING_STATE.get('borrowings', [])) + 1}"
        BUILDING_STATE.setdefault("borrowings", []).append({
            "id": borrowing_id,
            "item_id": item_id,
            "lender_id": lender_id,
            "borrower_id": user_id,
            "start": start,
            "due": due,
            "status": "active",
        })

        # mark item as borrowed
        item["status"] = "borrowed"

        # update impact (simple constants)
        BUILDING_STATE["impact"]["borrows_count"] += 1
        BUILDING_STATE["impact"]["co2_saved_kg"] += constants.IMPACT["CO2_PER_BORROW_KG"]
        BUILDING_STATE["impact"]["waste_avoided_kg"] += constants.IMPACT["WASTE_PER_BORROW_KG"]

        persist()
        return {"result": "borrow_created", "borrowing_id": borrowing_id}

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
        categories = metadata.get("categories", [])
        if not categories:
            persist()
            return None

        # store intents per user
        intents_store = BUILDING_STATE.setdefault("disposal_intents", [])
        intent_id = f"intent-{len(intents_store) + 1}"
        intents_store.append({
            "id": intent_id,
            "user_id": user_id,
            "categories": categories,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        # Check if any category reached threshold to create an event
        created_events = []
        for cat in categories:
            # count intents mentioning this category
            count = sum(1 for it in intents_store if cat in it.get("categories", []))
            if count >= constants.DISPOSAL_INTENT_THRESHOLD:
                # create a swap/collection event for this category
                estimated_items = constants.ESTIMATED_ITEMS_PER_INTENT * count
                co2 = constants.IMPACT["CO2_PER_EVENT_ITEM_KG"] * estimated_items
                waste = constants.IMPACT["WASTE_PER_EVENT_ITEM_KG"] * estimated_items
                metadata_event = {"category": cat, "intents_count": count}
                ev = _create_event("collection", "disposal_intent", metadata_event, co2, waste)
                created_events.append(ev)
                # remove intents for that category to avoid duplicate events
                # (simple approach)
                BUILDING_STATE["disposal_intents"] = [it for it in intents_store if cat not in it.get("categories", [])]
                intents_store = BUILDING_STATE["disposal_intents"]

        persist()
        return {"result": "disposal_registered", "created_events": created_events}

    # NOOP or unknown
    persist()
    return None
