import datetime
from model.in_memmory_db import BUILDING_STATE, persist


def apply_action(user_id: str, intent: str, action: dict):
    if action["action_type"] == "create_borrow":
        item_id = action["item_id"]
        lender_id = action["lender_id"]
        start = action["suggested_start"]
        due = action["suggested_due"]

        borrowing_id = f"borrowing-{len(BUILDING_STATE['borrowings'])+1}"

        BUILDING_STATE["borrowings"].append({
            "id": borrowing_id,
            "item_id": item_id,
            "lender_id": lender_id,
            "borrower_id": user_id,
            "start": start,
            "due": due,
            "status": "active",
        })

        # mark item as borrowed
        for item in BUILDING_STATE["items"]:
            if item["id"] == item_id:
                item["status"] = "borrowed"

        # update impact (simple constants)
        BUILDING_STATE["impact"]["borrows_count"] += 1
        BUILDING_STATE["impact"]["co2_saved_kg"] += 2.0
        BUILDING_STATE["impact"]["waste_avoided_kg"] += 1.0

    elif action["action_type"] == "register_disposal_intent":
        categories = action.get("categories", [])
        # TODO: store intents, and maybe create event if enough neighbors – your logic
        # For now we can just bump events or log it.

    persist()