from pathlib import Path
import json
from typing import Dict, List, Set

# Public API (must keep names):
# - BUILDING_STATE: dict
# - persist(): function

# Storage file path (as requested)
DATA_FILE = Path("./model/db.json")

# Default structure
_DEFAULT_STATE: Dict = {
    "building": {},
    "residents": [],
    "items": [],
    "borrowings": [],
    "events": [],
    "impact": {},
    "disposal_intents": [],
}

# Allowed status sets
_BORROW_STATUSES: Set[str] = {
    "waiting_for_confirm",
    "active",
    "returned",
    "cancelled",
    "return_requested",
}
_ITEM_STATUSES: Set[str] = {"available", "unavailable", "archived", "borrowed", "requested"}


def _read_raw_state() -> Dict:
    if not DATA_FILE.exists():
        return json.loads(json.dumps(_DEFAULT_STATE))
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # corrupted file -> start with default
        return json.loads(json.dumps(_DEFAULT_STATE))
    # ensure top-level keys exist
    for k in _DEFAULT_STATE.keys():
        if k not in data or data[k] is None:
            data[k] = json.loads(json.dumps(_DEFAULT_STATE[k]))
    return data


def _cleanup_state(state: Dict) -> Dict:
    # Normalize residents and collect ids
    residents: List[Dict] = state.get("residents", []) or []
    resident_ids: Set[str] = {r.get("id") for r in residents if r.get("id")}

    # Filter items: must have id, owner_id present in residents
    raw_items: List[Dict] = state.get("items", []) or []
    items: List[Dict] = []
    for it in raw_items:
        iid = it.get("id")
        owner = it.get("owner_id")
        if not iid:
            continue
        if not owner or owner not in resident_ids:
            continue
        # normalize item status
        status = it.get("status")
        if status not in _ITEM_STATUSES:
            it["status"] = "available"
        items.append(it)

    # Collect item ids set
    item_ids: Set[str] = {it.get("id") for it in items}

    # Filter borrowings: must have id, valid item_id, valid borrower/lender, status in allowed
    raw_borrows: List[Dict] = state.get("borrowings", []) or []
    borrows: List[Dict] = []
    for b in raw_borrows:
        bid = b.get("id")
        item_id = b.get("item_id")
        lender_id = b.get("lender_id")
        borrower_id = b.get("borrower_id")
        status = b.get("status")
        if not bid:
            continue
        if not item_id or item_id not in item_ids:
            continue
        if not lender_id or lender_id not in resident_ids:
            continue
        if not borrower_id or borrower_id not in resident_ids:
            continue
        if status not in _BORROW_STATUSES:
            continue
        borrows.append(b)

    # Map item_id -> list of borrow statuses
    borrow_by_item: Dict[str, List[str]] = {}
    for b in borrows:
        iid = b.get("item_id")
        st = b.get("status")
        borrow_by_item.setdefault(iid, []).append(st)

    # Recompute item status consistency
    for it in items:
        iid = it.get("id")
        statuses = borrow_by_item.get(iid, [])
        if any(st == "active" for st in statuses):
            it["status"] = "borrowed"
        elif any(st == "waiting_for_confirm" for st in statuses):
            # Mark as requested while pending confirmation
            it["status"] = "requested"
        elif any(st == "return_requested" for st in statuses):
            # Return requested but item still with borrower; keep as borrowed until owner confirms
            it["status"] = "borrowed"
        else:
            # if wrongly marked borrowed/requested -> make available
            if it.get("status") in ("borrowed", "requested"):
                it["status"] = "available"
            elif it.get("status") not in _ITEM_STATUSES:
                it["status"] = "available"

    # Ensure top-level structure
    cleaned = {
        "building": state.get("building", {}) or {},
        "residents": residents,
        "items": items,
        "borrowings": borrows,
        "events": state.get("events", []) or [],
        "impact": state.get("impact", {}) or {},
        "disposal_intents": state.get("disposal_intents", []) or [],
    }
    return cleaned


# Load, clean, and expose BUILDING_STATE
BUILDING_STATE: Dict = _cleanup_state(_read_raw_state())


def persist():
    """Clean current BUILDING_STATE and write to disk as UTF-8 JSON"""
    global BUILDING_STATE
    BUILDING_STATE = _cleanup_state(BUILDING_STATE)
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(BUILDING_STATE, f, ensure_ascii=False, indent=2)
