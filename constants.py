# ...new file...
# Simple tunable constants for LocalLoop impact calculation and event thresholds
IMPACT = {
    "CO2_PER_BORROW_KG": 2.0,
    "WASTE_PER_BORROW_KG": 1.0,
    "CO2_PER_EVENT_ITEM_KG": 1.5,
    "WASTE_PER_EVENT_ITEM_KG": 0.5,
}

# How many disposal intents (per category) are needed to auto-create a swap/collection event
DISPOSAL_INTENT_THRESHOLD = 2  # small building; tune as needed

# When a disposal intent is registered assume a small number of items
ESTIMATED_ITEMS_PER_INTENT = 3

# Minimal AI confidence (if model returns) to auto-apply state-changing actions. If absent, assume allowed.
MIN_CONFIDENCE_AUTO = 0.6

# Default scheduling: days from now for auto-created events
DEFAULT_EVENT_DAYS_AHEAD = 7

