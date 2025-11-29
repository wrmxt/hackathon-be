import json


def seed_data():
    return {
        "building": {
            "id": "druzstevna-12",
            "name": "Družstevná 12",
            "city": "Bratislava",
            "flats_count": 24,
        },
        "residents": [
            {"id": "anna", "name": "Anna", "floor": 3, "trusted_score": 0.9},
            {"id": "peter", "name": "Peter", "floor": 5, "trusted_score": 0.8},
            {"id": "jana", "name": "Jana", "floor": 2, "trusted_score": 0.6},
        ],
        "items": [
            {"id": "item-1", "name": "Drill", "owner_id": "peter",
             "risk_level": "low", "status": "available"},
            {"id": "item-2", "name": "Party set", "owner_id": "anna",
             "risk_level": "low", "status": "available"},
        ],
        "borrowings": [],
        "events": [],
        "impact": {
            "co2_saved_kg": 0.0,
            "waste_avoided_kg": 0.0,
            "borrows_count": 0,
            "events_count": 0,
        },
    }


def persist():
    with open("./model/db.json", "w") as f:
        json.dump(BUILDING_STATE, f, indent=2)

BUILDING_STATE = json.load(open("./model/db.json"))

