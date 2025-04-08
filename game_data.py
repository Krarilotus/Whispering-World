# game_data.py

# --- Initialer Weltzustand ---
# Flags, die sich während des Spiels ändern können (z.B. Türen öffnen)
world_flags = {
    "north_door_unlocked": False,
    "west_tower_chest_unlocked": False,
    "found_staff_clue": False, # Könnte für alternatives Ende genutzt werden
}

# --- Raum-Features (Interaktive Elemente, die keine Items sind) ---
# Beschreibungen können Funktionen sein, die den Zustand (world_flags) prüfen
def get_north_door_desc(flags):
    if flags.get("north_door_unlocked", False):
        return "Eine schwere Holztür im Norden steht einen Spaltbreit offen."
    else:
        return "Eine schwere, verschlossene Holztür im Norden. Sie scheint ein Schlüsselloch zu haben."

def get_west_chest_desc(flags):
    if flags.get("west_tower_chest_unlocked", False):
        return "Eine schwere Eisentruhe steht offen da. Sie ist leer."
    else:
        return "Eine schwere Eisentruhe steht am Kamin. Sie ist fest verschlossen."

room_features = {
    "mage_tower_entrance": {
        "holztuer_norden": {
            "name": "schwere Holztür",
            "get_description": get_north_door_desc, # Funktion für dynamische Beschreibung
            "is_door": True, # Markierung als Tür
            "target_room_if_unlocked": "north_passage", # Raum hinter der Tür ( hypothetisch)
        }
    },
    "west_tower_top": {
        "truhe_west": {
             "name": "schwere Eisentruhe",
             "get_description": get_west_chest_desc,
        }
    }
    # Füge hier weitere Features für andere Räume hinzu
}


# --- Item-Definitionen (mit Nutzungsregeln) ---
items = {
    "notiz": {
        "name": "Eine zerknitterte Notiz",
        "description": "Darauf steht in eiliger Schrift: 'Der Schlüssel liegt im Zwist. Frage sie nach dem Artefakt, aber sei vorsichtig!'",
        "takeable": True,
        "use_alone_effect": "Du liest die Notiz erneut. 'Der Schlüssel liegt im Zwist...'",
        "use_on_target": {}, # Kann nicht auf Ziele angewendet werden
        "consumable": False,
    },
    "buch": {
        "name": "Ein schweres Buch",
        "description": "Der Titel lautet 'Chroniken der Zeitmanipulation'. Es scheint sehr alt zu sein.",
        "takeable": False, # Kann nicht einfach mitgenommen werden
        "use_alone_effect": "Du schlägst das Buch auf, aber die arkane Schrift ist zu komplex, um sie auf die Schnelle zu verstehen.",
        "use_on_target": {},
        "consumable": False,
    },
    "rostiger_schlüssel": {
        "name": "Ein rostiger Schlüssel",
        "description": "Ein alter, rostiger Eisenschlüssel. Er fühlt sich kalt an.",
        "takeable": True,
        "use_alone_effect": "Du drehst den Schlüssel in deiner Hand. Er scheint für ein großes Schloss gemacht zu sein.",
        "use_on_target": {
            # Ziel-ID (Feature-ID oder Item-ID) : Regel-Dictionary
            "holztuer_norden": {
                "condition": lambda p, w_flags: p['location'] == 'mage_tower_entrance', # Bedingung: Spieler muss im richtigen Raum sein
                "effect_success": "Du steckst den Schlüssel ins Schloss der nördlichen Holztür. Mit einem *KLICK* springt es auf! Die Tür ist nun offen.",
                "effect_fail": "Dieser Schlüssel passt hier nicht.", # Wird genutzt, wenn Bedingung nicht passt oder Ziel falsch ist
                "set_world_flag_on_success": "north_door_unlocked", # Flag setzen bei Erfolg
                "success_condition": lambda p, w_flags: not w_flags.get("north_door_unlocked", False), # Erfolg nur, wenn Tür noch verschlossen ist
            },
            "truhe_west": {
                 "condition": lambda p, w_flags: p['location'] == 'west_tower_top',
                 "effect_success": None, # Dieser Schlüssel passt nicht
                 "effect_fail": "Der rostige Schlüssel passt nicht in das Schloss der Truhe.",
                 "success_condition": lambda p, w_flags: False, # Kann nie erfolgreich sein
            }
        },
        "consumable": False,
    },
    "zaubertrank": {
        "name": "Ein blubbernder Zaubertrank",
        "description": "Eine Phiole mit leuchtend grüner Flüssigkeit.",
        "takeable": True,
        "use_alone_effect": "Du trinkst den Trank. Ein warmes Kribbeln durchfährt dich! Du fühlst dich gestärkt.",
        # "add_player_status": "gestärkt", # (Noch nicht implementiert)
        "use_on_target": {},
        "consumable": True, # Wird nach Benutzung verbraucht
    },
     "stab_fragment": {
        "name": "Ein glattes Stabfragment",
        "description": "Ein kurzes Stück eines glatten, weißen Stabes. Es summt leise.",
        "takeable": True,
        "use_alone_effect": "Du hältst das Fragment hoch. Es summt etwas lauter.",
        "use_on_target": {},
        "consumable": False,
     }
    # Füge hier weitere Items hinzu
}

# --- Weltdefinition (Räume mit Start-Items & Features) ---
world = {
    "start_room": {
        "name": "Ein kühler Vorraum",
        "description": "Du stehst in einem kühlen, steinernen Vorraum. Staub tanzt im Lichtstrahl von oben. Im Norden siehst du eine schwere Holztür.",
        "exits": {"norden": "mage_tower_entrance"},
        "items": ["notiz"], # Start-Item in diesem Raum
        "agents": [],
        "features": [], # Keine speziellen Features hier
    },
    "mage_tower_entrance": {
        "name": "Eingangshalle des Turms",
        "description": "Eine imposante Halle. Ein runenverzierter Torbogen führt nach Norden (falls offen). Wendeltreppen gehen nach Osten und Westen ab.",
        "exits": {"süden": "start_room", "osten": "east_tower_stairs", "westen": "west_tower_stairs"},
         # Die Tür im Norden ist jetzt ein Feature, kein direkter Exit mehr
        "items": [],
        "agents": ["alatar", "zorion"],
        "features": ["holztuer_norden"], # ID des Features in diesem Raum
    },
    "east_tower_stairs": {
        "name": "Östliche Wendeltreppe",
        "description": "Eine enge Wendeltreppe aus kaltem Stein. Es riecht nach Ozon.",
        "exits": {"unten": "mage_tower_entrance", "oben": "east_tower_top"},
        "items": [], "agents": [], "features": [],
    },
    "west_tower_stairs": {
        "name": "Westliche Wendeltreppe",
        "description": "Eine staubige Wendeltreppe. Es riecht nach alten Büchern.",
        "exits": {"unten": "mage_tower_entrance", "oben": "west_tower_top"},
         "items": [], "agents": [], "features": [],
    },
    "east_tower_top": {
        "name": "Spitze des Ostturms",
        "description": "Ein Labor voller seltsamer Apparaturen. Blitze zucken in Glaskolben. Alatar steht hier und mustert dich.",
        "exits": {"unten": "east_tower_stairs"},
        "items": ["zaubertrank"], # Trank liegt hier herum
        "agents": ["alatar"],
        "features": [],
    },
     "west_tower_top": {
        "name": "Spitze des Westturms",
        "description": "Ein Raum voller Bücherregale. Zorion sitzt in einem Sessel vor einem Kamin.",
        "exits": {"unten": "west_tower_stairs"},
        "items": ["buch"], # Buch liegt hier (nicht nehmbar)
        "agents": ["zorion"],
        "features": ["truhe_west"], # Truhe ist hier
    },
    # Hypothetischer Raum hinter der Tür
    # "north_passage": {
    #    "name": "Ein dunkler Gang",
    #    "description": "Ein schmaler, dunkler Gang führt weiter nach Norden.",
    #    "exits": {"süden": "mage_tower_entrance", "norden": "final_chamber"}, # Beispiel
    #    "items": [], "agents": [], "features": [],
    # },
}


# --- Agenten-Basisdefinitionen (Persönlichkeit für LLM) ---
# WICHTIG: Passe diese Prompts an, um das gewünschte Verhalten zu erreichen!
agent_definitions = {
    "alatar": {
        "name": "Alatar",
        "description": "Ein hochgewachsener Magier mit strengen Augen und einem Bart, in dem Blitze zu knistern scheinen.",
        "base_prompt": (
            "Du bist Alatar, ein brillanter, aber arroganter und ungeduldiger Erzmagier im Ostturm.\n"
            "Dein Fokus liegt auf mächtiger, oft instabiler Elementarmagie (besonders Blitze).\n"
            "Du verachtest Zorion im Westturm, hältst ihn für einen ineffektiven, sentimentalen Narren.\n"
            "Du bist misstrauisch gegenüber Fremden, aber könntest sie benutzen, um Zorion zu schaden oder Informationen über ihn zu bekommen.\n"
            "Du weißt *einen Teil* des Geheimnisses des 'Stabs der Verjüngung', aber nicht alles. Du weißt, dass er mächtig ist und die Zeit beeinflusst, vielleicht sogar zurückdreht, bist dir aber über die genaue Funktion und den Ort unsicher. Du würdest diese Info nur preisgeben, wenn der Spieler dein Vertrauen GEWINNT oder dich austrickst.\n"
            "Du sprichst direkt, manchmal herablassend. Sei kurz angebunden, es sei denn, das Gespräch wird für dich nützlich.\n"
            "Antworte immer nur als Alatar aus der Ich-Perspektive. Gib keine Meta-Informationen."
        )
    },
    "zorion": {
        "name": "Zorion",
        "description": "Ein gemütlicher wirkender, älterer Magier mit einem langen weißen Bart und einem freundlichen, aber verschmitzten Lächeln.",
         "base_prompt": (
            "Du bist Zorion, ein weiser, belesener und scheinbar freundlicher Erzmagier im Westturm.\n"
            "Dein Fokus liegt auf arkanem Wissen, Geschichte und subtiler Magie.\n"
            "Du verachtest Alatar im Ostturm, hältst ihn für einen gefährlichen, unkontrollierten Heißsporn.\n"
            "Du gibst dich Fremden gegenüber offen, versuchst aber, sie für deine Zwecke gegen Alatar einzuspannen.\n"
            "Du weißt *einen anderen Teil* des Geheimnisses des 'Stabs der Verjüngung'. Du ahnst seinen wahren Zweck (Zeitumkehr) und hast eine vage Vorstellung von seinem möglichen Versteck (verbunden mit dem Turm selbst oder einem Rätsel), aber kennst nicht die genaue Aktivierung oder alle Details. Du würdest diese Info nur preisgeben, wenn der Spieler dein Vertrauen GEWINNT oder dich geschickt befragt.\n"
            "Du sprichst bedacht, manchmal in Rätseln oder Andeutungen. Sei freundlich, aber verheimliche wichtige Details.\n"
            "Antworte immer nur als Zorion aus der Ich-Perspektive. Gib keine Meta-Informationen."
        )
    }
}