# actions.py
import sys # Für sys.exit in quit

# --- Hilfsfunktionen (könnten auch in eine eigene utils.py) ---

def _find_item_in_list(item_id_or_name, item_list, items_data):
    """Sucht nach einer Item-ID in einer Liste anhand von ID oder Name."""
    target_id = item_id_or_name.lower()
    for item_id in item_list:
        if item_id in items_data and (target_id == item_id or target_id == items_data[item_id]['name'].lower()):
            return item_id
    return None

def _parse_use_command(argument_str):
    """Versucht, 'benutze X auf/mit Y' zu parsen. Gibt (objekt, target) zurück."""
    parts = argument_str.lower().split(maxsplit=1)
    object_name = parts[0] if parts else ""
    target_name = ""

    if len(parts) > 1:
        rest = parts[1]
        separators = [" auf ", " an ", " mit "] # Mögliche Trenner
        for sep in separators:
            if sep in rest:
                obj_part, target_part = rest.split(sep, 1)
                # Füge den Teil vor dem Separator wieder zum Objekt hinzu, falls getrennt
                object_name = f"{object_name} {obj_part}".strip()
                target_name = target_part.strip()
                break
        else: # Wenn kein Separator gefunden wurde, ist alles nach dem ersten Wort Teil des Objekts
             object_name = argument_str

    return object_name, target_name


# --- Aktions-Funktionen ---
# Jede Funktion gibt (Nachricht_an_Spieler, Erfolg_oder_nicht) zurück

def do_quit():
    """Beendet das Spiel."""
    print("Auf Wiedersehen!")
    sys.exit(0)
    # return "Auf Wiedersehen!", True # Wird nie erreicht, aber der Vollständigkeit halber

def do_look(player, world_data, items_data, agents_data, features_data, world_flags):
    """Beschreibt den aktuellen Raum."""
    room_id = player["location"]
    if room_id not in world_data:
        return "Fehler: Du befindest dich an einem unbekannten Ort!", False
    room = world_data[room_id]

    # Baue die Beschreibung zusammen
    description = f"\n--- {room['name']} ---\n"
    description += room['description'] + "\n"

    # Features im Raum beschreiben
    feature_ids = room.get('features', [])
    if feature_ids:
        for feature_id in feature_ids:
            if feature_id in features_data.get(room_id, {}):
                feature = features_data[room_id][feature_id]
                # Nutze die get_description Funktion für dynamische Beschreibungen
                desc_func = feature.get('get_description')
                if callable(desc_func):
                    description += "\n" + desc_func(world_flags)
                else: # Fallback auf statische Beschreibung oder Namen
                     description += f"\nDu siehst hier: {feature.get('name', feature_id)}."

    # Items im Raum auflisten
    items_in_room_ids = room.get('items', [])
    if items_in_room_ids:
        description += "\nDu siehst hier herumliegen:"
        for item_id in items_in_room_ids:
            if item_id in items_data:
                description += f"\n- {items_data[item_id]['name']} ({item_id})"
            else:
                 description += f"\n- Ein unbekanntes Objekt ({item_id})"

    # Agenten im Raum auflisten
    agent_ids_in_room = room.get('agents', [])
    if agent_ids_in_room:
        description += "\nAnwesende Personen:"
        # Greife auf die globalen Agenten-Daten zu (Namen etc.)
        # `active_agents` wird hier nicht direkt gebraucht, nur die Definitionen
        from game_data import agent_definitions # Import hier oder global? Besser global in main.
        for agent_id in agent_ids_in_room:
            if agent_id in agent_definitions:
                 description += f"\n- {agent_definitions[agent_id]['name']}"
            else:
                 description += f"\n- Eine unbekannte Person ({agent_id})"

    # Ausgänge auflisten (besondere Behandlung für Türen)
    exits = room.get('exits', {})
    possible_exits = []
    if exits:
        for direction, target_room in exits.items():
             possible_exits.append(f"- {direction.capitalize()}")

    # Prüfe Features auf offene Türen als Ausgänge
    for feature_id in feature_ids:
         if feature_id in features_data.get(room_id, {}):
             feature = features_data[room_id][feature_id]
             if feature.get('is_door', False) and world_flags.get(f"{feature_id}_unlocked", False):
                 # Finde die Richtung, die diese Tür repräsentiert (z.B. aus der ID 'holztuer_norden')
                 direction = feature_id.split('_')[-1] # Extrahiere 'norden'
                 possible_exits.append(f"- {direction.capitalize()} (durch die offene {feature.get('name', 'Tür')})")


    if possible_exits:
         description += "\nMögliche Ausgänge:" + "".join(sorted(possible_exits)) # Sortiert für Konsistenz
    else:
         description += "\nEs gibt keine offensichtlichen Ausgänge."


    return description, True # Looking ist immer erfolgreich

def do_go(player, world_data, features_data, world_flags, direction):
    """Bewegt den Spieler in eine Richtung."""
    room_id = player["location"]
    if room_id not in world_data:
        return "Fehler: Du bist verloren!", False
    room = world_data[room_id]

    # Prüfe normale Ausgänge
    exits = room.get('exits', {})
    if direction in exits:
        player["location"] = exits[direction]
        return f"Du gehst nach {direction.capitalize()}.", True

    # Prüfe Features (offene Türen)
    feature_ids = room.get('features', [])
    for feature_id in feature_ids:
         # Prüfe, ob der Feature-Name die Richtung enthält (z.B. holztuer_norden -> norden)
         if direction in feature_id and feature_id in features_data.get(room_id, {}):
             feature = features_data[room_id][feature_id]
             # Ist es eine Tür und ist sie offen?
             if feature.get('is_door', False) and world_flags.get(f"{feature_id}_unlocked", False):
                  target_room = feature.get('target_room_if_unlocked')
                  if target_room and target_room in world_data:
                       player["location"] = target_room
                       return f"Du gehst durch die offene {feature.get('name', 'Tür')} nach {direction.capitalize()}.", True
                  else:
                       return f"Die offene {feature.get('name', 'Tür')} führt scheinbar nirgendwohin (Zielraum nicht definiert).", False
             elif feature.get('is_door', False): # Tür ist da, aber verschlossen
                  return f"Die {feature.get('name', 'Tür')} nach {direction.capitalize()} ist verschlossen.", False

    return "Du kannst nicht in diese Richtung gehen.", False


def do_examine(player, world_data, items_data, agents_data, features_data, world_flags, target_name):
    """Untersucht ein Objekt, eine Person oder ein Feature."""
    if not target_name:
        return "Was möchtest du untersuchen?", False

    # 1. Prüfe Agenten im Raum
    from main import active_agents # Greife auf die Instanzen aus main zu
    current_room_id = player["location"]
    agent_ids_in_room = world_data[current_room_id].get('agents', [])
    for agent_id in agent_ids_in_room:
         if agent_id in active_agents and target_name.lower() == active_agents[agent_id].name.lower():
              agent = active_agents[agent_id]
              return f"{agent.name}: {agent.description}", True

    # 2. Prüfe Items (Inventar zuerst, dann Raum)
    # Im Inventar
    item_id_inv = _find_item_in_list(target_name, player['inventory'], items_data)
    if item_id_inv:
        return f"{items_data[item_id_inv]['name']}: {items_data[item_id_inv]['description']}", True
    # Im Raum
    items_in_room = world_data[current_room_id].get('items', [])
    item_id_room = _find_item_in_list(target_name, items_in_room, items_data)
    if item_id_room:
         return f"{items_data[item_id_room]['name']}: {items_data[item_id_room]['description']}", True

    # 3. Prüfe Features im Raum
    feature_ids = world_data[current_room_id].get('features', [])
    for feature_id in feature_ids:
         if feature_id in features_data.get(current_room_id, {}):
              feature = features_data[current_room_id][feature_id]
              if target_name.lower() == feature['name'].lower() or target_name.lower() == feature_id:
                  desc_func = feature.get('get_description')
                  if callable(desc_func):
                      return desc_func(world_flags), True
                  else: # Fallback
                       return f"Du untersuchst {feature.get('name', feature_id)}. {feature.get('description', '')}", True


    return f"Du siehst hier nichts namens '{target_name}', das du untersuchen könntest.", False

def do_take(player, world_data, items_data, item_name_input):
    """Nimmt ein Item aus dem aktuellen Raum."""
    if not item_name_input:
        return "Was möchtest du nehmen?", False

    current_room_id = player["location"]
    items_in_room = world_data[current_room_id].get('items', [])

    item_id = _find_item_in_list(item_name_input, items_in_room, items_data)

    if item_id:
        if items_data[item_id]['takeable']:
            player['inventory'].append(item_id)
            world_data[current_room_id]['items'].remove(item_id) # Wichtig: Modifiziere die Weltdaten direkt
            return f"Du nimmst: {items_data[item_id]['name']}", True
        else:
            return f"Das kannst du nicht nehmen.", False
    else:
        # Prüfen, ob der Spieler es schon hat
        item_id_inv = _find_item_in_list(item_name_input, player['inventory'], items_data)
        if item_id_inv:
            return "Du hast das bereits.", False
        else:
            return f"Du siehst hier nichts namens '{item_name_input}', das du nehmen könntest.", False

def do_inventory(player, items_data):
    """Zeigt das Inventar des Spielers an."""
    if not player['inventory']:
        return "Dein Inventar ist leer.", True
    else:
        message = "Du hast bei dir:"
        for item_id in player['inventory']:
            if item_id in items_data:
                message += f"\n- {items_data[item_id]['name']} ({item_id})"
            else:
                message += f"\n- Unbekanntes Item ({item_id})"
        return message, True

def do_use(player, world_data, items_data, features_data, world_flags, argument_str):
    """Benutzt ein Item (alleine oder auf einem Ziel)."""
    if not argument_str:
        return "Was möchtest du benutzen?", False

    object_name, target_name = _parse_use_command(argument_str)

    if not object_name:
         return "Welches Objekt möchtest du benutzen?", False

    # Finde das zu benutzende Objekt im Inventar
    object_id = _find_item_in_list(object_name, player['inventory'], items_data)
    if not object_id:
        return f"Du hast '{object_name}' nicht im Inventar.", False

    item_def = items_data[object_id]
    target_id = None
    target_type = None # 'item' or 'feature'

    # --- Fall 1: Ziel angegeben ---
    if target_name:
        # Suche Ziel: Item im Inventar?
        target_id_inv = _find_item_in_list(target_name, player['inventory'], items_data)
        if target_id_inv:
             target_id = target_id_inv
             target_type = 'item'
        else:
            # Suche Ziel: Item im Raum?
            current_room_id = player["location"]
            items_in_room = world_data[current_room_id].get('items', [])
            target_id_room = _find_item_in_list(target_name, items_in_room, items_data)
            if target_id_room:
                 target_id = target_id_room
                 target_type = 'item'
            else:
                 # Suche Ziel: Feature im Raum?
                 feature_ids = world_data[current_room_id].get('features', [])
                 for f_id in feature_ids:
                      if f_id in features_data.get(current_room_id, {}):
                           feature = features_data[current_room_id][f_id]
                           if target_name.lower() == feature['name'].lower() or target_name.lower() == f_id:
                                target_id = f_id
                                target_type = 'feature'
                                break # Feature gefunden

        if not target_id:
            return f"Ziel '{target_name}' nicht gefunden.", False

        # Prüfe 'use_on_target'-Regeln des Objekts
        if 'use_on_target' in item_def and target_id in item_def['use_on_target']:
            rule = item_def['use_on_target'][target_id]

            # Prüfe optionale Bedingung der Regel
            condition_func = rule.get('condition')
            if condition_func and not condition_func(player, world_flags):
                return rule.get('effect_fail', "Das funktioniert so nicht."), False

            # Prüfe optionale Erfolgsbedingung
            success_cond_func = rule.get('success_condition')
            is_success = True # Standardmäßig erfolgreich, wenn keine Bedingung da ist
            if success_cond_func:
                 is_success = success_cond_func(player, world_flags)

            if is_success:
                 message = rule.get('effect_success', f"Du benutzt {item_def['name']} auf {target_name}.") # Standard-Erfolgsnachricht
                 # Nebeneffekte bei Erfolg:
                 flag_to_set = rule.get('set_world_flag_on_success')
                 if flag_to_set:
                     world_flags[flag_to_set] = True # Weltzustand ändern
                     print(f"[DEBUG: Flag '{flag_to_set}' gesetzt auf True]")

                 # Spezialfall: Truhe öffnen -> Inhalt hinzufügen
                 if target_id == "truhe_west" and flag_to_set == "west_tower_chest_unlocked":
                      new_item = "stab_fragment" # Item in der Truhe
                      if 'items' not in world_data[player['location']]:
                           world_data[player['location']]['items'] = []
                      world_data[player['location']]['items'].append(new_item)
                      message += f" In der Truhe findest du: {items_data[new_item]['name']}!"

                 # Objekt verbrauchen?
                 if item_def.get('consumable', False):
                     player['inventory'].remove(object_id)
                     message += f" ({item_def['name']} wurde verbraucht.)"

                 return message, True
            else:
                 # Erfolg nicht möglich (z.B. Tür schon offen)
                 return rule.get('effect_fail', "Das hat keinen Effekt mehr."), False
        else:
             # Keine Regel für dieses Objekt auf dieses Ziel
             return f"Du kannst '{object_name}' nicht auf '{target_name}' anwenden.", False

    # --- Fall 2: Kein Ziel angegeben ---
    else:
        effect = item_def.get('use_alone_effect')
        if effect:
            message = effect
            success = item_def.get('success', True) # Standardmäßig erfolgreich

            # Objekt verbrauchen?
            if item_def.get('consumable', False):
                player['inventory'].remove(object_id)
                message += f" ({item_def['name']} wurde verbraucht.)"

            # Flag setzen? (selten bei 'use alone')
            flag_to_set = item_def.get('set_world_flag_on_success')
            if flag_to_set:
                 world_flags[flag_to_set] = True

            return message, success
        else:
            return f"Wie möchtest du '{object_name}' benutzen?", False