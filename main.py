# main.py (Angepasst an actions.py)
import os
import time
import subprocess
import sys

# Globale Daten und Klassen importieren
import game_data
import actions
from agent import Agent # Stellt sicher, dass dies die Ollama-Version ist

# Globale Variable für das benötigte Modell
REQUIRED_MODEL = "llama3:8b-instruct" # Oder dein gewähltes Modell

# --- Spielzustand ---
player = {
    "location": "start_room",
    "inventory": [],
    "has_won": False,
}

# Dynamischer Weltzustand (wird durch Aktionen geändert)
world_flags = game_data.world_flags.copy() # Kopie, um Original nicht zu ändern

# Aktive Agenten Instanzen (global zugänglich für Aktionen und main)
# WICHTIG: `active_agents` muss global sein, damit `do_examine` darauf zugreifen kann
active_agents = {}

# --- Modell-Check Funktion (wie zuvor) ---
def check_and_pull_model():
    """Prüft, ob das benötigte Ollama-Modell vorhanden ist und lädt es ggf. herunter."""
    print("Prüfe lokale Ollama-Modelle...")
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True, encoding='utf-8')
        if REQUIRED_MODEL in result.stdout:
            print(f"Modell '{REQUIRED_MODEL}' ist verfügbar.")
            return True
        else:
            print(f"Benötigtes Modell '{REQUIRED_MODEL}' nicht gefunden.")
            confirm = input(f"Soll das Modell '{REQUIRED_MODEL}' jetzt heruntergeladen werden? (Dies kann einige Zeit dauern) [j/N]: ")
            if confirm.lower() == 'j':
                print(f"Starte Download von '{REQUIRED_MODEL}'...")
                try:
                    process = subprocess.Popen(['ollama', 'pull', REQUIRED_MODEL], stdout=sys.stdout, stderr=sys.stderr)
                    process.wait()
                    if process.returncode == 0:
                        print(f"\nModell '{REQUIRED_MODEL}' erfolgreich heruntergeladen.")
                        return True
                    else:
                        print(f"\nFehler beim Herunterladen des Modells (Return Code: {process.returncode}).")
                        return False
                except FileNotFoundError:
                    print("FEHLER: Der Befehl 'ollama' wurde nicht gefunden. Ist Ollama korrekt installiert und im Systempfad?")
                    return False
                except Exception as e:
                    print(f"Fehler während des Modell-Downloads: {e}")
                    return False
            else:
                print("Modell-Download abgebrochen. Das Spiel kann nicht ohne das Modell funktionieren.")
                return False
    except FileNotFoundError:
        print("FEHLER: Der Befehl 'ollama' wurde nicht gefunden. Ist Ollama korrekt installiert und im Systempfad?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Auflisten der Ollama-Modelle (Code: {e.returncode}).")
        print("Stelle sicher, dass der Ollama-Dienst läuft.")
        if e.stderr: print("Ollama Fehlerausgabe:\n", e.stderr)
        if e.stdout: print("Ollama Ausgabe:\n", e.stdout)
        return False
    except Exception as e:
        print(f"Ein unerwarteter Fehler beim Prüfen der Ollama-Modelle: {e}")
        return False

# --- Agenten Initialisierung ---
def initialize_agents():
    """Erstellt Instanzen der Agent-Klasse."""
    global active_agents # Zugriff auf die globale Variable
    print("Die Präsenzen der Magier manifestieren sich...")
    for agent_id, definition in game_data.agent_definitions.items():
        active_agents[agent_id] = Agent(
            agent_id=agent_id,
            name=definition["name"],
            description=definition["description"],
            base_prompt=definition["base_prompt"]
        )
    print("Magier sind bereit.")


# --- Hilfsfunktion für Agentensuche (wird nur in main für 'sprich mit' gebraucht) ---
def find_agent_in_room_for_talk(agent_name_input):
    """Findet einen Agenten anhand des Namens im aktuellen Raum."""
    current_room_id = player["location"]
    agent_ids_in_room = game_data.world[current_room_id].get('agents', [])
    for agent_id in agent_ids_in_room:
        if agent_id in active_agents and agent_name_input.lower() == active_agents[agent_id].name.lower():
            return agent_id
    return None


# --- Siegbedingung prüfen ---
def check_win_condition(agent_response_text):
    """Prüft, ob die Antwort des Agenten die Siegbedingung auslöst."""
    # Muss global sein, da es player['has_won'] setzt
    global player
    win_keywords = ["stab der verjüngung", "zeit zurück", "reset", "fragment passt", "stab komplett"] # Schlüsselworte erweitert
    if isinstance(agent_response_text, str) and any(keyword in agent_response_text.lower() for keyword in win_keywords):
        # Optional: Genauere Prüfung, ob der Spieler auch das Fragment hat etc.
        print("\n***************************************************************")
        print("Ein seltsames Kribbeln durchfährt dich, als du die Worte hörst/liest.")
        print("Die Luft flimmert... war das die Information, die du gesucht hast?")
        print("Du hast etwas Wichtiges über den Stab der Verjüngung erfahren!")
        print("***************************************************************")
        player['has_won'] = True # Setze das Sieges-Flag


# --- Haupt-Spiel-Loop ---
def game_loop():
    """Die Hauptschleife des Spiels."""
    global player, world_flags # Zugriff auf globale Zustandsvariablen

    print("\nWillkommen bei Whispering Towers!")
    print("------------------------------------")
    print("Mögliche Befehle: 'gehe [richtung]', 'schaue', 'sprich mit [name]',")
    print("  'untersuche [objekt/person/feature]', 'nimm [objekt]', 'inventar',")
    print("  'benutze [objekt]' oder 'benutze [objekt] auf/mit [ziel]', 'quit'")
    print("------------------------------------")

    initialize_agents()

    # Zeige den ersten Raum an
    look_message, _ = actions.do_look(player, game_data.world, game_data.items, game_data.agent_definitions, game_data.room_features, world_flags)
    print(look_message)

    while not player['has_won']:
        user_input = input("\n> ")
        if not user_input: continue

        command, argument = actions.parse_input(user_input) # Nutze Parser aus actions

        message = ""
        success = False
        needs_room_description_update = False # Flag, ob der Raum neu beschrieben werden muss

        # --- Befehls-Verarbeitung über actions.py ---
        if command == "quit":
            actions.do_quit() # Beendet das Skript direkt
        elif command == "schaue":
            message, success = actions.do_look(player, game_data.world, game_data.items, game_data.agent_definitions, game_data.room_features, world_flags)
        elif command == "inventar":
            message, success = actions.do_inventory(player, game_data.items)
        elif command == "gehe":
            message, success = actions.do_go(player, game_data.world, game_data.room_features, world_flags, argument)
            if success: needs_room_description_update = True # Nach Bewegung neu schauen
        elif command == "untersuche":
            message, success = actions.do_examine(player, game_data.world, game_data.items, active_agents, game_data.room_features, world_flags, argument)
        elif command == "nimm":
            message, success = actions.do_take(player, game_data.world, game_data.items, argument)
            if success: needs_room_description_update = True # Raum hat sich geändert
        elif command == "benutze":
            message, success = actions.do_use(player, game_data.world, game_data.items, game_data.room_features, world_flags, argument)
            if success: needs_room_description_update = True # Benutzung könnte Raum ändern

        # --- Agenten-Interaktion (bleibt vorerst hier) ---
        elif command == "sprich":
            if not argument.startswith("mit "):
                 message = "Wen möchtest du ansprechen? (z.B. 'sprich mit Alatar')"
            else:
                agent_name_to_talk = argument[len("mit "):]
                agent_id = find_agent_in_room_for_talk(agent_name_to_talk)

                if agent_id:
                    agent = active_agents[agent_id]
                    current_room = game_data.world[player['location']]
                    print(f"\nDu wendest dich {agent.name} zu.")
                    player_dialogue_input = input(f"Deine Frage/Aussage an {agent.name}: ")

                    if not player_dialogue_input:
                        message = f"{agent.name} wartet..."
                    else:
                        # Rufe die LLM-Antwort ab
                        response_text = agent.get_response(player_dialogue_input, current_room['name'])
                        message = f"\n{agent.name}: \"{response_text}\""
                        # Prüfe nach jeder Agenten-Antwort die Siegbedingung
                        check_win_condition(response_text)
                        success = True # Gespräch war erfolgreich im Sinne einer Interaktion
                else:
                    message = f"Hier ist niemand namens '{agent_name_to_talk}'."
        else:
            message = "Unbekannter Befehl."

        # --- Ausgabe und Zustands-Update ---
        if message:
            print(message)

        if needs_room_description_update and not player['has_won']:
             # Zeige die neue Raumbeschreibung nach erfolgreicher Aktion
             look_message, _ = actions.do_look(player, game_data.world, game_data.items, game_data.agent_definitions, game_data.room_features, world_flags)
             print(look_message)


    # --- Spielende ---
    if player['has_won']:
        print("\nSpiel beendet. Du hast ein wichtiges Geheimnis aufgedeckt!")
    # Wenn die Schleife anders verlassen wird (z.B. durch quit), wird nichts mehr ausgegeben.


# --- Spiel starten ---
if __name__ == "__main__":
    # 1. Prüfen & Modell holen
    if not check_and_pull_model():
        print("\nErforderliches LLM-Modell nicht verfügbar oder Download fehlgeschlagen.")
        sys.exit(1)

    # 2. Spiel-Loop starten
    game_loop()