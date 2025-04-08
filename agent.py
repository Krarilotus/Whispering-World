# agent.py (Angepasste Version für Ollama)
import requests # Ersetzt google.generativeai
import json
import time

MAX_MEMORY_TURNS = 100
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate" # Ollamas Generate-Endpunkt
# Oder für OpenAI-kompatiblen Chat: OLLAMA_CHAT_ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"

# Wähle das Modell, das du mit 'ollama pull' heruntergeladen hast
# Stelle sicher, dass der Name exakt übereinstimmt!
MODEL_NAME = "llama3:8b-instruct" # ODER "mistral:7b-instruct", etc.

class Agent:
    def __init__(self, agent_id, name, description, base_prompt):
        self.id = agent_id
        self.name = name
        self.description = description
        self.base_prompt = base_prompt
        # Gedächtnis muss jetzt zum Ollama Format passen (einfache Liste für /api/generate)
        # Für den Chat-Endpunkt (/v1/...) wäre es eine Liste von {"role": ..., "content": ...}
        self.memory_context = "" # Einfacher String-Kontext für /api/generate
        self.memory_turns = [] # Liste von (Spieler-Input, Agenten-Antwort) Paaren

    def _update_memory(self, player_input, agent_response):
        """Aktualisiert das Gedächtnis (einfache String-Konkatenation für /api/generate)."""
        self.memory_turns.append((player_input, agent_response))
        if len(self.memory_turns) > MAX_MEMORY_TURNS:
            self.memory_turns.pop(0) # Entferne ältesten Turn

        # Baue den Kontext-String neu auf
        self.memory_context = "\n--- Conversation History ---\n"
        for p_input, a_resp in self.memory_turns:
            self.memory_context += f"Player: {p_input}\n{self.name}: {a_resp}\n"
        self.memory_context += "--- End History ---\n"


    def get_response(self, player_input, current_location_name):
        """Generiert eine Antwort vom lokalen Ollama LLM."""

        # Baue den vollständigen Prompt für Ollama (/api/generate)
        # System Prompt + Gedächtnis + Aktuelle Situation + Spieler Input
        full_prompt = (
            f"{self.base_prompt}\n"
            f"{self.memory_context}\n" # Füge die Historie ein
            f"--- Current Situation ---\n"
            f"You are in: {current_location_name}.\n"
            f"Player: {player_input}\n"
            f"{self.name}: " # Wichtig: Lasse das LLM hier weiterschreiben
        )

        print(f"[{self.name} denkt nach (lokal)...]")
        time.sleep(0.1) # Kurze Pause, lokale Inferenz dauert sowieso

        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": False, # Wir wollen die komplette Antwort, kein Streaming
            # --- Optionen (optional, zum Experimentieren) ---
            # "options": {
            #   "temperature": 0.7, # Kreativität vs. Fokus (0=deterministisch, >1=kreativer)
            #   "num_ctx": 2048,    # Kontextfenstergröße (an Modell anpassen)
            #   "top_p": 0.9,       # Wahrscheinlichkeits-Sampling
            # }
        }

        try:
            # Sende Anfrage an den lokalen Ollama Server
            response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120) # Timeout erhöhen!
            response.raise_for_status() # Fehler werfen, wenn Status nicht 200 OK ist

            response_data = response.json()
            agent_response_text = response_data.get('response', '').strip()

            # Filter leere Antworten oder nur Leerzeichen
            if not agent_response_text:
                 agent_response_text = f"{self.name} schweigt oder sammelt seine Gedanken."

            # Aktualisiere das Gedächtnis
            self._update_memory(player_input, agent_response_text)

            return agent_response_text

        except requests.exceptions.ConnectionError:
            print(f"!! FEHLER: Kann keine Verbindung zum Ollama-Server herstellen ({OLLAMA_ENDPOINT}).")
            print("!! Stelle sicher, dass Ollama läuft!")
            return f"{self.name} scheint nicht erreichbar zu sein. Läuft der lokale Server?"
        except requests.exceptions.Timeout:
            print(f"!! FEHLER: Zeitüberschreitung bei der Anfrage an Ollama. Das Modell braucht möglicherweise zu lange.")
            return f"{self.name} braucht ungewöhnlich lange zum Nachdenken... vielleicht ist er überlastet."
        except requests.exceptions.RequestException as e:
            print(f"!! Fehler bei der Kommunikation mit Ollama ({self.name}): {e}")
            # Versuche, die Antwort des Servers zu loggen, falls vorhanden
            try:
                print(f"!! Server-Antwort (falls vorhanden): {response.text}")
            except Exception:
                pass # Keine Antwort vorhanden
            return f"{self.name} hat gerade... Schwierigkeiten, einen klaren Gedanken zu fassen."
        except json.JSONDecodeError:
             print(f"!! FEHLER: Ungültige JSON-Antwort von Ollama: {response.text}")
             return f"{self.name} murmelt etwas Unverständliches."


# Hinweis: Wenn du den OpenAI-kompatiblen Endpunkt (/v1/chat/completions) verwenden willst,
# müsstest du die 'openai' Bibliothek installieren (`pip install openai`) und den Code anpassen,
# um die Chat-Struktur (Liste von Dictionaries mit 'role' und 'content') zu verwenden.
# Das ist oft etwas sauberer für Dialoge. Beispiel:
#
# from openai import OpenAI
# client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama') # API Key ist für Ollama egal
# response = client.chat.completions.create(
#    model=MODEL_NAME,
#    messages=[
#        {"role": "system", "content": self.base_prompt},
#        *self.memory, # Memory müsste Liste von Dics sein
#        {"role": "user", "content": player_input}
#    ]
# )
# agent_response_text = response.choices[0].message.content