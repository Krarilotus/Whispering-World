# Project: Whispering World (Arbeitstitel)

## 🚀 Schnellstart / How to Run

Dieses Projekt verwendet Python und ein lokales Large Language Model (LLM) über Ollama.

**Voraussetzungen:**

* **Python:** Python 3.8 oder neuer installiert ([python.org](https://www.python.org/)).
* **Ollama:** Ollama muss auf deinem System installiert sein und laufen ([ollama.com](https://ollama.com/)).

**Setup & Start:**

1.  **Code herunterladen/klonen:**
    ```bash
    # Wenn du Git benutzt:
    git clone <URL-deines-Git-Repositorys>
    cd whispering-world
    # Ansonsten: Lade den Code als ZIP herunter und entpacke ihn.
    # Wechsle dann im Terminal in das entpackte Verzeichnis.
    ```

2.  **Abhängigkeiten installieren:**
    Es wird empfohlen, eine virtuelle Umgebung zu verwenden.
    ```bash
    # Erstelle eine virtuelle Umgebung (optional, aber empfohlen)
    python -m venv .venv
    # Aktiviere sie (Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate)

    # Installiere die benötigten Python-Pakete
    pip install -r requirements.txt
    ```
    *(Stelle sicher, dass die `requirements.txt`-Datei `requests` und `python-dotenv` enthält, basierend auf unserem letzten Code.)*

3.  **Ollama starten:**
    **WICHTIG:** Stelle sicher, dass die Ollama-Anwendung oder der Ollama-Dienst auf deinem Computer im Hintergrund läuft, *bevor* du das Spiel startest!

4.  **Spiel starten:**
    Führe das Hauptskript im Terminal aus dem Projektverzeichnis heraus aus:
    ```bash
    python main.py
    ```

5.  **Erster Start / Modell-Download:**
    Wenn du das Spiel zum ersten Mal startest und das benötigte LLM (standardmäßig `llama3:8b-instruct`) noch nicht über Ollama heruntergeladen hast, wird das Skript dies erkennen. Es fragt dich dann, ob es das Modell jetzt herunterladen soll. Bestätige mit 'j' (oder 'y'). Der Download kann je nach Modellgröße und Internetverbindung einige Zeit dauern.

---

## 📜 Konzept

**Whispering Worlds** ist ein experimentelles Textadventure, das klassische Erkundung und Rätsellösung mit dynamischer Interaktion durch KI-gesteuerte Charaktere (Agenten) verbindet. Spieler navigieren durch eine Welt bestehend aus miteinander verbundenen Orten ("Räumen"), interagieren mit Objekten und verfolgen ein übergeordnetes Ziel.

Das Besondere an diesem Projekt ist der Einsatz von Sprachmodellen (LLMs), um NSCs (Nicht-Spieler-Charaktere) zum Leben zu erwecken. Diese Agenten besitzen:

* **Eigene Persönlichkeiten und Agenden:** Sie verfolgen eigene Ziele, die nicht immer mit denen des Spielers übereinstimmen.
* **Kontextuelles Gedächtnis:** Sie erinnern sich an vergangene Interaktionen mit dem Spieler.
* **Natürliche Sprachverarbeitung:** Spieler können über Texteingaben in natürlicher Sprache mit ihnen kommunizieren, über vordefinierte Befehle hinaus.
* **Dynamisches Verhalten:** Ihre Reaktionen und ihre Hilfsbereitschaft (oder Feindseligkeit) entwickeln sich basierend auf dem Verhalten des Spielers.

Das Spiel beinhaltet Rogelike-Elemente, die Wiederspielbarkeit fördern und bei bestimmten Ereignissen oder Entscheidungen zu einem Neustart des Spielzyklus führen können. Ein einzelner Durchlauf ist auf eine Spielzeit von ca. 10-30 Minuten ausgelegt.

**Zielgruppe:** Spieler von Textadventures, Fans von interaktiver Fiktion und Personen, die an der Anwendung von KI in Spielen interessiert sind.

## 🏗️ Geplante Architektur & Technischer Ansatz

Das Spiel wird modular aufgebaut sein, um Erweiterbarkeit und Wartbarkeit zu gewährleisten.

1.  **Game Engine Core (Python - `main.py`):**
    * Verwaltet den Spielzustand (Position des Spielers, Inventar, Welt-Flags).
    * Implementiert die grundlegende Spiel-Loop (Input -> Processing -> Output).
    * Orchestriert Aufrufe an Aktions- und Agenten-Module.

2.  **Action Logic (Python - `actions.py`):**
    * Enthält Funktionen für Standard-Spieleraktionen (`gehe`, `nimm`, `untersuche`, `benutze`, `inventar`, `schaue`).
    * Modifiziert den Spielzustand basierend auf den Aktionen und gibt Feedback-Nachrichten zurück.

3.  **World Representation (Python - `game_data.py`):**
    * Definition von Räumen (mit Exits, Start-Items, Features).
    * Definition von Items (mit Eigenschaften wie `takeable`, `consumable` und detaillierten `use_on_target` / `use_alone_effect` Regeln).
    * Definition von Raum-Features (Türen, Truhen etc.) mit Zuständen.
    * Definition der Agenten-Basis-Prompts und Persönlichkeiten.
    * Initialer Weltzustand (`world_flags`).

4.  **Agenten-Modul (Python - `agent.py` / KI-Integration):**
    * **Herzstück des Projekts.** Die `Agent`-Klasse verwaltet einzelne KI-NSCs.
    * **Prompt Engineering:** Nutzt den Basis-Prompt aus `game_data.py`.
    * **Gedächtnis-Management:** Implementiert eine (aktuell einfache) Konversationshistorie für den Prompt-Kontext. (LangMem/Alternativen als zukünftige Erweiterung).
    * **LLM-Anbindung:** Kommuniziert mit dem **lokalen Ollama-Server**, um Antworten basierend auf Prompt, Gedächtnis und Spieler-Input zu generieren.

5.  **State Management:**
    * Der Spielzustand (`player`, `world_flags`) wird aktuell im Speicher gehalten und geht beim Beenden verloren. (Persistenz wäre eine Erweiterung).

## 🚀 Technologie-Stack (Aktuell)

* **Programmiersprache:** Python 3
* **LLM-Integration:** Lokales LLM via **Ollama**
* **Python-Bibliotheken:** `requests` (für Ollama API), `python-dotenv` (optional für Konfiguration)
* **Datenstrukturen:** Python Dictionaries und Listen in `game_data.py`

## 🎯 Ziele & Motivation

* Entwicklung eines spielbaren Prototyps mit mindestens 1-2 KI-Agenten via lokalem LLM.
* Erforschung der Möglichkeiten und Herausforderungen bei der Integration von LLMs in interaktive Fiktion.
* Schaffung eines unterhaltsamen Spielerlebnisses für Freunde und Interessierte.
* Aufbau eines flexiblen Frameworks, das potenziell für weitere Mini-Adventures wiederverwendet werden kann.

## 💡 Herausforderungen

* Effektives Prompt-Design für glaubwürdige und konsistente Agenten-Persönlichkeiten mit lokalen Modellen.
* Balancing der Agenten (Hilfsbereitschaft vs. Behinderung des Spielers).
* **Performance lokaler LLMs:** Antwortzeiten können je nach Hardware variieren.
* **Ressourcenbedarf:** Lokale LLMs benötigen ausreichend RAM.
* **Gedächtnis-Implementierung:** Aktuell sehr einfach, Verbesserungspotenzial (z.B. LangMem).
* Gestaltung einer interessanten Spielwelt und sinnvoller Rätsel/Aufgaben.

## ✨ Zukünftige Erweiterungsideen

* **Persistenz:** Speichern und Laden des Spielstands.
* **Besseres Gedächtnis:** Integration von LangMem o.ä. für echtes Langzeitgedächtnis der Agenten.
* **Fine-Tuning:** Untersuchung von Fine-Tuning lokaler Modelle für spezifisches Agentenverhalten.
* Implementierung von Charakter-Stats.
* Hinzufügen weiterer KI-Agenten, Items, Räume und Rätsel.
* Ausbau der Story und der Welt.
* (Optional) Discord Bot-Integration.
* (Optional) Einfache grafische Repräsentation.

## 🤝 Beitragende

Aktuell ein kollaboratives Projekt von [Dein Name/Handle] und Gemini. Ideen und Feedback sind willkommen! (Anpassbar für GitHub)

## 📄 Lizenz

(Platzhalter - z.B. MIT License oder "License TBD")