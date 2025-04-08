# Project: Whispering Towers (Arbeitstitel)

## 📜 Konzept

**Whispering Towers** ist ein experimentelles Textadventure, das klassische Erkundung und Rätsellösung mit dynamischer Interaktion durch KI-gesteuerte Charaktere (Agenten) verbindet. Spieler navigieren durch eine Welt bestehend aus miteinander verbundenen Orten ("Räumen"), interagieren mit Objekten und verfolgen ein übergeordnetes Ziel.

Das Besondere an diesem Projekt ist der Einsatz von Sprachmodellen (LLMs), um NSCs (Nicht-Spieler-Charaktere) zum Leben zu erwecken. Diese Agenten besitzen:

* **Eigene Persönlichkeiten und Agenden:** Sie verfolgen eigene Ziele, die nicht immer mit denen des Spielers übereinstimmen.
* **Kontextuelles Gedächtnis:** Sie erinnern sich an vergangene Interaktionen mit dem Spieler.
* **Natürliche Sprachverarbeitung:** Spieler können über Texteingaben in natürlicher Sprache mit ihnen kommunizieren, über vordefinierte Befehle hinaus.
* **Dynamisches Verhalten:** Ihre Reaktionen und ihre Hilfsbereitschaft (oder Feindseligkeit) entwickeln sich basierend auf dem Verhalten des Spielers.

Das Spiel beinhaltet Roguelike-Elemente, die Wiederspielbarkeit fördern und bei bestimmten Ereignissen oder Entscheidungen zu einem Neustart des Spielzyklus führen können. Ein einzelner Durchlauf ist auf eine Spielzeit von ca. 10-30 Minuten ausgelegt.

**Zielgruppe:** Spieler von Textadventures, Fans von interaktiver Fiktion und Personen, die an der Anwendung von KI in Spielen interessiert sind.

## 🏗️ Geplante Architektur & Technischer Ansatz

Das Spiel wird modular aufgebaut sein, um Erweiterbarkeit und Wartbarkeit zu gewährleisten.

1.  **Game Engine Core (Python):**
    * Verwaltet den Spielzustand (Position des Spielers, Inventar, Zustand der Welt/Räume, Flags für Ereignisse).
    * Implementiert die grundlegende Spiel-Loop (Input -> Processing -> Output).
    * Definiert die Struktur der Welt (Räume, Verbindungen, Objekte).
    * Speichert und lädt Spielstände (insbesondere für Agenten-Gedächtnis relevant).

2.  **Input Parser:**
    * Unterscheidet zwischen Standard-Textadventure-Befehlen (z.B. `gehe nach norden`, `nimm schlüssel`, `untersuche tisch`) und natürlicher Sprache für die Interaktion mit Agenten.
    * Leitet Standardbefehle an die Game Engine weiter.
    * Leitet natürliche Spracheingaben an das entsprechende Agenten-Modul weiter.

3.  **World Representation:**
    * Definition von Räumen, Objekten und deren Eigenschaften (beschreibbar, nehmbar, benutzbar etc.).
    * Wahrscheinlich über Konfigurationsdateien (z.B. JSON oder YAML) oder direkt in Python-Datenstrukturen.

4.  **Agenten-Modul (KI-Integration):**
    * **Herzstück des Projekts.** Für jeden KI-Agenten wird eine Instanz verwaltet.
    * **Prompt Engineering:** Jeder Agent erhält einen Basis-Prompt, der seine Persönlichkeit, seine geheime Agenda, sein Wissen und den aktuellen Spielkontext definiert.
    * **Gedächtnis-Management:**
        * (+) **Nutzung von LangMem (oder vergleichbarer Memory-Bibliothek):** Um das Problem begrenzter LLM-Kontextfenster zu umgehen und ein persistentes Gedächtnis für die Agenten zu ermöglichen.
        * (+) LangMem soll vergangene Interaktionen mit dem Spieler speichern.
        * (+) Relevante Informationen aus der gespeicherten Historie werden bei neuen Interaktionen abgerufen (z.B. durch semantische Suche oder Zusammenfassung) und dem Agenten als Teil des Kontexts zur Verfügung gestellt.
        * Dies ermöglicht es Agenten, sich an Details aus früheren Gesprächen zu "erinnern" und ihr Verhalten entsprechend anzupassen.
    * **LLM-Anbindung:** Schnittstelle zu einem oder mehreren Sprachmodellen (lokal oder API-basiert). Hier wird die natürliche Spracheingabe des Spielers zusammen mit dem Kontext-Prompt (inkl. abgerufener Gedächtnisinhalte) verarbeitet, um die Antwort des Agenten zu generieren.
    * **(Optional) Fine-Tuning:** Exploration von Fine-Tuning-Techniken, um das Verhalten der Agenten spezifischer zu gestalten.


5.  **State Management:**
    * Speicherung des relevanten Spielzustands, insbesondere des Agenten-Gedächtnisses über Spielneustarts hinweg (je nach Designentscheidung für den Reset-Mechanismus).

6.  **(Optional) Deployment-Schnittstelle (z.B. Discord Bot):**
    * Ein Wrapper, der die Game Engine startet und die Ein-/Ausgabe über eine Plattform wie Discord ermöglicht (z.B. mittels `discord.py` oder `hikari`).

## 🚀 Technologie-Stack (Vorschlag)

* **Programmiersprache:** Python (starke Bibliotheken für Textverarbeitung, KI und ggf. Discord-Bots)
* **LLM-Integration:**
    * Bibliotheken wie `transformers` (Hugging Face), `LangChain` oder `LlamaIndex`.
    * Potenzielle Nutzung von lokalen Modellen (via Ollama, llama.cpp etc.) und/oder APIs (OpenAI, Anthropic, Google Gemini etc.).
* (+) **Agenten-Gedächtnis:** LangMem (oder alternative Memory-Lösungen innerhalb von LangChain/LlamaIndex)
* **Datenformate:** JSON oder YAML für Welt-/Objektdefinitionen.
* **Discord Bot (Optional):** `discord.py` oder `hikari`.

## 🎯 Ziele & Motivation

* Entwicklung eines spielbaren Prototyps mit mindestens 1-2 KI-Agenten.
* Erforschung der Möglichkeiten und Herausforderungen bei der Integration von LLMs in interaktive Fiktion.
* Schaffung eines unterhaltsamen Spielerlebnisses für Freunde und Interessierte.
* Aufbau eines flexiblen Frameworks, das potenziell für weitere Mini-Adventures wiederverwendet werden kann.
* Experimentieren mit Deployment-Optionen (insbesondere Discord).

## 💡 Herausforderungen

* Effektives Prompt-Design für glaubwürdige und konsistente Agenten-Persönlichkeiten und -Gedächtnisse.
* Balancing der Agenten (Hilfsbereitschaft vs. Behinderung des Spielers).
* (+) **Technische Umsetzung und Integration des Agenten-Gedächtnisses mit LangMem.**
* Performance und Kosten bei der Nutzung von LLMs (insbesondere bei lokalen Modellen oder API-Calls).
* (+) **Effiziente Abfrage und Zusammenfassung der Gedächtnisinhalte durch LangMem.**
* Gestaltung einer interessanten Spielwelt und sinnvoller Rätsel/Aufgaben.
* Integration und potenzielles Fine-Tuning von LLMs.

## ✨ Zukünftige Erweiterungsideen

* Implementierung von Charakter-Stats (z.B. Überzeugungskraft, Schleichen), die Interaktionsmöglichkeiten beeinflussen.
* Hinzufügen weiterer KI-Agenten mit komplexeren Beziehungen untereinander.
* Ausbau der Story und der Spielwelt.
* Dynamischere Welt, die sich auch ohne Spielerinteraktion verändert.
* (Optional) Einfache grafische Repräsentation der aktuellen Szene oder Karte.

## 🤝 Beitragende

Aktuell ein kollaboratives Projekt von [Dein Name/Handle] und Gemini. Ideen und Feedback sind willkommen! (Anpassbar für GitHub)

## 📄 Lizenz

(Platzhalter - z.B. MIT License oder "License TBD")
