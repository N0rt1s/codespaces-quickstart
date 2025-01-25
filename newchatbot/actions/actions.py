import json
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_md")

# Load the data
data_file = "./eye_diseases_symptoms.xlsx"
data = pd.read_excel(data_file).to_dict(orient="records")

class ActionFindDisease(Action):
    def name(self) -> Text:
        return "action_find_disease"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get the latest user input
        user_input = tracker.latest_message.get("text", "").lower()

        # Process the user input through the NLP model
        user_input_doc = nlp(user_input)

        # Retrieve the current symptoms stored in the slot
        current_symptoms = tracker.get_slot("symptoms") or []
        current_symptoms.append(user_input)

        # Find matching diseases based on semantic similarity
        matches = []
        for d in data:
            # Tokenize the symptoms and compare each symptom
            symptoms = [symptom.strip().lower() for symptom in d["Symptoms"].split(",")]
            for symptom in symptoms:
                symptom_doc = nlp(symptom)
                similarity = nlp(",".join(current_symptoms)).similarity(symptom_doc)

                if similarity > 0.7:
                    matches.append(d)
                    break  # No need to check other symptoms for this disease

        # If multiple diseases are found, ask for more symptoms
        if len(matches) > 1:
            print(matches)
            dispatcher.utter_message(text="There are multiple diseases that might match your symptoms. Can you tell me more symptoms?")
            return [SlotSet("symptoms", current_symptoms)]

        # If one disease is found, provide information about it
        if len(matches) == 1:
            disease_info = matches[0]
            dispatcher.utter_message(text=f"Based on the symptoms you've described, it seems you may have {disease_info['Disorder']}. Here are some details:")
            dispatcher.utter_message(text=f"Causes: {disease_info['Causes']}")
            dispatcher.utter_message(text=f"Prevention: {disease_info.get('Prevention', 'No prevention data available')}")
            dispatcher.utter_message(text=f"Treatment: {disease_info['Treatement']}")

            # Reset the symptoms slot for the next interaction
            return [SlotSet("symptoms", [])]

        # If no match is found, ask for more details
        dispatcher.utter_message(text="I couldn't find a disease matching your symptoms. Please provide more details.")
        return [SlotSet("symptoms", current_symptoms)]
    
class ActionFindCause(Action):
    def name(self) -> Text:
        return "action_find_cause"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text", "").lower()
        for d in data:
            if d["Disorder"].lower() in user_input:
                dispatcher.utter_message(text=f"Causes of {d['Disorder']}:\n{d['Causes']}")
                return []
        dispatcher.utter_message(text="I couldn't find information on that disease.")
        return []

class ActionFindPrevention(Action):
    def name(self) -> Text:
        return "action_find_prevention"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text", "").lower()
        for d in data:
            if d["Disorder"].lower() in user_input:
                if pd.isna(d["Prevention"]):
                    dispatcher.utter_message(text=f"No prevention data available for {d['Disorder']}.")
                else:
                    dispatcher.utter_message(text=f"Prevention for {d['Disorder']}:\n{d['Prevention']}")
                return []
        dispatcher.utter_message(text="I couldn't find prevention information for that disease.")
        return []

class ActionFindTreatment(Action):
    def name(self) -> Text:
        return "action_find_treatment"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text", "").lower()
        for d in data:
            if d["Disorder"].lower() in user_input:
                dispatcher.utter_message(text=f"Treatment for {d['Disorder']}:\n{d['Treatement']}")
                return []
        dispatcher.utter_message(text="I couldn't find treatment information for that disease.")
        return []
