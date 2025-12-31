"""
Data Schema for TinyGreet Dataset

Each conversation sample contains:
- input:  The user's greeting/farewell
- output: The appropriate response
- metadata: Context information
"""

SCHEMA = {
    "sample":  {
        "id": "unique_string",
        "input":  "user input text",
        "output": "model response text",
        "metadata": {
            "category": "greeting | farewell | response",
            "subcategory": "time_based | formality | situation | mood",
            "formality": "formal | neutral | casual",
            "time_of_day":  "morning | afternoon | evening | night | any",
            "situation": "first_meeting | returning | professional | casual | any",
            "mood": "enthusiastic | neutral | tired | any",
            "expected_response_mood": "enthusiastic | neutral | calm | any"
        }
    }
}

# Example samples:
EXAMPLES = [
    {
        "id": "greet_morning_formal_001",
        "input": "Good morning, how are you today?",
        "output": "Good morning!  I'm doing very well, thank you for asking. How are you? ",
        "metadata": {
            "category": "greeting",
            "subcategory": "time_based",
            "formality": "formal",
            "time_of_day": "morning",
            "situation": "any",
            "mood": "neutral",
            "expected_response_mood":  "enthusiastic"
        }
    },
    {
        "id": "farewell_casual_weekend_001",
        "input": "Alright, catch you later!  Have a good weekend!",
        "output": "You too! Have an awesome weekend! See ya Monday!",
        "metadata": {
            "category": "farewell",
            "subcategory": "time_based",
            "formality":  "casual",
            "time_of_day": "any",
            "situation": "work",
            "mood":  "enthusiastic",
            "expected_response_mood": "enthusiastic"
        }
    }
]