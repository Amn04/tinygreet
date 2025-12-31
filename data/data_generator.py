"""
Systematic Data Generator for TinyGreet

This is how real-world specialized LLMs expand their datasets:
- Template-based generation
- Combinatorial expansion
- Controlled variation

This will take us from 234 → 2000+ samples
"""

import json
import random
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from itertools import product


class GreetingDataGenerator:
    """
    Generate diverse greeting/farewell data through systematic expansion. 
    
    This mimics how companies like OpenAI, Anthropic create training data
    for specialized tasks - through careful template design and expansion.
    """
    
    def __init__(self, seed:  int = 42):
        self.random = random.Random(seed)
        self.generated_data:  List[Dict] = []
        
        # ==================== BUILDING BLOCKS ====================
        
        # Names for personalization
        self. names = [
            "Alex", "Jordan", "Sam", "Taylor", "Casey", "Morgan", "Riley",
            "Jamie", "Drew", "Avery", "Quinn", "Blake", "Charlie", "Dakota",
            "Elliot", "Finley", "Gray", "Harper", "Indigo", "Jesse",
            "friend", "everyone", "all", "team", "folks", "guys", "people"
        ]
        
        # Time periods
        self.time_contexts = {
            "morning":  {
                "time_range": "6am-12pm",
                "greetings": ["Good morning", "Morning", "Top of the morning"],
                "casual_greetings": ["Mornin'", "Morning!", "G'morning"],
                "responses_enthusiastic": [
                    "Good morning!  Ready to take on the day? ",
                    "Morning!  Hope you slept well! ",
                    "Good morning! It's going to be a great day!",
                    "Morning! Coffee time! ☕",
                ],
                "responses_calm": [
                    "Good morning.  I hope you're well.",
                    "Morning. How did you sleep?",
                    "Good morning.  Ready for the day ahead? ",
                ],
                "responses_tired": [
                    "Morning...  *yawn* Need coffee first.. .",
                    "Ugh, morning already? Hey.. .",
                    "Morning... is it too early to go back to bed?",
                ]
            },
            "afternoon": {
                "time_range": "12pm-5pm",
                "greetings": ["Good afternoon", "Afternoon"],
                "casual_greetings": ["Afternoon!", "Hey, afternoon!"],
                "responses_enthusiastic":  [
                    "Good afternoon! How's your day going?",
                    "Afternoon! Having a productive day?",
                    "Good afternoon! Hope your day is going well! ",
                ],
                "responses_calm": [
                    "Good afternoon.  How has your day been?",
                    "Afternoon. Everything going smoothly?",
                ],
                "responses_tired": [
                    "Afternoon... hitting that post-lunch slump.. .",
                    "Hey afternoon...  counting down to 5pm...",
                ]
            },
            "evening": {
                "time_range": "5pm-9pm",
                "greetings": ["Good evening", "Evening"],
                "casual_greetings":  ["Evening!", "Hey, good evening!"],
                "responses_enthusiastic": [
                    "Good evening! How was your day?",
                    "Evening!  Ready to relax? ",
                    "Good evening! Any plans for tonight?",
                ],
                "responses_calm":  [
                    "Good evening. I hope you had a pleasant day.",
                    "Evening. Time to unwind.",
                ],
                "responses_tired":  [
                    "Evening... what a long day...",
                    "Hey...  finally evening.  So tired.",
                ]
            },
            "night":  {
                "time_range": "9pm-6am",
                "greetings": ["Good night", "Night", "Goodnight"],
                "casual_greetings": ["Night!", "G'night", "Nighty night"],
                "responses_enthusiastic": [
                    "Good night! Sweet dreams!",
                    "Night night! Sleep tight!",
                    "Good night! See you tomorrow!",
                ],
                "responses_calm": [
                    "Good night. Rest well.",
                    "Night. See you tomorrow.",
                    "Sleep well. Good night.",
                ],
                "responses_tired":  [
                    "Night... finally...  zzz",
                    "Good night... so ready for sleep...",
                ]
            }
        }
        
        # Generic greetings (no time context)
        self.generic_greetings = {
            "formal": {
                "inputs": [
                    "Hello.",
                    "Greetings.",
                    "How do you do?",
                    "Good day.",
                    "Salutations.",
                    "Hello there.",
                    "It's a pleasure to see you.",
                ],
                "responses": [
                    "Hello! How may I assist you today?",
                    "Greetings! It's wonderful to meet you.",
                    "How do you do? I'm doing quite well, thank you.",
                    "Good day to you as well!",
                    "Hello! The pleasure is mine.",
                    "Greetings! How can I be of service?",
                ]
            },
            "neutral": {
                "inputs": [
                    "Hello",
                    "Hi",
                    "Hey",
                    "Hi there",
                    "Hello there",
                ],
                "responses":  [
                    "Hello! How are you? ",
                    "Hi! What's going on?",
                    "Hey!  Good to see you! ",
                    "Hi there! How can I help?",
                    "Hello!  Nice to hear from you!",
                ]
            },
            "casual": {
                "inputs": [
                    "Hey! ",
                    "Yo!",
                    "Sup? ",
                    "What's up?",
                    "Howdy! ",
                    "Hiya!",
                    "Heyo!",
                    "Wassup? ",
                    "What's good?",
                    "Yo yo yo! ",
                ],
                "responses":  [
                    "Hey! What's up?",
                    "Yo! How's it going?",
                    "Not much, you?  What's happening?",
                    "Hey hey!  What's going on?",
                    "Howdy! How ya doing?",
                    "Hiya! Great to see you!",
                    "Sup! Just chilling, you? ",
                    "What's good!  Everything's cool here!",
                ]
            }
        }
        
        # Farewells
        self.farewells = {
            "formal": {
                "inputs": [
                    "Goodbye.",
                    "Farewell.",
                    "It was a pleasure.",
                    "I must be going now.",
                    "I shall take my leave.",
                    "Until we meet again.",
                    "I bid you farewell.",
                ],
                "responses":  [
                    "Goodbye! It was wonderful speaking with you.",
                    "Farewell! Until next time.",
                    "The pleasure was all mine.  Take care.",
                    "Of course.  It was lovely seeing you.",
                    "Goodbye! I look forward to our next meeting.",
                    "Until then!  Be well.",
                ]
            },
            "neutral": {
                "inputs": [
                    "Bye",
                    "Goodbye",
                    "See you",
                    "Take care",
                    "See you later",
                    "Until next time",
                ],
                "responses":  [
                    "Bye! Take care!",
                    "Goodbye!  See you soon!",
                    "See you!  Have a great day!",
                    "You too! Take care!",
                    "See you later! Stay well!",
                    "Until next time! Be good!",
                ]
            },
            "casual": {
                "inputs": [
                    "Bye! ",
                    "Later!",
                    "Peace! ",
                    "See ya!",
                    "Catch ya later!",
                    "Gotta go!",
                    "I'm out!",
                    "Deuces!",
                    "Laters!",
                    "Ciao!",
                    "Toodles!",
                    "Bouncing! ",
                ],
                "responses":  [
                    "Bye! Talk soon!",
                    "Later! Stay cool!",
                    "Peace out! ",
                    "See ya! Take it easy!",
                    "Catch ya!  Don't be a stranger!",
                    "Alright, bye! Hit me up later!",
                    "Aight, peace! ",
                    "Deuces! Stay safe!",
                    "Ciao! Until next time!",
                    "Toodles! Bye bye! ",
                ]
            }
        }
        
        # Time-specific farewells
        self.time_farewells = {
            "day":  [
                ("Have a good day!", "Thanks, you too!  Enjoy your day!"),
                ("Have a great day!", "You too! Make it a good one!"),
                ("Enjoy your day!", "Thanks!  You as well!"),
            ],
            "evening": [
                ("Have a good evening!", "Thank you!  Enjoy your evening! "),
                ("Have a nice evening!", "You too! Have a relaxing night!"),
                ("Enjoy your evening!", "Thanks! You too! "),
            ],
            "weekend": [
                ("Have a great weekend!", "Thanks!  You too! Any fun plans?"),
                ("Enjoy your weekend!", "You too! Have a good one!"),
                ("Have a good weekend!", "Thanks!  Enjoy yours!"),
                ("Happy weekend!", "Happy weekend to you too!"),
            ],
            "week": [
                ("Have a good week!", "Thanks! You too! "),
                ("Have a great week ahead!", "You as well! Take care!"),
            ]
        }
        
        # How are you variations
        self.how_are_you = {
            "formal": {
                "inputs": [
                    "How are you?",
                    "How are you doing?",
                    "How have you been?",
                    "I trust you are well? ",
                    "How do you fare?",
                    "Are you well?",
                ],
                "responses":  [
                    "I'm doing very well, thank you for asking.  And yourself?",
                    "Quite well, thank you!  How are you? ",
                    "I've been well, thank you for your concern. And you?",
                    "Yes, quite well, thank you! I hope you are too.",
                    "I'm doing splendidly! How about yourself?",
                ]
            },
            "neutral": {
                "inputs": [
                    "How are you?",
                    "How's it going?",
                    "How are things?",
                    "How have you been?",
                    "Everything okay?",
                ],
                "responses":  [
                    "I'm good, thanks!  How about you?",
                    "Pretty good!  What's new with you?",
                    "Things are going well!  You?",
                    "Been good!  And yourself?",
                    "All good here! How are you? ",
                ]
            },
            "casual": {
                "inputs": [
                    "How's it going?",
                    "What's up?",
                    "How ya doing?",
                    "Whatcha up to?",
                    "How's life?",
                    "What's new?",
                    "How's everything?",
                    "You good?",
                    "Alright? ",
                ],
                "responses":  [
                    "Going good! You?",
                    "Not much!  What about you?",
                    "Doing alright!  Yourself?",
                    "Just chilling! You?",
                    "Life's good! Can't complain!",
                    "Same old, same old! You?",
                    "Everything's cool! What's up with you?",
                    "Yeah I'm good! You?",
                    "Yeah alright! You good?",
                ]
            }
        }
        
        # Thank you / appreciation
        self.thanks = {
            "formal": {
                "inputs": [
                    "Thank you.",
                    "Thank you very much.",
                    "I appreciate it.",
                    "I'm grateful for your help.",
                    "Many thanks.",
                    "Thank you kindly.",
                ],
                "responses":  [
                    "You're most welcome!",
                    "It was my pleasure.",
                    "Of course! Happy to help.",
                    "The pleasure is mine.",
                    "You're very welcome!  Anytime.",
                ]
            },
            "casual": {
                "inputs": [
                    "Thanks!",
                    "Thanks a lot!",
                    "Thanks so much!",
                    "Thx!",
                    "Ty!",
                    "Thanks dude!",
                    "Thanks man!",
                    "Appreciate it!",
                    "Thanks a bunch!",
                    "You're awesome, thanks!",
                ],
                "responses":  [
                    "No problem!",
                    "You got it!",
                    "Anytime! ",
                    "No worries!",
                    "Sure thing!",
                    "Happy to help!",
                    "You bet!",
                    "Of course!",
                    "Don't mention it!",
                    "Glad I could help!",
                ]
            }
        }
        
        # Apologies
        self. apologies = {
            "formal": {
                "inputs":  [
                    "I apologize.",
                    "My apologies.",
                    "I'm sorry for the inconvenience.",
                    "Please forgive me.",
                    "I apologize for any trouble.",
                ],
                "responses":  [
                    "No apology necessary.",
                    "It's quite alright.",
                    "Please, don't worry about it.",
                    "There's nothing to forgive.",
                    "No trouble at all.",
                ]
            },
            "casual": {
                "inputs": [
                    "Sorry!",
                    "My bad!",
                    "Oops, sorry!",
                    "Sorry about that!",
                    "Ah sorry!",
                    "Whoops, sorry!",
                ],
                "responses":  [
                    "No worries!",
                    "It's all good!",
                    "Don't sweat it!",
                    "No problem at all!",
                    "Happens to the best of us!",
                    "All good, no stress!",
                ]
            }
        }
        
        # Conversation continuers / acknowledgments
        self. acknowledgments = {
            "inputs": [
                "Okay", "Ok", "OK", "K",
                "Alright", "Aight",
                "Sure", "Sure thing",
                "Got it", "Gotcha",
                "I see", "I understand",
                "Right", "Yep", "Yeah", "Yes",
                "Uh huh", "Mhm",
                "Cool", "Nice", "Great", "Awesome",
            ],
            "responses": [
                "Great!  Let me know if you need anything else.",
                "Perfect! I'm here if you have more questions.",
                "Alright!  Anything else I can help with?",
                "Sounds good! ",
                "Okay!  Just let me know! ",
                "Got it! I'm here when you need me.",
                "Awesome! Happy to help anytime!",
            ]
        }
        
        # Punctuation variations for augmentation
        self. punctuation_variations = [
            ("!", ""),      # Remove exclamation
            ("!", ". "),     # Replace with period
            (".", "!"),     # Add enthusiasm
            (".", ""),      # Remove period
            ("? ", ""),      # Remove question mark (for casual)
        ]
        
        # Case variations
        self. case_transforms = [
            lambda x: x,                    # Original
            lambda x: x.lower(),            # Lowercase
            lambda x: x. upper(),            # Uppercase (for enthusiasm)
            lambda x: x.capitalize(),       # Capitalize first only
        ]
    
    def _create_sample(
        self,
        input_text: str,
        output_text: str,
        category: str,
        subcategory: str,
        formality: str,
        time_of_day: str = "any",
        mood: str = "neutral",
        source: str = "generated"
    ) -> Dict:
        """Create a standardized sample dictionary."""
        return {
            "input": input_text. strip(),
            "output": output_text. strip(),
            "category": category,
            "subcategory": subcategory,
            "formality": formality,
            "time_of_day": time_of_day,
            "mood": mood,
            "source": source
        }
    
    def _add_name_variations(self, greeting: str, response: str) -> List[Tuple[str, str]]:
        """Add name variations to greetings."""
        variations = [(greeting, response)]
        
        # Only add names to some greetings
        if any(g in greeting.lower() for g in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
            for name in self.random.sample(self. names, min(5, len(self. names))):
                if greeting.endswith(("!", ".", "?")):
                    new_greeting = greeting[:-1] + f" {name}" + greeting[-1]
                else:
                    new_greeting = f"{greeting} {name}"
                
                # Sometimes add name to response too
                if self.random.random() < 0.3:
                    new_response = f"Hi {name}! " + response
                else:
                    new_response = response
                    
                variations.append((new_greeting, new_response))
        
        return variations
    
    def _add_punctuation_variations(self, text: str, is_casual: bool) -> List[str]:
        """Generate punctuation variations."""
        variations = [text]
        
        if is_casual: 
            for old, new in self.punctuation_variations: 
                if old in text:
                    variations.append(text.replace(old, new, 1))
        
        return list(set(variations))  # Remove duplicates
    
    def generate_time_based_greetings(self) -> List[Dict]:
        """Generate all time-based greetings."""
        samples = []
        
        for time_period, data in self.time_contexts.items():
            # Formal greetings
            for greeting in data["greetings"]:
                for response in data["responses_calm"]:
                    sample = self._create_sample(
                        greeting, response,
                        category="greeting",
                        subcategory="time_based",
                        formality="formal",
                        time_of_day=time_period,
                        mood="calm"
                    )
                    samples.append(sample)
            
            # Casual greetings with different moods
            for greeting in data["casual_greetings"]:
                # Enthusiastic responses
                for response in data["responses_enthusiastic"]:
                    sample = self._create_sample(
                        greeting, response,
                        category="greeting",
                        subcategory="time_based",
                        formality="casual",
                        time_of_day=time_period,
                        mood="enthusiastic"
                    )
                    samples.append(sample)
                
                # Tired responses
                for response in data. get("responses_tired", []):
                    sample = self._create_sample(
                        greeting, response,
                        category="greeting",
                        subcategory="time_based",
                        formality="casual",
                        time_of_day=time_period,
                        mood="tired"
                    )
                    samples.append(sample)
            
            # Add name variations for some samples
            for greeting in data["greetings"][:2]: 
                for name in self.random. sample(self.names[: 10], 3):
                    named_greeting = f"{greeting}, {name}!"
                    response = self. random.choice(data["responses_enthusiastic"])
                    sample = self._create_sample(
                        named_greeting, response,
                        category="greeting",
                        subcategory="time_based",
                        formality="neutral",
                        time_of_day=time_period,
                        mood="enthusiastic"
                    )
                    samples.append(sample)
        
        return samples
    
    def generate_generic_greetings(self) -> List[Dict]:
        """Generate generic (non-time-based) greetings."""
        samples = []
        
        for formality, data in self.generic_greetings. items():
            for input_text in data["inputs"]: 
                for response in data["responses"]: 
                    sample = self._create_sample(
                        input_text, response,
                        category="greeting",
                        subcategory="generic",
                        formality=formality
                    )
                    samples.append(sample)
                
                # Add punctuation variations for casual
                if formality == "casual":
                    for variation in self._add_punctuation_variations(input_text, True):
                        if variation != input_text: 
                            sample = self._create_sample(
                                variation,
                                self.random.choice(data["responses"]),
                                category="greeting",
                                subcategory="generic",
                                formality=formality
                            )
                            samples.append(sample)
        
        return samples
    
    def generate_farewells(self) -> List[Dict]: 
        """Generate farewell data."""
        samples = []
        
        # Standard farewells by formality
        for formality, data in self.farewells.items():
            for input_text in data["inputs"]: 
                for response in data["responses"]: 
                    sample = self._create_sample(
                        input_text, response,
                        category="farewell",
                        subcategory="generic",
                        formality=formality
                    )
                    samples.append(sample)
        
        # Time-specific farewells
        for time_type, pairs in self.time_farewells.items():
            for input_text, response in pairs:
                sample = self._create_sample(
                    input_text, response,
                    category="farewell",
                    subcategory="time_based",
                    formality="neutral",
                    time_of_day=time_type
                )
                samples.append(sample)
                
                # Add casual variations
                casual_input = input_text.lower().replace("have a ", "").replace("!", "")
                if casual_input != input_text. lower():
                    sample = self._create_sample(
                        casual_input. capitalize() + "! ",
                        response,
                        category="farewell",
                        subcategory="time_based",
                        formality="casual",
                        time_of_day=time_type
                    )
                    samples.append(sample)
        
        return samples
    
    def generate_how_are_you(self) -> List[Dict]:
        """Generate 'how are you' variations."""
        samples = []
        
        for formality, data in self.how_are_you. items():
            for input_text in data["inputs"]: 
                for response in data["responses"]: 
                    sample = self._create_sample(
                        input_text, response,
                        category="greeting",
                        subcategory="how_are_you",
                        formality=formality
                    )
                    samples.append(sample)
        
        return samples
    
    def generate_thanks_and_apologies(self) -> List[Dict]:
        """Generate thank you and apology responses."""
        samples = []
        
        # Thanks
        for formality, data in self. thanks.items():
            for input_text in data["inputs"]:
                for response in data["responses"]:
                    sample = self._create_sample(
                        input_text, response,
                        category="response",
                        subcategory="thanks",
                        formality=formality
                    )
                    samples.append(sample)
        
        # Apologies
        for formality, data in self.apologies.items():
            for input_text in data["inputs"]:
                for response in data["responses"]:
                    sample = self._create_sample(
                        input_text, response,
                        category="response",
                        subcategory="apology",
                        formality=formality
                    )
                    samples.append(sample)
        
        return samples
    
    def generate_acknowledgments(self) -> List[Dict]: 
        """Generate acknowledgment responses."""
        samples = []
        
        for input_text in self. acknowledgments["inputs"]:
            for response in self.acknowledgments["responses"]: 
                sample = self._create_sample(
                    input_text, response,
                    category="response",
                    subcategory="acknowledgment",
                    formality="casual" if len(input_text) < 4 else "neutral"
                )
                samples.append(sample)
        
        return samples
    
    def generate_compound_greetings(self) -> List[Dict]:
        """Generate compound greetings (greeting + how are you)."""
        samples = []
        
        compound_patterns = [
            ("Hi!  How are you?", "Hi! I'm doing great, thanks for asking!  How about you?"),
            ("Hey!  What's up?", "Hey! Not much, just chilling.  You?"),
            ("Hello! How have you been?", "Hello! I've been well, thank you!  And yourself?"),
            ("Good morning! How are you today?", "Good morning! I'm doing wonderfully, thanks!  How are you?"),
            ("Hey there! How's it going?", "Hey!  Going pretty good!  What's new with you?"),
            ("Hi! Long time no see!", "Hi! I know, right? It's been too long!  How have you been? "),
            ("Hello! Nice to see you again!", "Hello! Great to see you too! How have things been?"),
            ("Hey! How's life treating you?", "Hey! Can't complain!  Life's good. How about you?"),
            ("Good afternoon! How's your day going?", "Good afternoon! It's going well, thanks!  Yours?"),
            ("Hi there! Everything okay?", "Hi! Yes, everything's great! Thanks for asking! "),
        ]
        
        for input_text, response in compound_patterns: 
            sample = self._create_sample(
                input_text, response,
                category="greeting",
                subcategory="compound",
                formality="neutral"
            )
            samples.append(sample)
        
        return samples
    
    def generate_contextual_greetings(self) -> List[Dict]:
        """Generate greetings with situational context."""
        samples = []
        
        contexts = [
            # Work context
            ("Hey!  Ready for the meeting?", "Hey!  Yep, all prepared.  See you there!"),
            ("Good morning! Big day today!", "Morning! I know, wish me luck! "),
            ("Hi! How was your weekend?", "Hi!  It was great, thanks!  How was yours?"),
            ("Hey!  TGIF!", "Hey! Finally Friday! Any weekend plans?"),
            ("Good morning! Happy Monday!", "Morning! Happy Monday...  if that's a thing, haha!"),
            
            # Return from absence
            ("Welcome back!", "Thanks!  Good to be back!  Did I miss anything?"),
            ("Hey stranger!  Where have you been?", "Hey! I know, I've been so busy.  Good to catch up!"),
            ("Look who's here!", "Haha, I'm back! Miss me? "),
            
            # First meeting
            ("Hi!  I don't think we've met.  I'm Alex.", "Hi Alex! Nice to meet you!  I'm the assistant here."),
            ("Hello, I'm new here.", "Hello!  Welcome!  Let me know if you need any help settling in."),
            
            # Casual catch-up
            ("Hey! Haven't seen you in ages!", "I know!  We should catch up properly soon!"),
            ("Hi! What have you been up to? ", "Hi! Oh, you know, the usual. Staying busy!"),
        ]
        
        for input_text, response in contexts: 
            formality = "casual" if any(c in input_text. lower() for c in ["hey", "hi!", "tgif"]) else "neutral"
            sample = self._create_sample(
                input_text, response,
                category="greeting",
                subcategory="contextual",
                formality=formality
            )
            samples.append(sample)
        
        return samples
    
    def generate_all(self) -> List[Dict]:
        """Generate all data."""
        all_samples = []
        
        print("Generating time-based greetings...")
        all_samples.extend(self.generate_time_based_greetings())
        
        print("Generating generic greetings...")
        all_samples.extend(self. generate_generic_greetings())
        
        print("Generating farewells...")
        all_samples.extend(self. generate_farewells())
        
        print("Generating 'how are you' variations...")
        all_samples.extend(self.generate_how_are_you())
        
        print("Generating thanks and apologies...")
        all_samples.extend(self.generate_thanks_and_apologies())
        
        print("Generating acknowledgments...")
        all_samples.extend(self.generate_acknowledgments())
        
        print("Generating compound greetings...")
        all_samples.extend(self.generate_compound_greetings())
        
        print("Generating contextual greetings...")
        all_samples.extend(self.generate_contextual_greetings())
        
        # Add unique IDs
        for i, sample in enumerate(all_samples):
            sample["id"] = f"gen_{i: 05d}"
        
        # Shuffle
        self.random.shuffle(all_samples)
        
        self.generated_data = all_samples
        return all_samples
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total":  len(self.generated_data),
            "by_category": defaultdict(int),
            "by_subcategory": defaultdict(int),
            "by_formality": defaultdict(int),
            "by_time":  defaultdict(int),
            "by_mood": defaultdict(int),
        }
        
        for sample in self. generated_data: 
            stats["by_category"][sample["category"]] += 1
            stats["by_subcategory"][sample["subcategory"]] += 1
            stats["by_formality"][sample["formality"]] += 1
            stats["by_time"][sample. get("time_of_day", "any")] += 1
            stats["by_mood"][sample. get("mood", "neutral")] += 1
        
        # Calculate text statistics
        input_lengths = [len(s["input"]) for s in self.generated_data]
        output_lengths = [len(s["output"]) for s in self.generated_data]
        
        stats["avg_input_length"] = sum(input_lengths) / len(input_lengths)
        stats["avg_output_length"] = sum(output_lengths) / len(output_lengths)
        stats["max_input_length"] = max(input_lengths)
        stats["max_output_length"] = max(output_lengths)
        
        return stats
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("GENERATED DATASET STATISTICS")
        print("=" * 60)
        
        print(f"\n📊 Total samples:  {stats['total']}")
        
        print("\n📁 By Category:")
        for cat, count in sorted(stats["by_category"].items()):
            pct = count / stats["total"] * 100
            bar = "█" * int(pct / 2)
            print(f"   {cat:<15} {count:4} ({pct:5.1f}%) {bar}")
        
        print("\n📂 By Subcategory:")
        for subcat, count in sorted(stats["by_subcategory"].items()):
            print(f"   {subcat:20} {count:4}")
        
        print("\n🎭 By Formality:")
        for form, count in sorted(stats["by_formality"].items()):
            pct = count / stats["total"] * 100
            bar = "█" * int(pct / 2)
            print(f"   {form:15} {count:4} ({pct:5.1f}%) {bar}")
        
        print("\n⏰ By Time of Day:")
        for time, count in sorted(stats["by_time"].items()):
            print(f"   {time:15} {count:4}")
        
        print("\n😊 By Mood:")
        for mood, count in sorted(stats["by_mood"].items()):
            print(f"   {mood:15} {count:4}")
        
        print(f"\n📏 Text Lengths:")
        print(f"   Avg input:   {stats['avg_input_length']:.1f} chars")
        print(f"   Avg output: {stats['avg_output_length']:.1f} chars")
        print(f"   Max input:  {stats['max_input_length']} chars")
        print(f"   Max output: {stats['max_output_length']} chars")
    
    def save(self, filepath: str):
        """Save generated data to JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.generated_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved {len(self.generated_data)} samples to {filepath}")


def main():
    """Main generation pipeline."""
    print("=" * 60)
    print("TinyGreet Data Generator")
    print("=" * 60)
    
    generator = GreetingDataGenerator(seed=42)
    
    # Generate all data
    data = generator.generate_all()
    
    # Print statistics
    generator. print_statistics()
    
    # Save to file
    import os
    os. makedirs("processed", exist_ok=True)
    generator.save("processed/generated_data.json")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE DATA (5 random examples)")
    print("=" * 60)
    
    for sample in generator.random.sample(data, 5):
        print(f"\n[{sample['category']}/{sample['formality']}]")
        print(f"   Input:  {sample['input']}")
        print(f"   Output: {sample['output']}")


if __name__ == "__main__": 
    main()