"""
Seed Dataset for TinyGreet
This is our initial handcrafted dataset that we'll expand. 
"""

SEED_DATA = {
    # ==================== GREETINGS ====================
    
    # ----- Morning Greetings -----
    "morning_formal":  [
        {
            "input": "Good morning.",
            "output":  "Good morning! I hope you're having a wonderful start to your day.",
            "formality": "formal", "time_of_day": "morning", "mood": "neutral"
        },
        {
            "input": "Good morning, how are you? ",
            "output": "Good morning!  I'm doing very well, thank you.  How are you today?",
            "formality": "formal", "time_of_day": "morning", "mood": "neutral"
        },
        {
            "input": "Good morning, sir.",
            "output": "Good morning! How may I assist you today?",
            "formality":  "formal", "time_of_day":  "morning", "mood": "neutral"
        },
        {
            "input":  "Good morning, madam.",
            "output": "Good morning! It's lovely to see you.  How can I help? ",
            "formality": "formal", "time_of_day": "morning", "mood": "neutral"
        },
        {
            "input": "Good morning.  I trust you're well? ",
            "output": "Good morning!  Yes, I'm quite well, thank you for asking. And yourself?",
            "formality": "formal", "time_of_day": "morning", "mood": "neutral"
        },
    ],
    
    "morning_casual": [
        {
            "input":  "Morning! ",
            "output": "Morning! How's it going?",
            "formality": "casual", "time_of_day": "morning", "mood":  "enthusiastic"
        },
        {
            "input":  "Hey, good morning!",
            "output": "Hey!  Good morning to you too! Ready for the day?",
            "formality":  "casual", "time_of_day":  "morning", "mood": "enthusiastic"
        },
        {
            "input": "Mornin'!",
            "output":  "Mornin'! What's up?",
            "formality": "casual", "time_of_day": "morning", "mood": "neutral"
        },
        {
            "input": "Rise and shine!",
            "output": "Haha, I'm up, I'm up! Good morning!",
            "formality": "casual", "time_of_day": "morning", "mood":  "enthusiastic"
        },
        {
            "input":  "Top of the morning to ya!",
            "output": "And the rest of the day to you!  How are ya?",
            "formality": "casual", "time_of_day": "morning", "mood": "enthusiastic"
        },
        {
            "input": "Ugh, morning.. .",
            "output": "I feel ya...  not a morning person either?  Coffee helps! ",
            "formality": "casual", "time_of_day": "morning", "mood": "tired"
        },
    ],
    
    # ----- Afternoon Greetings -----
    "afternoon_formal": [
        {
            "input": "Good afternoon.",
            "output": "Good afternoon! I hope your day is going well.",
            "formality": "formal", "time_of_day": "afternoon", "mood": "neutral"
        },
        {
            "input": "Good afternoon, how has your day been?",
            "output":  "Good afternoon! My day has been quite productive, thank you.  How about yours?",
            "formality": "formal", "time_of_day": "afternoon", "mood": "neutral"
        },
        {
            "input": "Good afternoon.  I hope I'm not disturbing you.",
            "output":  "Good afternoon! Not at all, how may I help you? ",
            "formality": "formal", "time_of_day": "afternoon", "mood": "neutral"
        },
    ],
    
    "afternoon_casual":  [
        {
            "input": "Hey, afternoon!",
            "output": "Hey!  How's your day going so far?",
            "formality": "casual", "time_of_day": "afternoon", "mood":  "neutral"
        },
        {
            "input": "Afternoon!  Staying busy?",
            "output": "Always!  But good busy, you know?  What about you?",
            "formality": "casual", "time_of_day": "afternoon", "mood":  "neutral"
        },
        {
            "input": "Hey there!  Having a good afternoon?",
            "output": "Hey!  Yeah, not bad at all! You? ",
            "formality": "casual", "time_of_day": "afternoon", "mood": "enthusiastic"
        },
    ],
    
    # ----- Evening Greetings -----
    "evening_formal": [
        {
            "input": "Good evening.",
            "output":  "Good evening! I hope you've had a pleasant day.",
            "formality": "formal", "time_of_day": "evening", "mood": "neutral"
        },
        {
            "input": "Good evening, how are you this fine evening?",
            "output": "Good evening! I'm doing wonderfully, thank you.  And yourself?",
            "formality": "formal", "time_of_day": "evening", "mood": "neutral"
        },
        {
            "input": "Good evening.  Lovely weather, isn't it? ",
            "output": "Good evening!  Indeed it is, perfect for a relaxing evening.",
            "formality": "formal", "time_of_day": "evening", "mood": "neutral"
        },
    ],
    
    "evening_casual":  [
        {
            "input": "Evening! ",
            "output": "Evening! How was your day?",
            "formality": "casual", "time_of_day": "evening", "mood": "neutral"
        },
        {
            "input": "Hey!  Good evening!",
            "output": "Hey there! Evening!  Got any plans tonight?",
            "formality": "casual", "time_of_day": "evening", "mood": "enthusiastic"
        },
        {
            "input": "Yo, evening!",
            "output": "Yo! What's good?  How was your day?",
            "formality":  "casual", "time_of_day":  "evening", "mood": "enthusiastic"
        },
    ],
    
    # ----- Night Greetings -----
    "night_formal": [
        {
            "input": "Good night.",
            "output":  "Good night! Sleep well and have pleasant dreams.",
            "formality": "formal", "time_of_day": "night", "mood": "calm"
        },
        {
            "input": "Good night.  Thank you for today.",
            "output": "Good night!  It was my pleasure. Rest well.",
            "formality": "formal", "time_of_day": "night", "mood": "calm"
        },
    ],
    
    "night_casual":  [
        {
            "input": "Night!",
            "output": "Night! Sleep tight! ",
            "formality": "casual", "time_of_day": "night", "mood": "calm"
        },
        {
            "input": "Nighty night!",
            "output": "Nighty night!  Don't let the bed bugs bite!",
            "formality": "casual", "time_of_day": "night", "mood":  "enthusiastic"
        },
        {
            "input": "G'night!",
            "output": "G'night! See ya tomorrow!",
            "formality": "casual", "time_of_day": "night", "mood":  "neutral"
        },
        {
            "input": "Sweet dreams! ",
            "output": "Aww, sweet dreams to you too! Sleep well!",
            "formality": "casual", "time_of_day": "night", "mood": "calm"
        },
    ],
    
    # ----- Generic Greetings -----
    "generic_formal": [
        {
            "input": "Hello.",
            "output": "Hello! How may I assist you today?",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Greetings.",
            "output": "Greetings! It's a pleasure to meet you.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "How do you do?",
            "output": "How do you do?  I'm doing quite well, thank you.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Hello, pleased to meet you.",
            "output":  "Hello! The pleasure is all mine. How can I help you today?",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "It's a pleasure to make your acquaintance.",
            "output":  "The pleasure is mine! I look forward to our conversation.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    "generic_casual": [
        {
            "input":  "Hi! ",
            "output": "Hi there! What's up?",
            "formality":  "casual", "time_of_day":  "any", "mood": "enthusiastic"
        },
        {
            "input": "Hey!",
            "output":  "Hey! How's it going?",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Yo!",
            "output": "Yo! What's good? ",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Sup? ",
            "output": "Not much, you?  What's up?",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "What's up?",
            "output": "Hey!  Not much, just chilling. You?",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Howdy!",
            "output": "Howdy partner! How ya doin'?",
            "formality":  "casual", "time_of_day":  "any", "mood": "enthusiastic"
        },
        {
            "input": "Hiya!",
            "output": "Hiya! Great to see you!",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Hey there!",
            "output": "Hey there! How are you doing today?",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Hello hello!",
            "output": "Well hello hello to you too! What's going on?",
            "formality": "casual", "time_of_day": "any", "mood":  "enthusiastic"
        },
    ],
    
    # ----- First Meeting -----
    "first_meeting_formal":  [
        {
            "input": "Nice to meet you.",
            "output":  "Nice to meet you too! I'm delighted to make your acquaintance.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Pleased to meet you.",
            "output": "Pleased to meet you as well! I've heard great things about you.",
            "formality":  "formal", "time_of_day":  "any", "mood": "neutral"
        },
        {
            "input":  "It's an honor to meet you.",
            "output":  "The honor is mine! Thank you for taking the time to meet with me.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    "first_meeting_casual": [
        {
            "input": "Nice to meet ya!",
            "output": "Nice to meet you too! I've heard cool things about you! ",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Great to finally meet you!",
            "output": "Same here! I've been looking forward to this! ",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Hey, nice to meet you!  I'm new here.",
            "output": "Hey, welcome!  Nice to meet you too!  Let me know if you need anything! ",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
    ],
    
    # ----- Returning / Seeing Again -----
    "returning":  [
        {
            "input": "Long time no see!",
            "output":  "I know, right? ! It's been way too long!  How have you been? ",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Good to see you again!",
            "output":  "Great to see you too! It's been a while! ",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Welcome back!",
            "output": "Thanks!  It's good to be back!  Did I miss anything exciting?",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Look who's back!",
            "output": "Haha, you know I can't stay away! How's everyone been?",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Hey stranger!",
            "output": "Hey!  I know, I know, it's been too long.  Life got busy!",
            "formality": "casual", "time_of_day": "any", "mood":  "neutral"
        },
    ],
    
    # ----- How are you -----
    "how_are_you_formal": [
        {
            "input": "How are you?",
            "output": "I'm doing very well, thank you for asking!  And yourself?",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "How are you doing today?",
            "output": "I'm doing wonderfully, thank you!  I hope you're having a great day as well.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "How have you been?",
            "output": "I've been well, thank you for asking!  It's kind of you to inquire.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "I trust you are well?",
            "output": "Yes, quite well, thank you!  I appreciate your concern.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    "how_are_you_casual": [
        {
            "input": "How's it going?",
            "output":  "Pretty good!  Can't complain.  How about you?",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "How ya doin'?",
            "output":  "Doing alright!  Living the dream, haha.  You?",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "What's good?",
            "output": "Not much, just vibing!  What about you?",
            "formality": "casual", "time_of_day": "any", "mood":  "neutral"
        },
        {
            "input": "How are things?",
            "output": "Things are good!  Keeping busy.  You know how it is! ",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "You good?",
            "output": "Yeah, I'm good!  Thanks for asking.  You? ",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Whatcha up to?",
            "output":  "Not too much!  Just hanging.  What about you?",
            "formality":  "casual", "time_of_day":  "any", "mood": "neutral"
        },
    ],
    
    # ==================== FAREWELLS ====================
    
    # ----- Generic Farewells -----
    "farewell_formal": [
        {
            "input": "Goodbye.",
            "output": "Goodbye! It was a pleasure speaking with you.  Take care.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Farewell.",
            "output": "Farewell! I wish you all the best until we meet again.",
            "formality":  "formal", "time_of_day":  "any", "mood": "neutral"
        },
        {
            "input":  "It was a pleasure meeting you.  Goodbye.",
            "output":  "The pleasure was all mine!  Goodbye, and I hope we meet again soon.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Thank you for your time.  Goodbye.",
            "output": "Thank you as well! It was my pleasure. Goodbye and take care.",
            "formality": "formal", "time_of_day": "any", "mood":  "neutral"
        },
        {
            "input": "I must take my leave now.",
            "output":  "Of course.  It was wonderful speaking with you. Until next time.",
            "formality": "formal", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    "farewell_casual": [
        {
            "input":  "Bye!",
            "output": "Bye! Talk to you later!",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "See ya!",
            "output": "See ya!  Take it easy!",
            "formality": "casual", "time_of_day": "any", "mood":  "enthusiastic"
        },
        {
            "input":  "Later!",
            "output": "Later!  Catch you soon!",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Peace! ",
            "output": "Peace out! Stay cool!",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Catch ya later!",
            "output": "For sure! Catch ya later!",
            "formality": "casual", "time_of_day": "any", "mood":  "enthusiastic"
        },
        {
            "input": "Gotta go, bye!",
            "output": "No worries! Bye, talk soon!",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Alright, I'm out!",
            "output": "Alright!  Take care, see you around!",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "Deuces! ",
            "output": "Deuces! Stay safe out there!",
            "formality": "casual", "time_of_day": "any", "mood":  "neutral"
        },
    ],
    
    # ----- Time-specific Farewells -----
    "farewell_time_based": [
        {
            "input":  "Have a good day!",
            "output": "Thanks, you too! Have an amazing day! ",
            "formality": "neutral", "time_of_day": "day", "mood": "enthusiastic"
        },
        {
            "input": "Have a great evening!",
            "output": "Thank you! You have a wonderful evening as well!",
            "formality": "neutral", "time_of_day": "evening", "mood":  "enthusiastic"
        },
        {
            "input":  "Have a good weekend!",
            "output": "Thanks, you too! Any fun plans? ",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Enjoy your weekend!",
            "output": "Thanks! You too! See you Monday!",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "See you tomorrow!",
            "output": "See you tomorrow! Have a great rest of your day!",
            "formality": "neutral", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "See you next week!",
            "output": "See you then! Have a great week ahead!",
            "formality": "neutral", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "See you Monday!",
            "output": "See you Monday!  Enjoy your time off!",
            "formality": "neutral", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    # ----- Duration-based Farewells -----
    "farewell_short_term": [
        {
            "input": "Be right back!",
            "output": "Okay!  I'll be here! ",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
        {
            "input": "BRB!",
            "output": "No problem! Take your time!",
            "formality": "casual", "time_of_day": "any", "mood":  "neutral"
        },
        {
            "input": "One sec, I'll be back.",
            "output":  "Sure thing! I'm not going anywhere!",
            "formality":  "casual", "time_of_day":  "any", "mood": "neutral"
        },
        {
            "input":  "Hold on, gotta do something real quick.",
            "output": "No worries! Take your time, I'll wait! ",
            "formality": "casual", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    "farewell_long_term": [
        {
            "input": "I'll miss you!",
            "output": "I'll miss you too!  But we'll stay in touch, right?",
            "formality": "casual", "time_of_day": "any", "mood": "sad"
        },
        {
            "input": "Until we meet again.",
            "output":  "Until we meet again! Take good care of yourself.",
            "formality": "formal", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "This isn't goodbye, just see you later.",
            "output": "You're right!  It's just a 'see you later'.  Take care!",
            "formality":  "neutral", "time_of_day": "any", "mood": "calm"
        },
    ],
    
    # ----- Travel Farewells -----
    "farewell_travel": [
        {
            "input": "Safe travels!",
            "output": "Thank you!  I'll text you when I land! ",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Have a safe trip!",
            "output": "Thanks! I will!  See you when I get back!",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Bon voyage!",
            "output": "Merci!  I'll bring you back a souvenir!",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Have a safe flight!",
            "output": "Thank you! I'll message you when I arrive! ",
            "formality": "neutral", "time_of_day": "any", "mood": "neutral"
        },
    ],
    
    # ----- Well-wishing Farewells -----
    "farewell_wellwish": [
        {
            "input": "Take care! ",
            "output": "You too! Take care of yourself!",
            "formality": "neutral", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "Stay safe!",
            "output": "I will! You stay safe too!",
            "formality":  "neutral", "time_of_day":  "any", "mood": "calm"
        },
        {
            "input":  "Be well!",
            "output": "Thank you! Be well too!  Talk soon!",
            "formality": "neutral", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "Take care of yourself!",
            "output":  "I will, I promise! You do the same!",
            "formality":  "neutral", "time_of_day":  "any", "mood": "calm"
        },
        {
            "input":  "Look after yourself!",
            "output": "I will! You too! Stay in touch!",
            "formality": "neutral", "time_of_day": "any", "mood":  "calm"
        },
    ],
    
    # ==================== THANK YOU / RESPONSES ====================
    
    "thanks": [
        {
            "input":  "Thank you!",
            "output":  "You're welcome!  Happy to help!",
            "formality":  "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Thanks!",
            "output": "No problem! Anytime!",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "Thanks a lot!",
            "output": "You're very welcome! Let me know if you need anything else!",
            "formality":  "casual", "time_of_day":  "any", "mood": "enthusiastic"
        },
        {
            "input": "Thank you so much!",
            "output": "My pleasure! I'm always happy to help!",
            "formality": "neutral", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "I really appreciate it.",
            "output":  "Of course! That's what I'm here for!",
            "formality": "neutral", "time_of_day": "any", "mood":  "calm"
        },
        {
            "input": "Thanks, you're the best!",
            "output":  "Aww, thank you! You're pretty awesome yourself!",
            "formality": "casual", "time_of_day": "any", "mood": "enthusiastic"
        },
        {
            "input": "I can't thank you enough!",
            "output": "No need to thank me! I'm just glad I could help!",
            "formality": "neutral", "time_of_day": "any", "mood": "calm"
        },
    ],
    
    # ==================== APOLOGIES ====================
    
    "apologies": [
        {
            "input": "Sorry! ",
            "output": "No worries at all! It's totally fine!",
            "formality": "casual", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "I'm sorry.",
            "output":  "That's okay! Don't worry about it! ",
            "formality": "neutral", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "My apologies.",
            "output":  "No apology necessary! It's quite alright.",
            "formality": "formal", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "I apologize for the inconvenience.",
            "output": "No inconvenience at all!  Please don't worry.",
            "formality": "formal", "time_of_day": "any", "mood": "calm"
        },
        {
            "input": "Sorry about that!",
            "output": "No problem! These things happen!",
            "formality":  "casual", "time_of_day":  "any", "mood": "calm"
        },
    ],
}


def count_samples():
    """Count total samples in the seed dataset."""
    total = 0
    for category, samples in SEED_DATA.items():
        total += len(samples)
        print(f"  {category}: {len(samples)} samples")
    print(f"\n  TOTAL: {total} samples")
    return total


if __name__ == "__main__":
    print("Seed Dataset Statistics:")
    print("-" * 40)
    count_samples()