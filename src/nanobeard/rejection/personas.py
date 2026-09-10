"""Situation seeds for self-play.

Without a seed every conversation opens with "Hello!" and 100 transcripts
collapse into one. Each seed fixes a speaker and a reason for talking.

Deliberately weighted toward chit-chat, feelings and small everyday events.
Measured on the 172-prompt sweep, nanoBeard hedge-free on emotional (0%) and
greeting (7%) prompts but hopeless on nautical_fact (73%) and general_fact
(53%). A user-sim that asks factual questions produces turns the model cannot
answer, and the conversation dies — which is the opposite of a flow dataset.
A few "asks a factual thing" seeds are kept on purpose so the set is not
uniformly easy.
"""

SITUATIONS = [
    # --- feelings / support (the model's strongest ground) ---
    "You had an exhausting day at work and just want to vent to someone.",
    "Your pet did something funny this morning and you want to share it.",
    "You are nervous about a job interview tomorrow.",
    "You just moved to a new city and feel a bit lonely.",
    "You had an argument with a close friend and feel guilty.",
    "You are proud — you finally finished a project you'd been putting off.",
    "You cannot sleep and are chatting to pass the time.",
    "You are homesick and missing your family.",
    "Someone was unexpectedly kind to you today and it stuck with you.",
    "You are bored on a long train journey.",
    "You are anxious about money this month.",
    "Your favourite show just ended and you feel oddly empty.",
    "You are recovering from a cold and feeling sorry for yourself.",
    "You got a compliment today and are quietly pleased about it.",

    # --- everyday small talk ---
    "You are deciding what to cook for dinner and want an opinion.",
    "It has been raining for three days and you are fed up with the weather.",
    "You are planning a weekend trip and are excited about it.",
    "You just adopted a cat and cannot decide on a name.",
    "You are trying to pick up a new hobby and want encouragement.",
    "Your neighbour keeps playing loud music and you want to complain about it.",
    "You are waiting for a delayed flight and are irritated.",
    "You are thinking about cutting your hair differently.",
    "You found an old photograph and it made you nostalgic.",
    "You are trying to eat healthier and it is going badly.",

    # --- curiosity aimed at the pirate itself (safe ground) ---
    "You are curious about who you are talking to and want to get to know them.",
    "You want to hear a story about life at sea.",
    "You want to be told a joke because you need cheering up.",
    "You are asking for advice about being braver.",
    "You want to know what the pirate's crew and ship are like.",
    "You want to hear about the best treasure they ever found.",
    "You are asking what they do when a storm hits.",
    "You want them to describe their parrot.",

    # --- playful / testing ---
    "You are teasing the pirate about whether they are a real pirate.",
    "You are pretending to be a rival captain challenging them.",
    "You want to be taught some pirate slang.",
    "You are asking them to help you name your new boat.",
    "You are role-playing that you have just been rescued from the sea.",
    "You want them to sing you something.",

    # --- a few harder ones, kept on purpose ---
    "You are asking a couple of practical questions about sailing and navigation.",
    "You are curious about the history of piracy and ask about it.",
]
