import csv
import re
import random
from pathlib import Path

SEED = 42
PER_DOMAIN = 2000
TRAIN_SPLIT = 0.8

TEMPLATES = {
    "NAVIGATION": {
        "templates": [
            "go to the {loc}", "take me to the {loc}", "move to the {loc}",
            "walk to the {loc}", "navigate to the {loc}", "can you go to the {loc}",
            "please go to the {loc}", "i want to go to the {loc}",
            "let's go to the {loc}", "head to the {loc}", "bring me to the {loc}",
            "lead the way to the {loc}", "head over to the {loc}",
            "follow me", "follow me to the {loc}", "come with me to the {loc}",
            "follow me please", "come follow me", "walk with me",
            "come along with me", "tag along with me", "stay close and follow me",
            "keep following me", "walk behind me", "stay with me",
            "stop moving", "stop right there", "halt", "freeze", "don't move",
            "stop walking", "stay still", "stop right now", "hold on stop",
            "quit moving around", "please stop moving now", "stay where you are",
            "don't go anywhere", "hold your position",
            "return to dock", "go back to your dock", "return home",
            "go to your charging station", "go back to base", "go back home",
            "return to your station", "dock yourself", "go charge yourself",
            "time to go back to the dock", "head back to base",
            "turn {dir}", "go {dir}", "move {dir}", "turn around", "spin around",
            "rotate {dir}", "face the {loc}", "look towards the {loc}",
            "go forward", "go backward", "move a little {dir}",
            "take {num} steps {dir}", "walk {num} steps forward",
            "come here", "come closer", "back up a little", "come to me",
            "get closer to me", "move away from the {obj}", "avoid the {obj}",
            "go around the {obj}", "stay away from the {obj}",
            "be careful near the {obj}", "watch out for the {obj}",
        ],
        "slots": {
            "loc": [
                "kitchen", "bedroom", "living room", "bathroom", "garden",
                "hallway", "garage", "dining room", "study room", "balcony",
                "front door", "back door", "playroom", "nursery", "attic",
                "basement", "porch", "laundry room", "closet", "pantry",
                "window", "door", "table", "sofa", "bookshelf", "couch",
                "corridor", "entrance", "stairs", "patio",
            ],
            "dir": ["left", "right", "forward", "backward", "around"],
            "num": ["two", "three", "four", "five", "a few", "six", "ten"],
            "obj": [
                "wall", "chair", "table", "stairs", "door", "toy",
                "box", "bag", "shoe", "cat", "dog", "lamp", "rug",
            ],
        },
    },
    "STUDY": {
        "templates": [
            "explain {topic} to me", "can you explain {topic}",
            "what is {topic}", "tell me about {topic}", "teach me about {topic}",
            "i want to learn about {topic}", "help me understand {topic}",
            "what does {topic} mean", "how does {topic} work",
            "why is {topic} important", "describe {topic} for me",
            "give me an explanation of {topic}", "break down {topic} for me",
            "can you teach me {topic}", "i don't understand {topic}",
            "i need help with {topic}", "what can you tell me about {topic}",
            "quiz me on {subj}", "test me on {subj}",
            "give me a quiz about {subj}", "ask me questions about {subj}",
            "let's do a quiz on {subj}", "can you quiz me on {subj}",
            "test my knowledge of {subj}", "quiz time for {subj}",
            "i want a quiz on {subj}", "give me a test on {subj}",
            "let's see how much i know about {subj}",
            "what is the answer to {math}", "solve {math}", "what's {math}",
            "can you help me with {math}", "how do i solve {math}",
            "help me with this math problem {math}",
            "review {topic} with me", "let's review {topic}",
            "go over {topic} again", "can we review {subj} please",
            "i need to review {topic}", "help me review for my {subj} test",
            "let's study {subj} together", "i have a question about {topic}",
            "can you clarify {topic}", "i'm confused about {topic}",
            "explain {topic} in simple words", "make {topic} easier to understand",
            "what's the difference between {topic} and {topic2}",
            "give me an example of {topic}", "show me how {topic} works",
            "is {topic} the same as {topic2}", "why do we need to learn {topic}",
            "how is {topic} used in real life", "summarize {topic} for me",
            "give me a summary of {subj}", "what are the main points of {topic}",
            "read this lesson about {topic}", "what chapter is {topic} in",
            "what's the definition of {topic}", "spell {topic} for me",
            "how do you spell {topic}", "help me practice {subj}",
        ],
        "slots": {
            "topic": [
                "photosynthesis", "gravity", "fractions", "the solar system",
                "dinosaurs", "volcanoes", "the water cycle", "multiplication",
                "division", "addition", "subtraction", "the alphabet",
                "colors", "shapes", "animals", "plants", "weather",
                "the human body", "electricity", "magnets", "sound",
                "light", "the moon", "the sun", "atoms", "molecules",
                "history", "geography", "biology", "chemistry",
                "ecosystems", "the food chain", "continents", "oceans",
                "planets", "cells", "erosion", "condensation",
            ],
            "topic2": [
                "evaporation", "condensation", "mass", "weight",
                "speed", "velocity", "energy", "force", "heat", "temperature",
            ],
            "subj": [
                "math", "science", "english", "history", "geography",
                "biology", "physics", "chemistry", "reading", "spelling",
                "grammar", "vocabulary", "social studies", "arithmetic",
            ],
            "math": [
                "five plus three", "twelve minus seven", "six times four",
                "twenty divided by five", "three plus nine",
                "eight times two", "fifteen minus six", "ten plus ten",
                "seven times three", "nine divided by three",
                "half of twenty", "double fourteen", "two times eight",
                "sixteen minus nine", "four plus eleven",
            ],
        },
    },
    "STORY": {
        "templates": [
            "tell me a story about {char}", "i want to hear a story about {char}",
            "can you tell me a {genre} story", "start a new story about {char}",
            "make up a story about {char}", "once upon a time story about {char}",
            "begin a tale about {char}", "tell me an adventure about {char}",
            "read me a story about {char}", "narrate a story about {char}",
            "i want a {genre} story", "tell me a {genre} tale",
            "create a story with {char}", "invent a story about {char}",
            "spin me a yarn about {char}", "i wanna hear about {char}",
            "make up something about {char}", "tell a {genre} story please",
            "continue the story", "what happens next", "and then what happened",
            "keep going", "go on with the story", "continue please",
            "tell me more", "don't stop the story", "what's the next part",
            "what happened after that", "i want to hear more", "then what",
            "keep telling the story", "more please", "go on",
            "what comes next in the story", "keep reading",
            "i choose the {choice}", "let's go with the {choice}",
            "take the {choice}", "i pick the {choice}",
            "go through the {choice}", "i want the {choice}",
            "choose the {choice}", "let's try the {choice}",
            "does {char} win", "will {char} be okay", "is {char} safe",
            "what does {char} find", "where does {char} go next",
            "who does {char} meet", "how does the story end",
            "is there a happy ending", "what happens to {char}",
            "pause the story", "stop the story for now", "save this story",
            "i'll come back to the story later", "bookmark this story",
            "hold on pause the story", "wait pause for a second",
            "can you tell a story with {char} and {char2}",
            "what if {char} meets {char2}",
            "start a story in a {setting}", "tell me a bedtime story",
            "tell me a short story", "tell me a funny story",
            "another story please", "new story about {char}",
        ],
        "slots": {
            "char": [
                "a dragon", "a princess", "a brave knight", "a talking cat",
                "a pirate", "a robot", "a dinosaur", "a wizard",
                "a fairy", "a superhero", "a bunny", "an astronaut",
                "a detective", "a mermaid", "a unicorn", "a puppy",
                "a bear", "an alien", "a monkey", "a ninja",
                "a witch", "a prince", "a lion", "a penguin",
            ],
            "char2": [
                "a friendly ghost", "a lost kitten", "a wise owl",
                "a tiny mouse", "a giant", "a talking tree",
                "a baby dragon", "a silly frog",
            ],
            "genre": [
                "funny", "scary", "adventure", "magical", "mystery",
                "space", "underwater", "fantasy", "silly", "exciting",
                "happy", "sad", "brave", "action", "spooky",
            ],
            "choice": [
                "left path", "right path", "secret door", "magic portal",
                "dark cave", "bright forest", "tall mountain",
                "hidden tunnel", "golden key", "silver bridge",
            ],
            "setting": [
                "castle", "forest", "spaceship", "underwater kingdom",
                "magical land", "pirate ship", "haunted house",
                "enchanted garden", "cloud city", "desert island",
            ],
        },
    },
    "CHAT": {
        "templates": [
            "what's your favorite {thing}", "do you like {thing}",
            "tell me about yourself", "what are you", "who made you",
            "how are you today", "do you have feelings",
            "what do you think about {thing}",
            "tell me a joke", "tell me something funny", "make me laugh",
            "say something silly", "do you know any jokes", "tell me a riddle",
            "tell me a fun fact", "did you know that {fact}",
            "what's interesting about {fact}",
            "why is the sky {color}", "why is grass green",
            "why do we sleep", "how do airplanes fly", "why is water wet",
            "what happens when we dream", "why do we have to eat",
            "where does rain come from", "what makes thunder",
            "how do birds fly", "why is the ocean salty",
            "hi", "hello", "hey", "good morning", "good night",
            "good afternoon", "goodbye", "see you later",
            "how's it going", "what's up", "howdy",
            "i'm bored", "i'm happy today", "i'm feeling {emotion}",
            "i had a {dayq} day", "i like {thing}",
            "my favorite {thingtype} is {thing}",
            "do you want to be friends", "you're funny", "you're smart",
            "thank you", "thanks a lot", "you're the best",
            "i love talking to you", "you are so cool",
            "can you sing a song", "do you know any songs",
            "what's your name", "how old are you", "where do you live",
            "do you eat food", "do you sleep",
            "what time is it", "what day is it today",
            "what's the weather like", "is it going to rain",
            "can you count to {num}", "say {word} in a funny voice",
            "what do you do for fun", "are you real",
            "who is your best friend", "what's your favorite joke",
        ],
        "slots": {
            "thing": [
                "color", "animal", "food", "sport", "movie", "book",
                "game", "song", "place", "toy", "ice cream", "pizza",
                "chocolate", "dinosaurs", "robots", "cars", "dogs",
                "cats", "birds", "flowers", "stars", "the moon",
                "pandas", "penguins", "rainbows",
            ],
            "thingtype": ["color", "animal", "food", "subject", "movie", "game"],
            "color": ["blue", "red", "pink", "orange", "purple", "yellow"],
            "fact": [
                "space", "the ocean", "animals", "dinosaurs", "the brain",
                "butterflies", "dolphins", "elephants", "the sun", "ants",
            ],
            "emotion": [
                "sad", "happy", "excited", "tired", "bored",
                "scared", "angry", "confused", "silly", "curious",
            ],
            "dayq": ["good", "bad", "great", "okay", "terrible", "wonderful"],
            "num": ["ten", "twenty", "a hundred", "a million"],
            "word": ["banana", "supercalifragilistic", "robot", "dinosaur", "spaghetti"],
        },
    },
    "PET": {
        "templates": [
            "hey there little {petname}", "hello my little {petname}",
            "good morning {petname}", "hi {petname} how are you",
            "aww hello {petname}", "hey {petname} i missed you",
            "hi there little {petname}", "welcome back {petname}",
            "good to see you {petname}", "hey {petname} you're so cute",
            "hi my sweet {petname}", "hello there {petname}",
            "good evening {petname}", "hey buddy",
            "nice to see you again {petname}", "i missed you {petname}",
            "you're such a good {petname}", "you're my favorite {petname}",
            "i love you {petname}", "you make me so happy {petname}",
            "let's play {petgame} together", "wanna play {petgame}",
            "can we play {petgame}", "i want to play {petgame} with you",
            "play {petgame} with me", "let's do {petgame} together",
            "come play {petgame} with me", "do you want to play {petgame}",
            "let's have fun playing {petgame}", "time for {petgame}",
            "give me a {gesture}", "i want a {gesture}",
            "can i have a {gesture}", "show me a {gesture}",
            "give me some love {petname}", "come give me a {gesture}",
            "let's {gesture}", "i need a {gesture} from you",
            "do a {trick} for me", "show me your {trick}",
            "can you do a {trick}", "do you know how to {trick}",
            "let me see you {trick}", "try doing a {trick}",
            "show me a cool {trick}", "do your best {trick}",
            "make a {face} for me", "show me your {face}",
            "do a {face}", "can you make a {face}",
            "let me see your {face}", "put on a {face}",
            "are you feeling {petmood}", "don't be {petmood}",
            "it's okay if you're {petmood}", "i'm here for you little {petname}",
            "don't worry {petname}", "everything will be fine {petname}",
            "i'll take care of you {petname}", "there there {petname}",
            "cheer up {petname}", "it's alright {petname}",
            "you're such a good {petname}", "you did great {petname}",
            "i'm proud of you {petname}", "good job {petname}",
            "well done {petname}", "you're my best friend {petname}",
            "you're adorable {petname}", "you're so sweet {petname}",
            "you make me smile {petname}", "aww {petname} you're the best",
            "pet pet pet {petname}", "scratch behind your ears {petname}",
            "belly rub time {petname}", "come here {petname} let me pet you",
            "let's cuddle {petname}", "snuggle time {petname}",
            "give me your {paw}", "shake {paw} {petname}",
            "roll over {petname}", "sit down {petname}",
        ],
        "slots": {
            "petname": [
                "robot", "buddy", "friend", "pal", "cutie",
                "sweetie", "little one", "baby", "darling",
                "love", "munchkin", "pumpkin", "sunshine",
                "nugget", "boo", "cupcake", "honeybun",
            ],
            "petgame": [
                "fetch", "tag", "hide and seek", "catch", "peek-a-boo",
                "chase", "simon says", "follow the leader",
                "red light green light", "pretend", "dress up", "ball",
                "tug of war", "treasure hunt", "patty cake",
            ],
            "gesture": [
                "hug", "high five", "fist bump", "handshake",
                "pat on the head", "cuddle", "boop", "nose boop",
                "kiss", "wave", "thumbs up",
            ],
            "trick": [
                "spin", "dance", "jump", "wiggle", "wave",
                "bow", "twirl", "hop", "shake", "balance",
                "flip", "march",
            ],
            "face": [
                "happy face", "sad face", "surprised face", "silly face",
                "angry face", "confused face", "wink", "smile",
                "pouty face", "cute face", "excited face",
            ],
            "petmood": [
                "sad", "lonely", "tired", "scared", "upset",
                "grumpy", "sleepy", "worried", "nervous",
            ],
            "paw": ["paw", "hand", "arm", "little paw"],
        },
    },
    "GAME": {
        "templates": [
            "let's play {game}", "i want to play {game}",
            "can we play {game}", "start a game of {game}",
            "begin a new game of {game}", "let's start {game}",
            "play {game} with me", "time to play {game}",
            "i feel like playing {game}", "how about a game of {game}",
            "open {game}", "launch {game}", "load up {game}",
            "new game please", "start a new game", "i want to play a game",
            "let's play something", "pick a game for me",
            "surprise me with a game", "what games do you have",
            "what games can we play", "got any {game} going",
            "the answer is {ans}", "i think it's {ans}",
            "my answer is {ans}", "i choose {ans}", "i pick {ans}",
            "i'll go with {ans}", "is it {ans}", "it must be {ans}",
            "i guess {ans}", "let me guess {ans}", "i bet it's {ans}",
            "{ans} is my final answer", "i'm going with {ans}",
            "definitely {ans}", "i'll say {ans}",
            "quit the game", "stop the game", "end the game",
            "i don't want to play anymore", "exit the game",
            "close the game", "game over", "i'm done playing",
            "let's stop playing", "enough gaming", "no more games",
            "give me a hint", "i need a hint", "can i get a clue",
            "help me out with this game", "i'm stuck give me a hint",
            "what's the hint", "one more clue please", "hint please",
            "give me another hint", "i don't know give me a hint",
            "what's my score", "how am i doing", "am i winning",
            "what level am i on", "next level", "make it harder",
            "make it easier", "skip this one", "next question",
            "i want a harder {game}", "easy mode please",
        ],
        "slots": {
            "game": [
                "chess", "tic-tac-toe", "trivia", "word guess", "math quiz",
                "memory match", "twenty questions", "riddles", "hangman",
                "i spy", "rock paper scissors", "number guessing",
                "spelling bee", "simon says", "would you rather",
                "true or false", "puzzle", "maze", "card game",
                "animal quiz", "geography quiz", "science quiz",
                "checkers", "connect four", "word scramble",
            ],
            "ans": [
                "A", "B", "C", "D", "option one", "option two",
                "true", "false", "yes", "no", "three", "seven",
                "the cat", "the dog", "blue", "red", "Paris",
                "the sun", "gravity", "water", "left", "right",
                "Jupiter", "oxygen", "five",
            ],
        },
    },
    "SYSTEM": {
        "templates": [
            "go to sleep", "time to sleep", "sleep mode", "take a nap",
            "power down", "shut down", "turn off", "hibernate",
            "goodnight go to sleep", "sleep now",
            "it's bedtime go to sleep", "close your eyes",
            "rest mode", "standby mode", "enter sleep mode now",
            "restart yourself", "restart", "reboot", "reboot yourself",
            "restart the system", "do a restart", "power cycle",
            "reset yourself", "start fresh", "restart everything",
            "refresh the system", "restart your brain", "reboot your system",
            "turn up the volume", "turn down the volume",
            "volume up", "volume down", "louder please", "quieter please",
            "make it louder", "make it quieter",
            "increase volume", "decrease volume", "set volume to {level}",
            "mute", "unmute", "mute the sound",
            "turn the sound off", "turn the sound on", "silence",
            "i can't hear you speak louder", "you're too loud", "you're too quiet",
            "stop everything", "stop", "cancel", "abort",
            "cancel everything", "stop all tasks", "emergency stop",
            "quit everything", "cancel what you're doing",
            "stop whatever you're doing",
            "shut up", "be quiet", "hush", "stop talking", "pause everything",
            "what's your battery level", "how much battery do you have",
            "are you charging", "check your status", "run diagnostics",
            "system check", "what version are you",
            "update yourself", "check for updates",
            "change language to {lang}", "switch to {lang}", "speak in {lang}",
            "set brightness to {level}", "dim the lights",
            "enable {sysfeature}", "disable {sysfeature}",
            "turn on {sysfeature}", "turn off {sysfeature}",
        ],
        "slots": {
            "level": [
                "maximum", "minimum", "fifty percent", "half",
                "seventy percent", "low", "medium", "high", "full",
            ],
            "lang": [
                "english", "spanish", "french", "hindi", "arabic",
                "chinese", "japanese", "german", "korean",
            ],
            "sysfeature": [
                "bluetooth", "wifi", "night mode", "do not disturb",
                "airplane mode", "dark mode", "notifications",
            ],
        },
    },
}

CHILD_PREFIXES = ["", "", "", "", "um ", "uh ", "hey ", "okay ", "so ", "hmm ", "oh "]
CHILD_SUFFIXES = ["", "", "", "", " please", " now", " okay", " thanks", " right now"]


def fill_template(tmpl, slots):
    result = tmpl
    for name in re.findall(r"\{(\w+)\}", tmpl):
        base = name.rstrip("0123456789")
        if base in slots:
            result = result.replace("{" + name + "}", random.choice(slots[base]), 1)
    return result


def generate_utterances(templates, slots, count):
    seen = set()
    max_tries = count * 25
    tries = 0

    while len(seen) < count and tries < max_tries:
        tmpl = random.choice(templates)
        utt = fill_template(tmpl, slots).strip().lower()
        seen.add(utt)
        tries += 1

    base = list(seen)
    while len(seen) < count:
        utt = random.choice(base)
        pre = random.choice(CHILD_PREFIXES)
        suf = random.choice(CHILD_SUFFIXES)
        seen.add((pre + utt + suf).strip())

    return list(seen)[:count]


def main():
    random.seed(SEED)
    out_dir = Path(__file__).parent
    rows = []

    print("Generating synthetic utterances...")
    for domain, cfg in TEMPLATES.items():
        utts = generate_utterances(cfg["templates"], cfg["slots"], PER_DOMAIN)
        rows.extend((u, domain) for u in utts)
        print(f"  {domain}: {len(utts)}")

    random.shuffle(rows)
    split = int(len(rows) * TRAIN_SPLIT)

    for fname, data in [("train.csv", rows[:split]), ("val.csv", rows[split:])]:
        p = out_dir / fname
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "domain"])
            w.writerows(data)
        print(f"Wrote {p} ({len(data)} rows)")

    print(f"\nTotal: {len(rows)} | Train: {split} | Val: {len(rows) - split}")


if __name__ == "__main__":
    main()
