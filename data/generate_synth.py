import argparse
import json
import random

TEMPLATES = {
  "MediaLibrary": [
    "what movies do i have",
    "show my library",
    "find {title}",
    "do i have {title}",
    "search my movies for {title}",
  ],
  "Playback": [
    "play {title}",
    "play {title} on the tv",
    "pause",
    "resume",
    "skip forward",
    "restart {title}",
  ],
  "SearchNews": [
    "what's going on in {city} today",
    "events in {city}",
    "news in {city}",
    "what should i do this weekend in {city}",
  ],
  "CommandControl": [
    "turn off the lights",
    "turn on the lights",
    "set the thermostat to {temp}",
    "lock the front door",
  ]
}

TITLES = ["alien", "dune", "blade runner", "the matrix", "arrival", "wall-e"]
CITIES = ["denver", "boulder", "aurora"]
TEMPS = ["68", "70", "72"]

def sample(label: str) -> str:
    t = random.choice(TEMPLATES[label])
    return (t
            .replace("{title}", random.choice(TITLES))
            .replace("{city}", random.choice(CITIES))
            .replace("{temp}", random.choice(TEMPS)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    labels = list(TEMPLATES.keys())
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            label = random.choice(labels)
            f.write(json.dumps({"text": sample(label), "label": label}) + "\n")

if __name__ == "__main__":
    main()
