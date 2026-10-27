"""AI Environment Properties Analyzer for Shibu AI.

Classifies a *described scenario* (not Shibu AI itself, unless the user
explicitly asks about Shibu AI / this chatbot) on four properties:

  Observability: Fully Observable | Partially Observable | Undetermined
  Predictability: Deterministic | Stochastic | Undetermined
  Task Structure: Sequential | Episodic | Undetermined
  Environment Dynamics: Static | Dynamic | Semi-dynamic | Undetermined

How it reasons (no hardcoded answer for an arbitrary scenario):
  1. Known scenario profiles cover the classic textbook cases
     (self-driving car, chess, poker, robot vacuum, ...). Each profile
     states its classifications AND why, in scenario-specific words.
  2. For any other scenario, keyword evidence in the user's own
     description is scored: hidden/limited sensors -> Partially
     Observable; dice/random/traffic/opponent -> Stochastic; fixed
     rules/same action same result -> Deterministic; moving agents
     -> Dynamic; a score that decays with time -> Semi-dynamic;
     independent tasks -> Episodic; actions that affect later
     decisions -> Sequential.
  3. A property with no usable evidence is "Undetermined", with a
     reason naming the missing information, instead of a guess.
  4. If the user never describes an environment scenario, the router
     must not call this at all (see should_analyze); unrelated
     questions therefore never get a forced table.

Validation: validate_analysis() checks every classification is in the
allowed set and every reason is non-empty before anything is rendered.
If validation fails, the caller must still show the normal answer.
"""

import re

ALLOWED = {
    "Observability": {"Fully Observable", "Partially Observable", "Undetermined"},
    "Predictability": {"Deterministic", "Stochastic", "Undetermined"},
    "Task Structure": {"Sequential", "Episodic", "Undetermined"},
    "Environment Dynamics": {"Static", "Dynamic", "Semi-dynamic", "Undetermined"},
}

# Classic scenarios: (match keywords, display name, classifications, reasons)
PROFILES = [
    (("self-driving car", "self driving car", "autonomous car", "driverless car"),
     "A self-driving car navigating through traffic",
     {"Observability": "Partially Observable",
      "Predictability": "Stochastic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Dynamic"},
     {"Observability": "The car cannot observe everything around it — sensors are limited and other road users can be hidden.",
      "Predictability": "Traffic and road conditions introduce uncertainty, so the same action does not always give the same outcome.",
      "Task Structure": "Current driving decisions affect future movements and positions.",
      "Environment Dynamics": "Other vehicles and pedestrians keep moving while the car is deciding."}),
    (("chess",),
     "A chess game",
     {"Observability": "Fully Observable",
      "Predictability": "Deterministic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Static"},
     {"Observability": "All pieces and their positions are visible to both players.",
      "Predictability": "The rules are fixed: the same board and move always lead to the same new board.",
      "Task Structure": "Each move changes the board and shapes every later decision.",
      "Environment Dynamics": "The board does not change while a player is thinking."}),
    (("tic-tac-toe", "tic tac toe", "noughts and crosses"),
     "A tic-tac-toe game",
     {"Observability": "Fully Observable",
      "Predictability": "Deterministic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Static"},
     {"Observability": "The whole grid is visible to both players.",
      "Predictability": "The rules are fixed: the same grid and move always give the same result.",
      "Task Structure": "Each move fills a cell and limits the moves that follow.",
      "Environment Dynamics": "The grid does not change while a player is deciding."}),
    (("poker",),
     "A poker game",
     {"Observability": "Partially Observable",
      "Predictability": "Stochastic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Static"},
     {"Observability": "Opponents' cards are hidden from the player.",
      "Predictability": "Shuffled cards and opponents' choices make outcomes uncertain.",
      "Task Structure": "Bets and cards in earlier rounds affect later rounds.",
      "Environment Dynamics": "The cards in play do not change while a player is deciding."}),
    (("robot vacuum", "vacuum cleaner robot", "cleaning robot"),
     "A robot vacuum cleaning a home",
     {"Observability": "Partially Observable",
      "Predictability": "Stochastic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Dynamic"},
     {"Observability": "Its sensors only see part of the room at a time and cannot know where all the dust is.",
      "Predictability": "Furniture, pets and people make what it meets next unpredictable.",
      "Task Structure": "Where it has already cleaned changes where it should go next.",
      "Environment Dynamics": "People and pets move around while it is working."}),
    (("warehouse robot", "robot navigating", "delivery robot"),
     "A robot navigating a warehouse",
     {"Observability": "Partially Observable",
      "Predictability": "Stochastic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Dynamic"},
     {"Observability": "Its sensors cannot see around every shelf or know every obstacle in advance.",
      "Predictability": "Other robots, workers and blocked aisles make outcomes uncertain.",
      "Task Structure": "Each movement changes its position and the route that follows.",
      "Environment Dynamics": "Other robots and workers keep moving while it decides."}),
    (("crossword",),
     "Solving a crossword puzzle",
     {"Observability": "Fully Observable",
      "Predictability": "Deterministic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Static"},
     {"Observability": "The whole grid and all clues are visible.",
      "Predictability": "A correct word is always correct; there is no randomness in the puzzle.",
      "Task Structure": "Letters filled in for one word help or constrain crossing words.",
      "Environment Dynamics": "The puzzle does not change while the solver is thinking."}),
    (("spam filter", "image classification", "image classifier"),
     "An image / spam classifier",
     {"Observability": "Fully Observable",
      "Predictability": "Deterministic",
      "Task Structure": "Episodic",
      "Environment Dynamics": "Static"},
     {"Observability": "The whole input (image or message) is available to the classifier.",
      "Predictability": "The same trained model gives the same label for the same input.",
      "Task Structure": "Each item is classified independently; one decision does not affect the next.",
      "Environment Dynamics": "The item does not change while it is being classified."}),
    (("shibu", "this chatbot", "this ai", "your environment", "your own environment"),
     "Shibu AI (this chatbot)",
     {"Observability": "Partially Observable",
      "Predictability": "Deterministic",
      "Task Structure": "Sequential",
      "Environment Dynamics": "Static"},
     {"Observability": "Shibu AI sees only the user's message, not the user's full situation or intent.",
      "Predictability": "The same message and rules lead to the same rule-engine answer.",
      "Task Structure": "Earlier messages set the conversation context for later answers.",
      "Environment Dynamics": "The chat does not change while Shibu AI is composing an answer."}),
]

# Keyword evidence for scenarios without a profile. Each entry is
# (property, classification, keywords, reason-template).
EVIDENCE = [
    ("Observability", "Partially Observable",
     ("hidden", "cannot see", "cannot observe", "limited sensor", "sensor",
      "unknown", "not visible", "blind spot", "occluded"),
     "The description mentions limited or hidden information, so the agent cannot see the full state."),
    ("Observability", "Fully Observable",
     ("fully visible", "all visible", "complete information", "whole board",
      "entire state", "everything is visible"),
     "The description says the full state is visible to the agent."),
    ("Predictability", "Stochastic",
     ("random", "dice", "traffic", "weather", "unpredictable", "uncertain",
      "opponent", "chance", "shuffled"),
     "The description mentions randomness or unpredictable factors, so outcomes can vary."),
    ("Predictability", "Deterministic",
     ("fixed rules", "same action", "same result", "no randomness", "rule-based"),
     "The description says fixed rules govern the outcome, so the same state and action give the same result."),
    ("Task Structure", "Episodic",
     ("independent", "separate task", "each task", "one by one", "does not affect the next"),
     "The description says tasks or decisions are independent of each other."),
    ("Task Structure", "Sequential",
     ("affects future", "affect later", "next step", "sequence", "route",
      "navigation", "planning", "previous decision"),
     "The description says current actions shape later decisions."),
    ("Environment Dynamics", "Dynamic",
     ("moving", "keep moving", "changing while", "other agents", "pedestrians",
      "vehicles moving", "live traffic"),
     "The description says other things keep changing while the agent decides."),
    ("Environment Dynamics", "Semi-dynamic",
     ("score decreases", "time affects", "time passing", "performance score",
      "deadline", "time limit"),
     "The description says the world itself stays still but passing time affects the score."),
    ("Environment Dynamics", "Static",
     ("does not change", "doesn't change", "fixed board", "unchanged",
      "stays the same"),
     "The description says the environment does not change while the agent decides."),
]

MISSING = {
    "Observability": "No information was given about what the agent can or cannot observe.",
    "Predictability": "No information was given about whether outcomes are fixed or involve randomness.",
    "Task Structure": "No information was given about whether decisions affect later decisions.",
    "Environment Dynamics": "No information was given about whether the environment changes during decision-making.",
}


def _clean_scenario(text):
    """Turn a classify request into a readable scenario label."""
    t = re.sub(r"(?i)^(classify|analyse|analyze|describe|what is|what are)\s+", "", text.strip())
    t = re.sub(r"(?i)\b(environment|environment properties|environment classification|properties of)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .?")
    return t[:120] if t else "The described scenario"


def should_analyze(text):
    """True only when the user actually asks about an AI environment.

    Explicit requests (classify / environment properties / environment
    type) and scenario + environment/agent wording count. A plain
    topic question ("What is a self-driving car?") does not, so
    unrelated answers are never forced through the analyzer.
    """
    low = text.lower()
    if re.search(r"\bclassify\b.*\benvironment\b|\benvironment properties\b|\benvironment classification\b|\bwhat (type|kind) of environment\b|\bfully observable\b|\bpartially observable\b", low):
        return True
    if re.search(r"\benvironment\b", low) and re.search(r"\b(agent|scenario|car|robot|game|chess|poker|crossword|classifier|shibu|chatbot)\b", low):
        return True
    if re.search(r"\bclassify\b", low) and re.search(r"\b(car|robot|game|chess|poker|crossword|agent|scenario)\b", low):
        return True
    return False


def is_explicit_request(text):
    low = text.lower()
    return bool(re.search(r"\bclassify\b|\benvironment properties\b|\benvironment classification\b|\bwhat (type|kind) of environment\b", low))


def analyze_environment(text):
    """Return an analysis dict, or None when this is not an environment question."""
    if not should_analyze(text):
        return None
    low = text.lower()
    for keywords, name, classes, reasons in PROFILES:
        if any(k in low for k in keywords):
            # "Shibu" profile only when the user really means this system.
            if name.startswith("Shibu AI") and not re.search(r"\bshibu\b|this chatbot|this ai|your environment", low):
                continue
            return _build(name, classes, reasons)
    # Generic reasoning from the user's own description.
    classes, reasons = {}, {}
    for prop in ALLOWED:
        found = None
        for p, cls, kws, reason in EVIDENCE:
            if p == prop and any(k in low for k in kws):
                found = (cls, reason)
                break
        if found:
            classes[prop], reasons[prop] = found
        else:
            classes[prop] = "Undetermined"
            reasons[prop] = MISSING[prop]
    return _build(_clean_scenario(text), classes, reasons)


def _build(scenario, classes, reasons):
    props = [{"property": p, "classification": classes[p], "reason": reasons[p]}
             for p in ("Observability", "Predictability", "Task Structure", "Environment Dynamics")]
    analysis = {"scenario": scenario, "properties": props}
    if not validate_analysis(analysis):
        return None
    return analysis


def validate_analysis(analysis):
    """Check classifications are in the allowed set and reasons exist."""
    try:
        if not analysis or not analysis.get("scenario"):
            return False
        props = analysis["properties"]
        if len(props) != 4:
            return False
        for item in props:
            if item["property"] not in ALLOWED:
                return False
            if item["classification"] not in ALLOWED[item["property"]]:
                return False
            if not str(item.get("reason", "")).strip():
                return False
        return True
    except Exception:
        return False


def format_environment_table(analysis):
    """Markdown table + scenario line, for the chat text and API clients."""
    lines = [f"Scenario: {analysis['scenario']}", "",
             "AI Environment Properties", "",
             "| Property | Classification | Reason |",
             "|---|---|---|"]
    for item in analysis["properties"]:
        lines.append(f"| {item['property']} | {item['classification']} | {item['reason']} |")
    return "\n".join(lines)
