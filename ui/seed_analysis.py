"""
Character name and genre extraction from narrative premises.
"""
import re
from typing import Dict, List, Tuple

GENRE_KEYWORDS = {
    "Horror": (
        "ghost murder blood dark fear dead night monster demon haunted shadow scream curse grave"
    ),
    "Thriller": (
        "spy vault corporate detective heist chase conspiracy investigate secret agent ransom"
        "obstructs inquiry concealed hidden"
    ),
    "Comedy": "funny laugh joke absurd prank hilarious witty humor silly",
    "Romance": "love kiss heart passion together relationship chemistry attraction proposal",
    "Drama": "family grief betrayal emotional conflict struggle truth orphan",
    "Fantasy": "magic dragon sword realm spell wizard kingdom ancient myth quest rune",
}

# Words that look like names but rarely are (sentence starters, domains, etc.)
NAME_STOPS = {
    "The", "A", "An", "In", "On", "At", "And", "He", "She", "It", "They", "But", "Or", "If",
    "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does",
    "Did", "Will", "Would", "Could", "Should", "May", "Might", "Must", "Shall", "Can",
    "To", "Of", "For", "With", "By", "From", "Into", "Through", "During", "Before", "After",
    "Above", "Below", "Between", "Under", "Again", "Further", "Then", "Once", "While",
    "When", "Where", "Why", "How", "All", "Each", "Few", "More", "Most", "Other", "Some",
    "Such", "No", "Nor", "Not", "Only", "Own", "Same", "So", "Than", "Too", "Very",
    "Just", "Also", "Now", "Here", "There", "This", "That", "These", "Those", "What",
    "Which", "Who", "Whom", "Whose", "His", "Her", "Its", "Our", "Your", "Their", "My",
    "Story", "Chapter", "Scene", "Premise", "Research", "Corporate", "Secret",
    "Hidden", "Concealed", "Intelligence", "Inquiry", "Vault", "City", "World", "Kingdom",
    "English", "Spanish", "French", "German", "Japanese", "Monday", "Tuesday", "January",
    "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "North", "South", "East", "West", "New", "Old", "Young",
    "Agent", "Character", "Lead", "Hero", "Villain", "Friend", "Ally", "Enemy",
}

TITLE_PREFIXES = (
    r"(?:Dr|Mr|Mrs|Ms|Prof|Professor|Director|Captain|Agent|Detective|Inspector|"
    r"Commander|General|Lady|Lord|Sir|Queen|King|Prince|Princess)\.?"
)

# Explicit naming patterns
EXPLICIT_NAME_PATTERNS = [
    re.compile(r"(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I),
    re.compile(
        r"(?:protagonist|hero|lead|villain|antagonist|ally|friend)\s*[:\-]\s*"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        re.I,
    ),
    re.compile(
        rf"({TITLE_PREFIXES})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        re.I,
    ),
]

PROPER_NOUN_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)

ROLE_KEYWORDS = {
    "antagonist": (
        "obstruct opposes blocks stops villain enemy antagonist betray attacks hunts "
        "pursues traps sabotages corrupt director against"
    ),
    "ally": (
        "helps assists supports ally friend companion intelligence supplies guides "
        "rescues protects advises partner lyra margin periphery"
    ),
    "protagonist": (
        "seeks investigates searches discovers protagonist hero lead main finds uncovers pursues"
    ),
}


def _genre_scores(text_lower: str) -> Dict[str, int]:
    scores = {}
    for genre, words in GENRE_KEYWORDS.items():
        scores[genre] = sum(1 for w in words.split() if w in text_lower)
    return scores


def _normalize_name(name: str) -> str:
    name = " ".join(name.split())
    parts = name.split()
    if not parts:
        return ""
    name = " ".join(p[:1].upper() + p[1:] if p else "" for p in parts)
    return _trim_name(name)


TRAILING_NON_NAME = {
    "Obstructs", "Investigates", "Supplies", "Seeks", "Finds", "Discovers", "Helps",
    "Supports", "Blocks", "Stops", "Chases", "Reveals", "Hides", "Opens", "Closes",
    "While", "When", "During", "After", "Before", "Because", "Although", "However",
    "Inquiry", "Intelligence", "Periphery", "Vault", "Corporate", "Concealed",
}


def _trim_name(name: str) -> str:
    parts = name.split()
    while len(parts) > 1 and parts[-1] in TRAILING_NON_NAME:
        parts.pop()
    return " ".join(parts)


TITLE_WORDS = {
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Professor", "Director", "Captain", "Agent",
    "Detective", "Inspector", "Commander", "General", "Lady", "Lord", "Sir",
    "Queen", "King", "Prince", "Princess",
}


def _is_valid_name_candidate(name: str) -> bool:
    if not name or len(name) < 2:
        return False
    parts = name.split()
    if parts and parts[0] in TITLE_WORDS and len(parts) >= 2:
        pass
    elif any(p in NAME_STOPS for p in parts):
        return False
    if name in NAME_STOPS:
        return False
    # Reject all-caps acronyms
    if name.isupper() and len(name) <= 4:
        return False
    # Single short words unlikely to be full names unless repeated
    if len(parts) == 1 and len(parts[0]) < 3:
        return False
    return True


def _score_name(name: str, full_text: str) -> float:
    score = 0.0
    parts = name.split()
    text_lower = full_text.lower()
    name_lower = name.lower()

    if len(parts) >= 2:
        score += 4.0
    if len(parts) == 3:
        score += 1.0

    occurrences = len(re.findall(re.escape(name_lower), text_lower))
    score += min(occurrences, 4) * 1.5

    if re.match(rf"^{TITLE_PREFIXES}\s+", name, re.I):
        score += 3.0

    # Penalize sentence-initial singletons: "While Elias" -> only Elias might be caught alone
    if len(parts) == 1:
        score -= 1.0
        # Boost if appears mid-sentence with lowercase before it
        if re.search(rf"[a-z]\s+{re.escape(name)}\b", full_text):
            score += 2.0

    return score


def extract_character_names(text: str, limit: int = 3) -> List[str]:
    """Return up to `limit` character names ranked by confidence."""
    if not text.strip():
        return []

    found: Dict[str, float] = {}

    for pattern in EXPLICIT_NAME_PATTERNS:
        for match in pattern.finditer(text):
            groups = [g for g in match.groups() if g]
            if not groups:
                continue
            if len(groups) >= 2 and re.match(TITLE_PREFIXES, groups[0], re.I):
                name = _normalize_name(f"{groups[0]} {groups[1]}")
            else:
                name = _normalize_name(groups[-1])
            if _is_valid_name_candidate(name):
                found[name] = found.get(name, 0) + 5.0

    for match in PROPER_NOUN_PATTERN.finditer(text):
        name = _normalize_name(match.group(1))
        if not _is_valid_name_candidate(name):
            continue
        score = _score_name(name, text)
        if score > 0:
            found[name] = found.get(name, 0) + score

    names_sorted = sorted(found.items(), key=lambda x: (-x[1], -len(x[0])))
    selected: List[str] = []
    for name, _ in names_sorted:
        if any(name != other and name in other for other in selected):
            continue
        selected = [other for other in selected if not (other != name and other in name)]
        if name not in selected:
            selected.append(name)
        if len(selected) >= limit:
            break

    return selected[:limit]


def assign_character_roles(text: str, names: List[str]) -> Dict[str, str]:
    """Map protagonist / antagonist / ally using titles, order, and local context."""
    roles = ["protagonist", "antagonist", "ally"]
    if not names:
        return {r: f"Character {i + 1}" for i, r in enumerate(roles)}

    text_lower = text.lower()
    unique = list(dict.fromkeys(names))

    titled = [n for n in unique if n.split()[0] in TITLE_WORDS]
    untitled = [n for n in unique if n not in titled]

    # Sort by first appearance in premise
    def first_index(n: str) -> int:
        i = text_lower.find(n.lower())
        return i if i >= 0 else 10_000

    untitled.sort(key=first_index)
    titled.sort(key=first_index)
    unique.sort(key=first_index)

    result: Dict[str, str] = {}
    used = set()

    # Lead: first untitled multi-word name, else first name in text
    protagonist_candidates = [n for n in untitled if len(n.split()) >= 2] or untitled or unique
    protagonist = protagonist_candidates[0]
    result["protagonist"] = protagonist
    used.add(protagonist)

    # Opponent: titled name (Director Kael), else name near obstruct/opposes keywords
    antagonist = None
    if titled:
        antagonist = titled[0]
    else:
        best_score = -1.0
        for name in unique:
            if name in used:
                continue
            idx = text_lower.find(name.lower())
            window = text_lower[max(0, idx - 60) : idx + len(name) + 60]
            score = sum(1 for kw in ROLE_KEYWORDS["antagonist"].split() if kw in window)
            if score > best_score:
                best_score = score
                antagonist = name
        if not antagonist:
            remaining = [n for n in unique if n not in used]
            antagonist = remaining[0] if remaining else unique[1 % len(unique)]

    result["antagonist"] = antagonist
    used.add(antagonist)

    # Support: remaining name, prefer one near help/supplies/intelligence
    ally = None
    best_score = -1.0
    for name in unique:
        if name in used:
            continue
        idx = text_lower.find(name.lower())
        window = text_lower[max(0, idx - 60) : idx + len(name) + 60]
        score = sum(1 for kw in ROLE_KEYWORDS["ally"].split() if kw in window)
        if score > best_score:
            best_score = score
            ally = name
    if not ally:
        remaining = [n for n in unique if n not in used]
        ally = remaining[0] if remaining else f"Character 3"

    result["ally"] = ally
    return result


def analyze_seed(
    text: str, api_key: str | None = None, prefer_gemini: bool = True
) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Returns (recommended_genre, detected_names, role_map).
    role_map keys: protagonist, antagonist, ally
  Uses Gemini when api_key is available; falls back to heuristics.
    """
    if not text.strip():
        return "Thriller", [], {}

    if prefer_gemini and api_key:
        try:
            from llm.entity_extractor import extract_entities_gemini

            data = extract_entities_gemini(text, api_key)
            if data.get("protagonist"):
                role_map = {
                    "protagonist": data["protagonist"],
                    "antagonist": data.get("antagonist") or "Character 2",
                    "ally": data.get("ally") or "Character 3",
                }
                names = data.get("detected_names") or list(role_map.values())
                return data.get("genre", "Drama"), names[:3], role_map
        except Exception:
            pass

    text_lower = re.sub(r"[^\w\s]", " ", text.lower())
    text_lower = re.sub(r"\s+", " ", text_lower)

    scores = _genre_scores(text_lower)
    recommended = max(scores, key=scores.get) if max(scores.values()) > 0 else "Thriller"

    names = extract_character_names(text, limit=5)
    role_map = assign_character_roles(text, names[:3])

    # Ensure three entries
    for i, role in enumerate(["protagonist", "antagonist", "ally"]):
        if role not in role_map or not role_map[role]:
            fallback_names = [n for n in names if n not in role_map.values()]
            role_map[role] = (
                fallback_names[0]
                if fallback_names
                else names[i] if i < len(names) else f"Character {i + 1}"
            )

    display_names = list(dict.fromkeys(role_map.values()))[:3]
    while len(display_names) < 3:
        display_names.append(f"Character {len(display_names) + 1}")

    return recommended, display_names, role_map
