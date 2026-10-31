from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class StoryContinuity:
    """Bounded, structured state passed between narrative scenes."""

    situation: str
    protagonist_goal: str
    established_facts: Tuple[str, ...] = ()
    character_knowledge: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    unresolved_conflicts: Tuple[str, ...] = ()
    previous_outcome: str = ""
    next_scene_purpose: str = ""

    def prompt_text(self) -> str:
        facts = "\n".join(f"- {item}" for item in self.established_facts[-8:]) or "- None yet."
        knowledge = "\n".join(
            f"- {character}: {', '.join(items[-4:])}"
            for character, items in sorted(self.character_knowledge.items())
            if items
        ) or "- No character knowledge recorded yet."
        conflicts = "\n".join(f"- {item}" for item in self.unresolved_conflicts[-5:]) or "- None recorded."
        return (
            f"CURRENT SITUATION: {self.situation}\n"
            f"PROTAGONIST GOAL: {self.protagonist_goal}\n"
            f"ESTABLISHED FACTS:\n{facts}\n"
            f"CHARACTER KNOWLEDGE:\n{knowledge}\n"
            f"UNRESOLVED CONFLICTS:\n{conflicts}\n"
            f"PREVIOUS OUTCOME: {self.previous_outcome or 'Opening scene; no prior outcome.'}\n"
            f"NEXT SCENE PURPOSE: {self.next_scene_purpose}"
        )

    def after_scene(self, summary: str, actions: dict, purpose: str) -> "StoryContinuity":
        facts = list(self.established_facts)
        facts.extend(f"{role} must {action.replace('_', ' ')}." for role, action in actions.items())
        knowledge = {
            character: tuple(items)
            for character, items in self.character_knowledge.items()
        }
        for role, action in actions.items():
            knowledge[role] = tuple(dict.fromkeys((*knowledge.get(role, ()), action.replace("_", " "))))
        return StoryContinuity(
            situation=summary.strip() or self.situation,
            protagonist_goal=self.protagonist_goal,
            established_facts=tuple(dict.fromkeys(facts))[-12:],
            character_knowledge=knowledge,
            unresolved_conflicts=self.unresolved_conflicts or ("The central conflict remains unresolved.",),
            previous_outcome=summary.strip(),
            next_scene_purpose=purpose,
        )
