from .models import RevealAudit, RevealContext, RevealDecision, SecretRecord, SecretStatus


class RevealGate:
    """Fail-closed evaluator for all reveal prerequisites."""

    def evaluate(self, secret: SecretRecord, context: RevealContext) -> RevealDecision:
        try:
            reasons = []
            if secret.status is SecretStatus.REVEALED:
                reasons.append("already_revealed")
            conditions = secret.conditions
            if conditions.min_scene is not None and context.scene_index < conditions.min_scene:
                reasons.append("minimum_scene_not_reached")
            if conditions.required_event and conditions.required_event not in context.events:
                reasons.append("required_event_missing")
            if any(item not in context.character_knowledge for item in conditions.character_knowledge):
                reasons.append("character_knowledge_missing")
            if conditions.prerequisite_secret and conditions.prerequisite_secret not in context.revealed_secret_ids:
                reasons.append("prerequisite_secret_missing")
            if conditions.explicit_instruction and conditions.explicit_instruction not in context.explicit_instructions:
                reasons.append("explicit_instruction_missing")
            return RevealDecision(not reasons, tuple(reasons))
        except Exception:
            return RevealDecision(False, ("invalid_reveal_context",))

    def authorize(
        self, secret: SecretRecord, context: RevealContext, actor: str
    ) -> tuple[SecretRecord, RevealAudit]:
        decision = self.evaluate(secret, context)
        audit = RevealAudit(
            secret.story_id, secret.secret_id, decision.allowed, actor,
            context.scene_index, decision.reasons,
        )
        if not decision.allowed:
            raise PermissionError("secret reveal rejected: " + ",".join(decision.reasons))
        revealed = SecretRecord(
            **{**secret.__dict__, "status": SecretStatus.REVEALED,
               "revealed_scene": context.scene_index, "revealed_at": audit.timestamp}
        )
        return revealed, audit


def authorize_reveal(secret: SecretRecord, context: RevealContext, actor: str = "system"):
    """Authorize one transition and return ``(revealed_record, audit)``."""
    return RevealGate().authorize(secret, context, actor)
