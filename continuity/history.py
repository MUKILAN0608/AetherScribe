from datetime import datetime, timezone
import json


def utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class SceneHistoryRepository:
    def __init__(self, session_factory, model):
        self.Session, self.Model = session_factory, model

    def list(self, story_id: str, limit: int = 8):
        with self.Session() as session:
            rows = (
                session.query(self.Model)
                .filter_by(story_id=story_id, accepted=True)
                .order_by(self.Model.scene_index.desc(), self.Model.id.desc())
                .limit(limit).all()
            )
            return [
                {
                    "id": row.id, "episode_id": row.episode_id,
                    "scene_index": row.scene_index, "story": row.story,
                    "summary": row.summary, "outcome": row.outcome,
                    "opening": row.opening, "ending": row.ending,
                    "beat": row.beat, "objective": row.objective,
                }
                for row in reversed(rows)
            ]

    def add(self, story_id, episode_id, scene_index, story, summary, outcome,
            opening, ending, beat, objective, accepted=True):
        with self.Session() as session:
            row = self.Model(
                story_id=story_id, episode_id=episode_id, scene_index=scene_index,
                story=story, summary=summary, outcome=outcome, opening=opening,
                ending=ending, beat=beat, objective=objective, accepted=accepted,
                diagnostics_json=json.dumps({}), created_at=utc_now_naive(),
            )
            session.add(row)
            session.commit()
            return row.id
