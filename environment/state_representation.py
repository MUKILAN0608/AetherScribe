import numpy as np

class StateRepresentation:
    """
    Converts narrative state into a numeric vector
    suitable for quantum encoding. Optimized for v5.4.2.
    """

    GENRE_MAP = {
        "Thriller": 0,
        "Horror": 1,
        "Comedy": 2,
        "Romance": 3,
        "Drama": 4,
        "Fantasy": 5
    }

    PHASE_MAP = {
        "intro": 0,
        "rising": 1,
        "climax": 2
    }

    @staticmethod
    def encode(phase, tension, genre, step):
        """
        Encodes the state into a vector: [tension, phase_val, step_val, genre_val]
        """
        phase_val = StateRepresentation.PHASE_MAP.get(phase.lower(), 0) / 2.0
        genre_val = StateRepresentation.GENRE_MAP.get(genre.capitalize(), 0) / 5.0
        step_val = step / 10.0 # Normalized step

        return np.array([
            tension,
            phase_val,
            step_val,
            genre_val
        ], dtype=np.float32)
