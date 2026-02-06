"""Generate a set of personas for an experiment."""

import random

from expforge.persona.model import PersonaSet, PersonaSpec
from expforge.persona.schema import GeneratorConfig
from expforge.persona.namegen import safe_generate_persona_name

DECIMALS = 2


def _round(v: float) -> float:
    return round(v, DECIMALS)


class PersonaGenerator:
    """Generates persona sets from a config (e.g. random or template-based)."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)

    def generate(self) -> PersonaSet:
        """Produce a PersonaSet for the experiment. Names from LLM (Gemini 2.0 Flash); numbers rounded to 2 decimals."""
        personas = []
        for i in range(self.config.n_personas):
            technical = _round(random.uniform(0.0, 1.0))
            determined = _round(random.uniform(0.0, 1.0))
            swearing = _round(random.uniform(0.0, 1.0))
            baseline_sentiment = _round(random.uniform(0.0, 1.0))
            name = safe_generate_persona_name(
                technical, determined, swearing, baseline_sentiment,
                fallback=f"Persona {i}",
            )
            personas.append(
                PersonaSpec(
                    id=f"persona_{i}",
                    name=name,
                    weight=1.0 / self.config.n_personas,
                    technical=technical,
                    determined=determined,
                    swearing=swearing,
                    baseline_sentiment=baseline_sentiment,
                    meta={},
                )
            )
        out = PersonaSet(experiment_id=self.config.experiment_id, personas=personas)
        out.normalize_weights()
        return out


def generate_persona_set(experiment_id: str, n_personas: int = 5, seed: int | None = None) -> PersonaSet:
    """Convenience: generate a persona set for an experiment."""
    config = GeneratorConfig(experiment_id=experiment_id, n_personas=n_personas, seed=seed)
    return PersonaGenerator(config).generate()
