"""Generate a set of personas for an experiment."""

import hashlib
import random

from expforge.persona.model import PersonaSet, PersonaSpec
from expforge.persona.schema import GeneratorConfig
from expforge.persona.namegen import safe_generate_persona_name

DECIMALS = 2


def _round(v: float) -> float:
    return round(v, DECIMALS)


def _effective_seed(base_seed: int | None, experiment_id: str) -> int | None:
    """Derive a stable seed per (base_seed, experiment_id) so different experiments get different personas."""
    if base_seed is None:
        return None
    # Use SHA-256 so the same (seed, experiment_id) always yields the same persona set
    h = hashlib.sha256(f"{base_seed}_{experiment_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


class PersonaGenerator:
    """Generates persona sets from a config (e.g. random or template-based)."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        effective = _effective_seed(config.seed, config.experiment_id)
        if effective is not None:
            random.seed(effective)

    def generate(self) -> PersonaSet:
        """Produce a PersonaSet for the experiment. Names from LLM (Gemini 2.0 Flash); numbers rounded to 2 decimals.
        Uses experiment-level variation so systems differ by ~0–30%: strong determination bias (full 0–1 range),
        very uneven persona weights, and tight spread around centre so experiments are clearly 'low finish' vs 'high finish'.
        """
        # When seeded, vary number of personas (3–6) so some experiments have few dominant types, others more blend
        n = self.config.n_personas
        if self.config.seed is not None:
            n = random.randint(3, 6)
        # Very uneven weights [0.05, 2.0]: some experiments dominated by one persona, others more balanced
        raw_weights = [random.uniform(0.05, 2.0) for _ in range(n)]
        total_w = sum(raw_weights)
        weights = [_round(w / total_w) for w in raw_weights]
        # Full-range determination centre: some experiments strongly low (more abandon), others strongly high (more finish)
        determination_centre = random.uniform(0.05, 0.95)
        # Tight spread (±0.35) so the population clearly leans one way
        determination_spread = 0.35
        personas = []
        for i in range(n):
            technical = _round(random.uniform(0.0, 1.0))
            determined = _round(max(0.0, min(1.0, determination_centre + random.uniform(-determination_spread, determination_spread))))
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
                    weight=weights[i],
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
