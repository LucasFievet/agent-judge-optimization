"""Load/save persona set from/to YAML."""

from pathlib import Path

from expforge.persona.model import PersonaSet, PersonaSpec

DECIMALS = 2


def _round(v: float) -> float:
    return round(v, DECIMALS)


def load_persona_set(path: Path | str) -> PersonaSet:
    """Load a PersonaSet from a YAML file."""
    import yaml

    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    experiment_id = data.get("experiment_id", path.stem)
    personas = [
        PersonaSpec(
            id=p["id"],
            name=p.get("name", ""),
            weight=float(p["weight"]),
            technical=float(p["technical"]),
            determined=float(p["determined"]),
            swearing=float(p["swearing"]),
            baseline_sentiment=float(p["baseline_sentiment"]),
            meta=p.get("meta", {}),
        )
        for p in data["personas"]
    ]
    return PersonaSet(experiment_id=experiment_id, personas=personas)


def save_persona_set(persona_set: PersonaSet, path: Path | str) -> None:
    """Save a PersonaSet to a YAML file."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "experiment_id": persona_set.experiment_id,
        "personas": [
            {
                "id": p.id,
                "name": p.name,
                "weight": _round(p.weight),
                "technical": _round(p.technical),
                "determined": _round(p.determined),
                "swearing": _round(p.swearing),
                "baseline_sentiment": _round(p.baseline_sentiment),
                **({"meta": p.meta} if p.meta else {}),
            }
            for p in persona_set.personas
        ],
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
