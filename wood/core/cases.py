"""Case selection and MAT image auto-detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


FACES = ("face_002", "face_005")
PROMPTS = ("add black sunglasses", "add headphones")
OBJECTIVES = ("vae_conditioning", "unet_prediction")
MODEL_NAME = "instruct"


@dataclass(frozen=True)
class Case:
    face_id: str
    prompt: str

    @property
    def slug(self) -> str:
        prompt_slug = self.prompt.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        return f"{self.face_id}__{prompt_slug}"


@dataclass(frozen=True)
class RunSpec:
    objective: str
    case: Case
    seed: int = 1234

    @property
    def model(self) -> str:
        return MODEL_NAME

    @property
    def slug(self) -> str:
        return f"blank_objective_ref__{self.model}__{self.objective}__{self.case.slug}"


def all_cases() -> list[Case]:
    return [Case(face_id, prompt) for face_id in FACES for prompt in PROMPTS]


def build_matrix(quick: bool = False) -> list[RunSpec]:
    cases = all_cases()
    if quick:
        return [
            RunSpec("vae_conditioning", cases[0]),
            RunSpec("unet_prediction", cases[1]),
        ]
    return [RunSpec(objective, case) for objective in OBJECTIVES for case in cases]


def resolve_image_path(mat_root: Path, face_id: str) -> Path:
    folder = mat_root / "data" / face_id
    if not folder.exists():
        raise FileNotFoundError(f"Missing MAT face folder: {folder}")
    for name in ("instruct_512.png", "flux_768.png", "master_1024.png"):
        path = folder / name
        if path.exists():
            return path
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if images:
        return images[0]
    raise FileNotFoundError(f"No usable image found for {face_id}. Expected instruct_512.png or another image.")


def print_resolved_cases(mat_root: Path) -> None:
    for face_id in FACES:
        path = resolve_image_path(mat_root, face_id)
        print(f"[wood] instruct {face_id}: {path}")
