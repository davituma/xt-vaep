"""Carregamento de configurações YAML e caminhos do projeto."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load(name: str) -> dict:
    """Carrega configs/<name>.yaml como dicionário."""
    with open(CONFIGS / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(*parts) -> Path:
    """Caminho absoluto relativo à raiz do projeto, criando o diretório pai."""
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
