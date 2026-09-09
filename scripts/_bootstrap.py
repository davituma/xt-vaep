"""Garante que `src` seja importável quando o script roda de qualquer diretório."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
