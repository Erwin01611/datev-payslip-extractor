"""Configuration layer."""

import os
from dataclasses import dataclass, field
from datetime import datetime

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_outputs_dir() -> str:
    """Generate a timestamped output folder on the Desktop."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.expanduser("~"), "Desktop", f"Payslip_Results_{timestamp}")


@dataclass
class Settings:
    """Runtime settings."""

    project_root: str = field(default=_project_root)
    inputs_dir: str = field(default=_project_root)
    outputs_dir: str = field(default_factory=_default_outputs_dir)
    pdf_dpi: int = 150


SETTINGS = Settings()
