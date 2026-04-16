from __future__ import annotations

from pathlib import Path
from typing import Any

from mako.template import Template


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def render_template(template_name: str, **context: Any) -> str:
    template_path = _TEMPLATE_DIR / template_name
    template = Template(filename=str(template_path), strict_undefined=True)
    return str(template.render(**context))
