#!/usr/bin/env python3
"""Install a Codex skill and launcher for an existing PaperBanana checkout."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
MARKER = '# PaperBanana Codex subscription launcher'


def install(bin_dir, skill_dir, python):
    # Resolving a venv's python symlink would silently select the system Python.
    python = python.expanduser().absolute()
    if not python.is_file():
        raise ValueError(f'Python not found: {python}. Create the repository .venv first.')
    launcher = bin_dir.expanduser().resolve() / 'paperbanana-author'
    skill_dir = skill_dir.expanduser().resolve()
    skill_file = skill_dir / 'SKILL.md'
    if launcher.exists() and MARKER not in launcher.read_text():
        raise ValueError(f'Refusing to replace an unrelated command: {launcher}')
    if skill_file.exists() and 'name: paperbanana-author' not in skill_file.read_text():
        raise ValueError(f'Refusing to replace an unrelated skill: {skill_file}')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for path in (launcher, skill_file):
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + '.backup-' + stamp))
    launcher.parent.mkdir(parents=True, exist_ok=True)
    skill_dir.mkdir(parents=True, exist_ok=True)
    launcher.write_text('#!/bin/sh\n' + MARKER + '\nexec ' + shlex.quote(str(python)) + ' ' +
                        shlex.quote(str(ROOT / 'integrations/codex/paperbanana_codex.py')) + ' "$@"\n')
    launcher.chmod(0o755)
    source = (ROOT / 'integrations/codex/SKILL.md').read_text()
    skill_file.write_text(source.replace('__PAPERBANANA_COMMAND__', shlex.quote(str(launcher))))
    return {'command': str(launcher), 'skill': str(skill_file), 'repository': str(ROOT)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bin-dir', type=Path, default=Path.home() / '.local/bin')
    home = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')))
    p.add_argument('--skill-dir', type=Path, default=home / 'skills/paperbanana-author')
    p.add_argument('--python', type=Path, default=ROOT / '.venv/bin/python')
    args = p.parse_args()
    try:
        result = install(args.bin_dir, args.skill_dir, args.python)
    except (ValueError, OSError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    import json
    print(json.dumps(result, indent=2))
    print('Open a new Codex chat to discover $paperbanana-author. Run the command with doctor to verify.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
