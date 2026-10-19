import os
from pathlib import Path
import tempfile
import unittest

from install import install


class InstallTests(unittest.TestCase):
    def test_venv_python_symlink_is_preserved_and_unrelated_launcher_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / 'system-python'
            actual.write_text('placeholder')
            venv = root / 'venv/bin/python'
            venv.parent.mkdir(parents=True)
            venv.symlink_to(actual)
            result = install(root / 'bin', root / 'skill', venv)
            launcher = Path(result['command'])
            self.assertIn(str(venv), launcher.read_text())
            self.assertNotIn(str(actual), launcher.read_text())
            self.assertTrue(os.access(launcher, os.X_OK))
            self.assertNotIn('__PAPERBANANA_COMMAND__', Path(result['skill']).read_text())
            launcher.write_text('#!/bin/sh\necho unrelated\n')
            with self.assertRaises(ValueError):
                install(root / 'bin', root / 'skill', venv)


if __name__ == '__main__':
    unittest.main()
