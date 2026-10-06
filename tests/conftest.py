import os
import tempfile
from pathlib import Path

# Tests use a throwaway database; this must run before any llmath module is imported.
os.environ["DATABASE_URL"] = f"sqlite:///{(Path(tempfile.mkdtemp(prefix='llmath_test_')) / 'test.db').as_posix()}"
