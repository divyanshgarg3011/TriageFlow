import os
import tempfile
from pathlib import Path

# Tests always use the offline model, no delays, and a throwaway database.
os.environ["DEMO_MODE"] = "1"
os.environ["DEMO_STEP_DELAY"] = "0"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")

import pytest  # noqa: E402

from app import db  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    db.init_db(reset=True)
    yield
