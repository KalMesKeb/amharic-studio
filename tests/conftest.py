"""Shared fixtures. No Qt here: the core has to stay testable headless."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amharic_studio.core.confusion import ConfusionModel  # noqa: E402
from amharic_studio.core.lexicon import Lexicon  # noqa: E402
from amharic_studio.core.project import Project  # noqa: E402
from amharic_studio.core.suggest import Suggester  # noqa: E402
from amharic_studio.data.seed_words import SEED_WORDS  # noqa: E402

#: Words the tests treat as unambiguously part of the language, on top of the seed list.
EXTRA_WORDS = (
    "ኢትዮጵያ",
    "ቤት",
    "ሰው",
    "መንግሥት",
    "ተማሪ",
    "አገር",
    "መጽሐፍ",
    "ልጅ",
    "ከተማ",
    "ታሪክ",
    "ቋንቋ",
    "ትምህርት",
    "ሰላም",
)


@pytest.fixture
def lexicon() -> Lexicon:
    lex = Lexicon(":memory:")
    lex.add_many(SEED_WORDS, source="seed")
    for word in EXTRA_WORDS:
        lex.add(word, freq=50, source="test")
    return lex


@pytest.fixture
def model() -> ConfusionModel:
    return ConfusionModel()


@pytest.fixture
def suggester(lexicon: Lexicon, model: ConfusionModel) -> Suggester:
    return Suggester(lexicon, model)


@pytest.fixture
def project(tmp_path: Path) -> Project:
    proj = Project.create(tmp_path / "Test.amproj", title="ሙከራ", author="ደራሲ")
    yield proj
    proj.close()
