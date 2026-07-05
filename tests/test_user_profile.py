import os

from uchi.user_profile import load_profile, remember_preference


def test_load_profile_missing_file_returns_empty(tmp_path):
    path = str(tmp_path / "user_profile.md")
    assert load_profile(path) == ""


def test_remember_preference_creates_file(tmp_path):
    path = str(tmp_path / "sub" / "user_profile.md")
    remember_preference("I prefer Python 3.10", path=path)
    assert os.path.exists(path)
    content = load_profile(path)
    assert "I prefer Python 3.10" in content


def test_remember_preference_appends_not_overwrites(tmp_path):
    path = str(tmp_path / "user_profile.md")
    remember_preference("first preference", path=path)
    remember_preference("second preference", path=path)
    content = load_profile(path)
    assert "first preference" in content
    assert "second preference" in content
    assert content.count("\n") == 2
