import pytest

from uchi.data_silo import DataSilo, DataSiloViolation


def test_no_restrictions_allows_anything(tmp_path):
    silo = DataSilo()
    silo.check(str(tmp_path / "anything.txt"))  # must not raise


def test_denied_path_is_blocked(tmp_path):
    hr = tmp_path / "hr"
    hr.mkdir()
    silo = DataSilo(denied_paths=[str(hr)])
    with pytest.raises(DataSiloViolation):
        silo.check(str(hr / "salaries.csv"))


def test_path_outside_denied_root_is_fine(tmp_path):
    hr = tmp_path / "hr"
    hr.mkdir()
    finance = tmp_path / "finance"
    finance.mkdir()
    silo = DataSilo(denied_paths=[str(hr)])
    silo.check(str(finance / "report.csv"))  # must not raise


def test_allow_list_permits_only_listed_roots(tmp_path):
    finance = tmp_path / "finance"
    finance.mkdir()
    hr = tmp_path / "hr"
    hr.mkdir()
    silo = DataSilo(allowed_paths=[str(finance)])
    silo.check(str(finance / "q3.csv"))  # allowed, must not raise
    with pytest.raises(DataSiloViolation):
        silo.check(str(hr / "salaries.csv"))


def test_deny_takes_priority_over_allow(tmp_path):
    finance = tmp_path / "finance"
    finance.mkdir()
    secret = finance / "secret"
    secret.mkdir()
    silo = DataSilo(allowed_paths=[str(finance)], denied_paths=[str(secret)])
    silo.check(str(finance / "public.csv"))  # allowed
    with pytest.raises(DataSiloViolation):
        silo.check(str(secret / "private.csv"))  # denied overrides allow


def test_symlink_inside_allowed_dir_pointing_into_denied_dir_is_still_blocked():
    """The real security-relevant case: a symlink physically sitting in
    the allowed silo but pointing at denied data must resolve to its real
    target (via realpath) and be blocked -- checking the surface path
    alone would be a bypass."""
    import tempfile
    import pathlib

    tmp_path = pathlib.Path(tempfile.mkdtemp())
    hr = tmp_path / "hr"
    hr.mkdir()
    secret = hr / "salaries.csv"
    secret.write_text("secret data")

    finance = tmp_path / "finance"
    finance.mkdir()
    link = finance / "innocuous.csv"
    link.symlink_to(secret)

    silo = DataSilo(allowed_paths=[str(finance)], denied_paths=[str(hr)])
    with pytest.raises(DataSiloViolation):
        silo.check(str(link))
