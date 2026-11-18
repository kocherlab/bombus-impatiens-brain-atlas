"""bimp_st.site: lookup order and the error for an unset key."""

import pytest

from bimp_st import site


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for k in site.KEYS:
        monkeypatch.delenv(site.env_name(k), raising=False)
    for names in site.ALIASES.values():
        for n in names:
            monkeypatch.delenv(n, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("BIMP_CONFIG", raising=False)
    monkeypatch.setattr(site, "REPO", tmp_path / "repo")
    return tmp_path


def test_default_is_repo_relative(isolated):
    assert site.get("data_repo") == site.Path(site.KEYS["data_repo"][0])


def test_unset_key_raises_naming_the_variable(isolated):
    with pytest.raises(site.SiteConfigError, match="BIMP_SNRNA_SOURCE"):
        site.get("snrna_source")


def test_lookup_order(isolated, monkeypatch):
    user = isolated / "xdg" / "bimp-st" / "site.toml"
    user.parent.mkdir(parents=True)
    user.write_text('snrna_source = "/from/user"\n')
    assert str(site.get("snrna_source")) == "/from/user"

    named = isolated / "named.toml"
    named.write_text('snrna_source = "/from/named"\n')
    monkeypatch.setenv("BIMP_CONFIG", str(named))
    assert str(site.get("snrna_source")) == "/from/named"

    monkeypatch.setenv("BIMP_SNRNA_SOURCE", "/from/env")
    assert str(site.get("snrna_source")) == "/from/env"


def test_alias_env_var(isolated, monkeypatch):
    monkeypatch.setenv("BIMP_MAIN_CHECKOUT", "/old/name")
    assert str(site.get("data_repo")) == "/old/name"


def test_unknown_key():
    with pytest.raises(KeyError, match="unknown site key"):
        site.lookup("nope")


def test_flat_reader_without_tomllib(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__

    def no_tomllib(name, *a, **k):
        if name == "tomllib":
            raise ModuleNotFoundError(name)
        return real(name, *a, **k)

    f = tmp_path / "site.toml"
    f.write_text('# comment\nsnrna_source = "/a/b"  # trailing\ndata_repo="/c"\n')
    monkeypatch.setattr(builtins, "__import__", no_tomllib)
    assert site._read_toml(f) == {"snrna_source": "/a/b", "data_repo": "/c"}
