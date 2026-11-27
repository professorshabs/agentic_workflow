"""E2E integrity checks: only skips that always fire count, untouched authoritative files are
not failed for things nobody may change, and the product lint skips protected test files."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from atlas.pipeline import verify as verify_mod
from atlas.pipeline.build import materialize_files
from atlas.pipeline.verify import check_e2e_tests, e2e_file_hashes, unconditional_skip_lines
from atlas.schemas import CheckResult, GeneratedFile

CONDITIONAL = textwrap.dedent(
    """
    import shutil
    import pytest


    def test_print_to_pdf():
        chrome = shutil.which("chromium")
        if not chrome:
            pytest.skip("no local Chromium for print-to-PDF")
        try:
            pages = 1
        except OSError:
            pytest.skip("Chromium could not run")
        assert pages == 1


    @pytest.mark.skipif(sys.platform == "win32", reason="posix only")
    def test_posix():
        assert True
    """
)


def test_skips_inside_if_or_except_and_skipif_are_conditional():
    assert unconditional_skip_lines("tests/test_e2e.py", CONDITIONAL) == []


@pytest.mark.parametrize(
    ("source", "line"),
    [
        ("import pytest\n\n@pytest.mark.skip(reason='later')\ndef test_a():\n    assert 1\n", 3),
        ("import pytest\n\n@pytest.mark.skip\ndef test_a():\n    assert 1\n", 3),
        ("from pytest import mark\n\n@mark.skip\ndef test_a():\n    assert 1\n", 3),
        ("import pytest\npytestmark = pytest.mark.skip('all')\n\ndef test_a():\n    assert 1\n", 2),
        ("import pytest\n\ndef test_a():\n    pytest.skip('todo')\n    assert 1\n", 4),
        ("import pytest\npytest.skip('module', allow_module_level=True)\n", 2),
    ],
)
def test_skips_that_always_fire_are_found(source, line):
    assert unconditional_skip_lines("tests/test_e2e.py", source) == [line]


def test_a_helpers_fallback_skip_after_a_loop_is_conditional():
    source = textwrap.dedent(
        """
        import pytest


        def body_colour(css):
            for key, value in css.items():
                if key in value:
                    return value
            pytest.skip("colours not expressed as hex")


        def test_contrast():
            \"\"\"Text contrast.\"\"\"
            assert body_colour({"a": "a1"})
        """
    )
    assert unconditional_skip_lines("tests/test_e2e.py", source) == []


def test_javascript_skips_and_unparsable_python_fall_back_to_patterns():
    assert unconditional_skip_lines("tests/e2e.test.js", "it.skip('x', () => {});\n") == [1]
    broken = "def test_a(:\n    pytest.skip('x')\n"
    assert unconditional_skip_lines("tests/test_e2e.py", broken) == [2]


def _materialized(tmp_path: Path, content: str) -> list[GeneratedFile]:
    files = [GeneratedFile(path="tests/test_e2e_product.py", content=content)]
    materialize_files(tmp_path, files)
    return files


def test_untouched_file_with_conditional_skips_passes(tmp_path):
    files = _materialized(tmp_path, CONDITIONAL)
    result = check_e2e_tests(
        tmp_path, ["tests/test_e2e_product.py"], expected_hashes=e2e_file_hashes(files)
    )
    assert result.passed, result.stderr
    assert "skip" not in result.stdout


def test_untouched_file_is_not_failed_for_what_nobody_may_change(tmp_path):
    content = "import pytest\n\n@pytest.mark.skip\ndef test_a():\n    assert 1\n"
    files = _materialized(tmp_path, content)
    result = check_e2e_tests(
        tmp_path, ["tests/test_e2e_product.py"], expected_hashes=e2e_file_hashes(files)
    )
    assert result.passed
    assert "not failing" in result.stdout and "unconditional skip marker (line 3)" in result.stdout


def test_changed_file_still_fails_on_its_skips_and_its_hash(tmp_path):
    files = _materialized(tmp_path, "def test_a():\n    assert 1\n")
    (tmp_path / "tests" / "test_e2e_product.py").write_text(
        "import pytest\n\ndef test_a():\n    pytest.skip('x')\n    assert 1\n", encoding="utf-8"
    )
    result = check_e2e_tests(
        tmp_path, ["tests/test_e2e_product.py"], expected_hashes=e2e_file_hashes(files)
    )
    assert not result.passed
    assert "sha256 mismatch" in result.stderr
    assert "unconditional skip marker (line 4)" in result.stderr


def test_without_hashes_the_content_checks_still_fail(tmp_path):
    _materialized(tmp_path, "import pytest\n\ndef test_a():\n    pytest.skip('x')\n")
    result = check_e2e_tests(tmp_path, ["tests/test_e2e_product.py"])
    assert not result.passed and "unconditional skip marker" in result.stderr


@pytest.mark.asyncio
async def test_product_lint_skips_protected_e2e_files(tmp_path, monkeypatch):
    seen: dict[str, list[str]] = {}

    async def fake_run(name, argv, cwd, timeout, env=None):
        seen[name] = list(argv)
        return CheckResult(name=name, command=" ".join(argv), passed=True, exit_code=0)

    monkeypatch.setattr(verify_mod, "_run_subprocess", fake_run)
    monkeypatch.setattr(verify_mod, "_find_tool", lambda name: "/usr/bin/ruff")

    checks = await verify_mod._run_python_checks(
        tmp_path, sys.executable, 10.0, {}, ["tests/test_e2e_product.py"]
    )

    argv = seen["python-ruff"]
    assert argv[argv.index("--extend-exclude") + 1] == "tests/test_e2e_product.py"
    assert argv[-1] == "."
    ruff = next(c for c in checks if c.name == "python-ruff")
    assert "not linted: authoritative E2E test file(s) tests/test_e2e_product.py" in ruff.stdout

    seen.clear()
    await verify_mod._run_python_checks(tmp_path, sys.executable, 10.0, {})
    assert "--extend-exclude" not in seen["python-ruff"]
