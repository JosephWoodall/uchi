from uchi.scratchpad import run_python


def test_stdout_capture(tmp_path):
    result = run_python("print('hello scratchpad')", root=str(tmp_path))
    assert result.ok
    assert result.stdout.strip() == "hello scratchpad"
    assert result.returncode == 0


def test_stderr_and_nonzero_exit_on_exception(tmp_path):
    result = run_python("raise ValueError('boom')", root=str(tmp_path))
    assert not result.ok
    assert result.returncode != 0
    assert "boom" in result.stderr


def test_arbitrary_code_no_run_wrapper_required(tmp_path):
    code = (
        "import statistics\n"
        "data = [1, 2, 3, 4, 5]\n"
        "print(statistics.mean(data))\n"
    )
    result = run_python(code, root=str(tmp_path))
    assert result.ok
    assert result.stdout.strip() == "3"


def test_timeout(tmp_path):
    result = run_python("import time\ntime.sleep(5)", timeout=0.5, root=str(tmp_path))
    assert not result.ok
    assert result.stderr == "TimeoutExpired"


def test_as_text_includes_exit_code_and_streams(tmp_path):
    result = run_python("print('out')\nimport sys\nsys.stderr.write('err')", root=str(tmp_path))
    text = result.as_text()
    assert "[exit code: 0]" in text
    assert "out" in text
    assert "err" in text


def test_scratch_file_cleaned_up(tmp_path):
    run_python("print(1)", root=str(tmp_path))
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("_scratch_")]
    assert leftovers == []
