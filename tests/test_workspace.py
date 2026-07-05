import os

import pytest

from uchi.workspace import (
    WorkspaceViolation,
    delete_file,
    list_files,
    read_file,
    resolve_path,
    write_file,
)


def test_write_read_roundtrip(tmp_path):
    root = str(tmp_path / "sandbox")
    write_file("notes/todo.txt", "hello sandbox", root=root)
    assert read_file("notes/todo.txt", root=root) == "hello sandbox"


def test_list_files(tmp_path):
    root = str(tmp_path / "sandbox")
    write_file("a.txt", "a", root=root)
    write_file("b.txt", "b", root=root)
    assert list_files(root=root) == ["a.txt", "b.txt"]


def test_delete_file(tmp_path):
    root = str(tmp_path / "sandbox")
    write_file("gone.txt", "x", root=root)
    delete_file("gone.txt", root=root)
    assert list_files(root=root) == []


def test_relative_traversal_is_blocked(tmp_path):
    root = str(tmp_path / "sandbox")
    os.makedirs(root, exist_ok=True)
    with pytest.raises(WorkspaceViolation):
        resolve_path("../../etc/passwd", root=root)


def test_absolute_path_outside_root_is_blocked(tmp_path):
    root = str(tmp_path / "sandbox")
    os.makedirs(root, exist_ok=True)
    outside = str(tmp_path / "outside.txt")
    with pytest.raises(WorkspaceViolation):
        resolve_path(outside, root=root)


def test_symlink_escape_is_blocked(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    outside_target = tmp_path / "secret.txt"
    outside_target.write_text("secret")
    link = root / "escape"
    link.symlink_to(outside_target)
    with pytest.raises(WorkspaceViolation):
        resolve_path("escape", root=str(root))


def test_absolute_path_inside_root_is_allowed(tmp_path):
    root = str(tmp_path / "sandbox")
    os.makedirs(root, exist_ok=True)
    inside_abs = os.path.join(os.path.realpath(root), "file.txt")
    resolved = resolve_path(inside_abs, root=root)
    assert resolved == inside_abs
