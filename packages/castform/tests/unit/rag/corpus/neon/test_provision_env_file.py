"""Security + correctness of provision.write_env_file (0600 atomic secret write).

Guards the temp-file hardening: the DSN env file is written via a fresh mkstemp in
the destination dir (never a predictable ``${path}.tmp``), so a symlink or
pre-create at the old predictable path cannot capture or truncate the secret write.
"""

from __future__ import annotations

import os
import stat

from castform.rag.corpus.neon.provision import write_env_file


def test_writes_values_at_mode_0600(tmp_path) -> None:
    env = tmp_path / "neon.env"
    write_env_file(str(env), {"NEON_CORPUS_DSN_RO": "postgresql://ro@h/db?a=1&b=2"})
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert 'NEON_CORPUS_DSN_RO="postgresql://ro@h/db?a=1&b=2"' in env.read_text()


def test_merges_existing_keys_and_preserves_unrelated_lines(tmp_path) -> None:
    env = tmp_path / "neon.env"
    env.write_text('NEON_CORPUS_DSN_RO="old"\nUNRELATED="keep"\n')
    write_env_file(str(env), {"NEON_CORPUS_DSN_RO": "new"})
    text = env.read_text()
    assert 'NEON_CORPUS_DSN_RO="new"' in text
    assert 'UNRELATED="keep"' in text


def test_no_predictable_tmp_file_left_behind(tmp_path) -> None:
    env = tmp_path / "neon.env"
    write_env_file(str(env), {"NEON_PROJECT_ID": "p"})
    assert not (tmp_path / "neon.env.tmp").exists()


def test_symlink_at_predictable_tmp_path_is_not_followed(tmp_path) -> None:
    # The old code opened ``${path}.tmp`` with O_CREAT|O_TRUNC (no O_EXCL/O_NOFOLLOW):
    # a symlink pre-created there would be followed and the victim truncated/written.
    victim = tmp_path / "victim"
    victim.write_text("SECRET_UNTOUCHED")
    env = tmp_path / "neon.env"
    os.symlink(victim, f"{env}.tmp")

    write_env_file(str(env), {"NEON_CORPUS_DSN_RW": "postgresql://rw@h/db"})

    # The victim behind the predictable-name symlink is never written through.
    assert victim.read_text() == "SECRET_UNTOUCHED"
    assert 'NEON_CORPUS_DSN_RW="postgresql://rw@h/db"' in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
