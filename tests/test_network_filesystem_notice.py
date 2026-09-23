"""cytome.open says, once, when a store is on a network file system."""
import io
import pathlib

import pytest

from cytome.utils.storage import filesystem_type, is_network_filesystem, network_filesystem_notice


def _mounts(tmp_path, rows):
    p = tmp_path / "mounts"
    p.write_text("".join(f"{dev} {mnt} {typ} rw 0 0\n" for dev, mnt, typ in rows))
    return str(p)


def test_longest_mount_prefix_wins_and_symlinks_resolve(tmp_path):
    real = tmp_path / "vol" / "store.cytome"; real.parent.mkdir(); real.write_text("")
    link = tmp_path / "home_link.cytome"; link.symlink_to(real)
    mounts = _mounts(tmp_path, [("/dev/sda1", "/", "ext4"),
                                ("srv:/export", str(tmp_path / "vol"), "nfs4"),
                                ("/dev/sdb1", str(tmp_path), "xfs")])
    assert filesystem_type(real, mounts=mounts) == "nfs4"
    assert filesystem_type(link, mounts=mounts) == "nfs4"          # resolved through the link
    assert filesystem_type(tmp_path / "other.cytome", mounts=mounts) == "xfs"


@pytest.mark.parametrize("fstype,network", [("nfs", True), ("nfs4", True), ("lustre", True),
                                            ("gpfs", True), ("fuse.sshfs", True), ("cifs", True),
                                            ("ext4", False), ("xfs", False), ("tmpfs", False),
                                            ("overlay", False), (None, False)])
def test_which_types_count_as_network(fstype, network):
    assert is_network_filesystem(fstype) is network


def test_notice_prints_once_per_store_and_not_for_local(tmp_path):
    net = tmp_path / "net" / "a.cytome"; net.parent.mkdir(); net.write_text("")
    loc = tmp_path / "loc" / "b.cytome"; loc.parent.mkdir(); loc.write_text("")
    mounts = _mounts(tmp_path, [("/dev/sda1", "/", "ext4"), ("srv:/x", str(tmp_path / "net"), "nfs")])
    out = io.StringIO()
    assert network_filesystem_notice(net, mounts=mounts, stream=out) is True
    assert network_filesystem_notice(net, mounts=mounts, stream=out) is False   # once
    assert network_filesystem_notice(loc, mounts=mounts, stream=out) is False   # local: silent
    text = out.getvalue()
    assert text.count("network file system") == 1 and "$TMPDIR" in text and "a.cytome" in text


def test_unreadable_mount_table_stays_silent(tmp_path):
    f = tmp_path / "c.cytome"; f.write_text("")
    assert filesystem_type(f, mounts=str(tmp_path / "missing")) is None
    assert network_filesystem_notice(f, mounts=str(tmp_path / "missing")) is False
