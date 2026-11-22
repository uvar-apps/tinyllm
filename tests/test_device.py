from __future__ import annotations

from pathlib import Path

import pytest

from tinyllm import device


def _topology(root: Path, gfx: int, name: str = "ip discovery") -> Path:
    for node, version in ((0, 0), (1, gfx)):
        d = root / str(node)
        d.mkdir(parents=True)
        (d / "name").write_text("" if version == 0 else name + "\n")
        (d / "properties").write_text(
            f"cpu_cores_count 16\nsimd_count 80\nsimd_per_cu 2\ngfx_target_version {version}\n"
        )
    return root


def test_detects_strix_halo(tmp_path: Path) -> None:
    gpus = device.detect_amd_gpus(_topology(tmp_path, 110501))
    assert len(gpus) == 1
    assert gpus[0].arch == "gfx1151" and gpus[0].compute_units == 40 and gpus[0].supported


def test_old_gpu_is_not_supported(tmp_path: Path) -> None:
    assert not device.detect_amd_gpus(_topology(tmp_path, 100300))[0].supported


def test_missing_topology_means_no_gpus(tmp_path: Path) -> None:
    assert device.detect_amd_gpus(tmp_path / "absent") == []


def test_auto_picks_kfd_amd_when_usable(tmp_path: Path) -> None:
    kfd = tmp_path / "kfd"
    kfd.write_text("")
    topo = _topology(tmp_path / "topo", 110501)
    assert device.resolve("auto", {}, topo, kfd) == "KFD+AMD"
    assert device.resolve("auto", {}, topo, tmp_path / "no-kfd") == "CPU"
    assert device.resolve("auto", {}, _topology(tmp_path / "old", 100300), kfd) == "CPU"


def test_aliases() -> None:
    assert device.resolve("cpu") == "CPU"
    assert device.resolve("CPU") == "CPU"
    assert device.resolve("nv") == "NV"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("AMD", "KFD+AMD"),
        ("AMD:LLVM", "KFD+AMD:LLVM"),
        ("KFD+AMD", "KFD+AMD"),
        ("KFD:0+AMD::gfx1151", "KFD:0+AMD::gfx1151"),
        ("CPU:LLVM", "CPU:LLVM"),
        ("CPU;AMD", "CPU;KFD+AMD"),
    ],
)
def test_harden_pins_amd_to_kfd(given: str, expected: str) -> None:
    assert device.harden(given, {}) == expected


@pytest.mark.parametrize("dev", ["PCI+AMD", "USB+AMD:LLVM", "pci+amd", "PCI+", "CPU;PCI+AMD"])
def test_harden_refuses_userspace_drivers(dev: str) -> None:
    with pytest.raises(device.DeviceConfigError, match="userspace"):
        device.harden(dev, {})


def test_harden_allows_userspace_when_asked() -> None:
    assert device.harden("PCI+AMD", {device.ENV_ALLOW_USERSPACE: "1"}) == "PCI+AMD"


def test_raw_dev_strings_are_hardened_by_resolve() -> None:
    assert device.resolve("AMD:HIP", {}) == "KFD+AMD:HIP"
    with pytest.raises(device.DeviceConfigError):
        device.resolve("PCI+AMD", {})


def test_tests_run_on_cpu() -> None:
    # conftest selects the CPU before anything imports tinygrad
    assert device.configure() == "CPU"
    with pytest.raises(device.DeviceConfigError, match="already configured"):
        device.configure("nv")
