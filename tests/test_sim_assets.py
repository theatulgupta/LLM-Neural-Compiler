from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_nnc_yard_world() -> None:
    tree = ET.parse(ROOT / "sim" / "worlds" / "nnc_yard.sdf")
    world = tree.getroot().find("world")
    assert world is not None
    assert world.attrib.get("name") == "nnc_yard"


def test_nnc_camera_model() -> None:
    tree = ET.parse(ROOT / "sim" / "models" / "nnc_mono_cam" / "model.sdf")
    cam = tree.find(".//sensor[@type='camera']/camera/image")
    assert cam is not None
    assert cam.findtext("width") == "640"
    assert cam.findtext("height") == "480"
    rate = tree.find(".//sensor[@type='camera']/update_rate")
    assert rate is not None and rate.text == "10"


def test_nnc_x500_cam_name() -> None:
    tree = ET.parse(ROOT / "sim" / "models" / "nnc_x500_cam" / "model.sdf")
    model = tree.getroot().find("model")
    assert model is not None
    assert model.attrib.get("name") == "nnc_x500_cam"


def test_server_config_placeholder() -> None:
    text = (ROOT / "sim" / "server.config").read_text(encoding="utf-8")
    assert "gz-sim-sensors-system" in text
    assert "@RENDER_ENGINE@" in text
    assert "libOpticalFlowSystem" not in text
    assert "libGstCameraSystem" not in text
