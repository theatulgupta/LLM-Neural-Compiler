"""Class-name lookup for detector status JSON."""

from nnc.labels import class_name, labels_for_task


def test_coco80_person_and_car() -> None:
    assert class_name("detect", 0) == "person"
    assert class_name("detect", 2) == "car"
    assert class_name("segment", 2) == "car"


def test_ssdlite_dining_table_is_67() -> None:
    assert class_name("detect-lite", 67) == "dining table"
    assert labels_for_task("detect-lite")[0] == "__background__"


def test_pose_is_person() -> None:
    assert class_name("pose", 0) == "person"
    assert class_name("classify", 7) == "7"
