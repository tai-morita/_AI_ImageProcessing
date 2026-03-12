from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import NamedTuple

import numpy as np
from scipy.ndimage import rotate

class VolumeData(NamedTuple):
    volume: np.ndarray
    px_size: float
    first_pos: np.ndarray
    rot_angles: list[float]
    fov_area: np.ndarray


def load_volume_from_raw(content: bytes) -> VolumeData:
    """Parse proprietary RAW bytes and reconstruct 3D volume."""
    len_of_xml = int.from_bytes(content[21:23], "little")
    volxml = content[25 : len_of_xml + 25].decode("ASCII")
    root = ET.fromstring(volxml)

    xmin = int(root.find("FYI/XMin").attrib["value"])
    xmax = int(root.find("FYI/XMax").attrib["value"])
    ymin = int(root.find("FYI/YMin").attrib["value"])
    ymax = int(root.find("FYI/YMax").attrib["value"])
    zmin = int(root.find("FYI/ZMin").attrib["value"])
    zmax = int(root.find("FYI/ZMax").attrib["value"])

    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1

    hu_available_elem = root.find("Attribute/bIsHuAvailable")
    hu_available = False
    if hu_available_elem is not None:
        hu_available = int(hu_available_elem.attrib["value"]) == 1

    if hu_available:
        hu_inter = int(root.find("Attribute/tfSystemV2HuIntercept").attrib["value"])
        hu_slope = int(root.find("Attribute/tfSystemV2HuSlope").attrib["value"])
    else:
        hu_inter = 0
        hu_slope = 100

    px_size = float(root.find("Attribute/tfXGridSize").attrib["value"])

    rot_angles = [
        float(root.find("Attribute/tfAntiAliasAngleInRadian").attrib["value"])
        / np.pi
        * 180
        + float(root.find("Attribute/tfInitialAngleInRadian").attrib["value"])
        / np.pi
        * 180,
        float(root.find("Attribute/tfAntiAliasAngleInDegree").attrib["value"]) + 90,
    ]

    t_a = float(root.find("Attribute/tfA").attrib["value"])
    t_b = float(root.find("Attribute/tfB").attrib["value"])

    total_len = xlen * ylen * zlen
    vol = np.frombuffer(
        content[len_of_xml + 61 : len_of_xml + 61 + total_len * 2], dtype=np.int16
    )
    vol = vol.reshape((xlen, ylen, zlen))
    vol = vol.transpose((2, 1, 0))
    vol = (np.array(vol, dtype=np.float32) * t_a + t_b) * hu_slope + hu_inter

    fov_area = vol != vol[0, 0, 0]
    vol[vol == vol[0, 0, 0]] = 0

    return VolumeData(
        volume=vol,
        px_size=px_size,
        first_pos=np.array([xmin, ymin, zmin]) * px_size,
        rot_angles=rot_angles,
        fov_area=fov_area,
    )


def choose_rotation_angle(volume: np.ndarray, px_size: float, rot_angles: list[float]) -> float:
    fov_size = volume.shape[1] * px_size
    return rot_angles[0] if fov_size <= 41 else rot_angles[1]


def rotate_volume_for_ai(volume: np.ndarray, angle: float) -> np.ndarray:
    return np.array(
        rotate(volume, angle, axes=(1, 2), reshape=False),
        dtype=np.float32,
    )


def estimate_contrast(volume: np.ndarray) -> float:
    valid = volume[(volume > 300) * (volume < 2000)]
    if valid.size == 0:
        return 1.0
    contrast = float(np.std(valid))
    return contrast if contrast > 0 else 1.0
