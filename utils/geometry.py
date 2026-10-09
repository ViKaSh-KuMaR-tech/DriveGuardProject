"""Pure geometry helpers (no OpenCV / MediaPipe dependency, easy to unit-test)."""
from __future__ import annotations

import numpy as np


def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """Straight-line distance between two 2-D points."""
    return float(np.linalg.norm(np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """
    Eye Aspect Ratio (Soukupova & Cech, 2016).

    `eye` is a (6, 2) array of landmarks ordered p1..p6:
        p1 = outer corner,  p4 = inner corner  (horizontal axis)
        p2, p3 = upper lid, p6, p5 = lower lid (p2<->p6 and p3<->p5 are vertical pairs)

                 p2   p3
           p1                p4
                 p6   p5

        EAR = ( |p2-p6| + |p3-p5| ) / ( 2 * |p1-p4| )

    Open eye: ~0.25-0.35. Closed eye: the vertical distances collapse, EAR -> ~0.
    Dividing by the eye width makes the value independent of how far the driver
    is from the camera.
    """
    vertical_a = euclidean(eye[1], eye[5])
    vertical_b = euclidean(eye[2], eye[4])
    horizontal = euclidean(eye[0], eye[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """
    Mouth Aspect Ratio using the INNER lip contour.

    `mouth` is an (8, 2) array ordered:
        [left_corner, right_corner,
         upper_1, lower_1, upper_2, lower_2, upper_3, lower_3]

        MAR = ( |u1-l1| + |u2-l2| + |u3-l3| ) / ( 3 * |left_corner - right_corner| )

    Closed mouth: ~0.0-0.1. Talking: ~0.2-0.4. Yawning (wide open): >0.5-0.6.
    Averaging three vertical gaps makes it less sensitive to one noisy landmark.
    """
    vertical = (
        euclidean(mouth[2], mouth[3])
        + euclidean(mouth[4], mouth[5])
        + euclidean(mouth[6], mouth[7])
    )
    horizontal = euclidean(mouth[0], mouth[1])
    if horizontal < 1e-6:
        return 0.0
    return vertical / (3.0 * horizontal)
