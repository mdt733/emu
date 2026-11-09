"""Single physical VL53L8CX 4x4 geometry contract.

45° x 45° FoV is an image-plane frustum, not a separable pitch/yaw fan.
One centre ray per zone approximates the 4x4 sensor's real multizone
histogram return (NOT a complete optical or multipath simulation).
All directions are BODY-FRAME unit vectors; the chassis orientation maps
them into world space. ST 4x4 zone order is row-major in our actor's view.
"""
from __future__ import annotations
import math

TOF_FOV_DEG = 45.0
TOF_TILT_DOWN_DEG = 70.0
TOF_YAW_DEG = 0.0
TOF_MOUNT_BODY_M = (0.0, 0.0, -0.030)
TOF_ZONE_SIDE = 4
# ST UM3109 §2.2: the receiving lens flips BOTH physical image axes.
# SPAD zone 0 is at array bottom-left but observes scene TOP-RIGHT;
# therefore, for the nominal package orientation shown in the datasheet,
# our actor's scene row-major [top-left..bottom-right] maps to raw IDs:
# [3,2,1,0; 7,6,5,4; 11,10,9,8; 15,14,13,12].
# A differently rotated sensor PCB requires a measured remap; do NOT
# silently treat ascending hardware indices as scene row-major.
TOF_HARDWARE_ZONE_IDS = tuple(4*row+(3-col)
                              for row in range(4) for col in range(4))

def tof4x4_rays_body() -> tuple[tuple[float, float, float], ...]:
    """Match the browser's physical square-frustum zone CENTRE directions."""
    t=math.radians(TOF_TILT_DOWN_DEG)
    c,s=math.cos(t),math.sin(t)
    half=math.tan(math.radians(TOF_FOV_DEG)/2)
    out=[]
    for row in range(TOF_ZONE_SIDE):
        for col in range(TOF_ZONE_SIDE):
            u=(col-1.5)*half/2
            v=(1.5-row)*half/2
            x,y,z=c+s*v,u,-s+c*v
            inv=1.0/math.sqrt(x*x+y*y+z*z)
            out.append((x*inv,y*inv,z*inv))
    return tuple(out)

TOF_RAYS_BODY = tof4x4_rays_body()
