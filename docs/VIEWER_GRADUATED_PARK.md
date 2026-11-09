# Historical graduated obstacle park — original v5032 browser fixture

**Archive context, 31 October 2026.** This describes the v5032 260M browser obstacle park, not the H40 120M source currently on `main`. The original independent v5032 data must not be attributed to H40. The [current H40 viewer guide](PLAYGROUND.md) is authoritative for the present Field/Park, fixed eight-prism Carpet and separate untrained sensor lab.

## Why this historical course was introduced

The earlier park put an unsupported crest and large obstacles (5–8 cm) in front of an actor that had only proven some smaller physical step crossings. A plausible walking animation could therefore conceal a real inability to traverse even the first obstacle.

The historical park had eight deterministic MuJoCo support challenges, with physical collision geometry rather than decorations:

| # | Feature | Gentle | Balanced | Stretch |
|---|---|---:|---:|---:|
| 1 | Raised step / landing | 1.2 cm | 2.0 cm | 3.0 cm |
| 2 | Camber ridge | 0.9 cm | 1.5 cm | 3.0 cm |
| 3 | Joined rise, flat crest, descent | 1.5 cm | 2.5 cm | 5.0 cm |
| 4 | Diagonal ridge | 0.9 cm | 1.5 cm | 3.0 cm |
| 5 | Ramp into sharp step down | 1.5 cm | 2.5 cm | 3.5 cm |
| 6 | Two shallow ripples | 0.6 cm | 1.2 cm | 2.4 cm |
| 7 | Rounded log | 1.5 cm | 2.4 cm | 4.0 cm |
| 8 | Two banks with a valley | 0.7 cm | 1.4 cm | 2.5 cm |

These were **browser scene parameters**, not certified robot limits. Autonomous navigation may steer around a challenge, and a single successful CPU MuJoCo attempt is not a matched MJX exam.

The old v5032 baseline achieved some independent 2 cm supported crossings but not robust 3 cm ascent. The old [v5032 release card](V5032_PINNACLE_RELEASE.md) holds the measured details, and its exact original source is recoverable from Git history. The H40 trainer now uses four recycled **1 m** tiles with eight convex prism colliders; its browser Carpet is a **fixed** 2×2 m diagnostic fixture. Use the [current handover](../handover.md) and [physical-quality contract](PHYSICAL_QUALITY.md) for next experiments.
