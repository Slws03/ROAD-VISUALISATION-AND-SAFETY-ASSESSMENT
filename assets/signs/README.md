# Sign crops for synthetic scenarios

Cut from the Malaysia road sign dataset (Roboflow, CC BY 4.0,
https://universe.roboflow.com/steven-ling-wei-seng/malaysia-road-sign-dataset-ssvcw)
using its own bounding boxes.

They exist so a drive with no warning signage can be given some, to exercise the
gap-check decision layer end to end. Any report built on a route they were pasted
into is a **synthetic scenario**, not a detection result.

| File | Class | Size | Notes |
| --- | --- | --- | --- |
| `bend.png` | Bend | 237x229 | Diamond sign; paste with the `diamond` mask so its corners take the destination background |
| `chevron.png` | Chevron | 48x73 | Cropped with surrounding post and background on purpose |

The context matters. A chevron cut tight to its own edge is detected erratically
-- 0.64 at 80 px, nothing at 110 or 150. The same sign cropped with its
surroundings is detected monotonically: 0.24, 0.65, 0.81, 0.86, 0.86 at 40, 55,
70, 90 and 120 px. The detector needs to see the sign's boundary against
something.
