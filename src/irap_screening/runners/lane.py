"""Convert a raw CLRerNet dump into canonical lane detections.

CLRerNet runs in an mmcv-pinned environment that this project deliberately does
not reproduce, so inference happens elsewhere (Colab) and only the numbers come
back. The split is on purpose: the Colab side does nothing but serialise
`lane.points` and scores, while every piece of arithmetic that could be
plausibly wrong lives here, where it is tested.

Coordinate mapping is the whole risk. CLRerNet's head postprocesses with its own
test_cfg geometry, not with the tensor it was handed, so its normalised points
are expressed in a virtual frame whose top `cut_height` rows were cut:

    x_norm in [0, 1]                        across the full image width
    y_norm in [cut_height / ori_img_h, 1]   across the retained rows

The runner fed it the crop rows [crop_top .. 1 - crop_bottom] of the original
image, so that y range maps back onto the crop, and the inverse is

    x = x_norm * source_w
    y = crop_top_px + (y_norm - cut_ratio) / (1 - cut_ratio) * canvas_h

This was verified against the annotated overlays the notebook produced: the
polylines sit on the markings. The competing reading -- y_norm normalised over
the whole original frame -- differs by a factor of 0.73 and would have been
obvious in those images.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from irap_screening.io.schema import DetectionRecord, Frame, LaneInstance, Transform
from irap_screening.io.sources.fixture import FixtureSource

# The fixed network input size; the crop is resized to this before inference.
MODEL_WH = (800, 320)


class RawLaneMeta(BaseModel):
    """Geometry the Colab run actually used, carried with the data.

    These are editable cells in the notebook, and the crop lives in a different
    cell from test_cfg. Reading them from the dump rather than hardcoding them
    means a re-run with a different crop cannot be silently mapped by stale
    constants.
    """

    model_config = {"extra": "allow"}

    crop_top: float = Field(ge=0.0, lt=1.0)
    crop_bottom: float = Field(ge=0.0, lt=1.0)
    cut_height: int = Field(ge=0)
    ori_img_h: int = Field(gt=0)
    ori_img_w: int = Field(gt=0)
    conf_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _cut_height_matches_crop(self) -> RawLaneMeta:
        """The inverse mapping is only valid when the model's cut equals the crop.

        CLRerNet spreads its output over the rows it believes were kept. If
        cut_height/ori_img_h and crop_top disagree, the points are stretched
        against the image by exactly their ratio -- a wrong answer that still
        looks like lanes.
        """
        cut_ratio = self.cut_height / self.ori_img_h
        if abs(cut_ratio - self.crop_top) > 1e-3:
            raise ValueError(
                f"cut_height/ori_img_h ({cut_ratio:.4f}) must equal crop_top "
                f"({self.crop_top:.4f}); CLRerNet maps its output onto the rows it "
                "believes were cut, so a mismatch stretches every lane vertically"
            )
        return self


class RawLane(BaseModel):
    conf: float = Field(ge=0.0, le=1.0)
    points_norm: list[tuple[float, float]]


class RawFrame(BaseModel):
    filename: str
    source_wh: tuple[int, int]
    lanes: list[RawLane] = Field(default_factory=list)


class RawLaneDump(BaseModel):
    meta: RawLaneMeta
    frames: list[RawFrame]


def crop_geometry(source_wh: tuple[int, int], meta: RawLaneMeta) -> tuple[int, int]:
    """Return (crop_top_px, canvas_h), reproducing the notebook's integer truncation.

    Float arithmetic here would shift points a pixel or two away from the
    overlays the mapping was validated against.
    """
    height = source_wh[1]
    crop_top_px = int(height * meta.crop_top)
    crop_bottom_px = int(height * meta.crop_bottom)
    crop_end = height - crop_bottom_px if crop_bottom_px > 0 else height
    return crop_top_px, crop_end - crop_top_px


def to_original_points(
    points_norm: list[tuple[float, float]],
    source_wh: tuple[int, int],
    meta: RawLaneMeta,
) -> list[tuple[float, float]]:
    """Map CLRerNet's normalised points back to original-image pixels."""
    width = source_wh[0]
    crop_top_px, canvas_h = crop_geometry(source_wh, meta)
    cut_ratio = meta.cut_height / meta.ori_img_h
    span = 1.0 - cut_ratio
    return [
        (x * width, crop_top_px + (y - cut_ratio) / span * canvas_h)
        for x, y in points_norm
    ]


def build_record(frame: Frame, raw: RawFrame, meta: RawLaneMeta) -> DetectionRecord:
    """Assemble one frame's lanes into a canonical record."""
    instances = [
        LaneInstance(
            id=f"{frame.frame_id}#lane#{index}",
            points=to_original_points(lane.points_norm, raw.source_wh, meta),
            conf=lane.conf,
        )
        for index, lane in enumerate(raw.lanes)
    ]
    return DetectionRecord(
        frame_id=frame.frame_id,
        image_path=frame.path,
        sha256=frame.sha256,
        type="lane",
        transform=Transform(
            source_wh=raw.source_wh,
            crop_top=meta.crop_top,
            crop_bottom=meta.crop_bottom,
            model_wh=MODEL_WH,
        ),
        instances=instances,
    )


def _pair_with_route(frames: list[Frame], dump: RawLaneDump) -> list[tuple[Frame, RawFrame]]:
    """Match raw entries to route frames by image basename, in both directions.

    The dump drives this conversion, not frames.csv, so a Colab run that stopped
    halfway would otherwise yield a lane.jsonl silently covering part of the
    route. A frame with no lane record contributes no contrast samples, so a
    partial file reads as a stretch of unmarked road.
    """
    by_name = {Path(frame.path).name: frame for frame in frames}
    raw_by_name = {raw.filename: raw for raw in dump.frames}

    unknown = sorted(set(raw_by_name) - set(by_name))
    if unknown:
        raise ValueError(
            f"{len(unknown)} dump entries match no row in frames.csv, "
            f"first few: {unknown[:5]}"
        )
    missing = sorted(set(by_name) - set(raw_by_name))
    if missing:
        raise ValueError(
            f"{len(missing)} route frames have no entry in the dump, "
            f"first few: {missing[:5]}; a partial run would read as unmarked road"
        )
    return [(by_name[name], raw_by_name[name]) for name in by_name]


def export(raw_path: Path, route_root: Path, out_path: Path | None = None) -> Path:
    """Read a CLRerNet dump and write detections/lane.jsonl for a route."""
    raw_path, route_root = Path(raw_path), Path(route_root)
    dump = RawLaneDump.model_validate(json.loads(raw_path.read_text(encoding="utf-8")))

    if not any(raw.lanes for raw in dump.frames):
        raise ValueError(
            f"{raw_path} contains no lanes on any of its {len(dump.frames)} frames. "
            "A frame with no lanes yields no contrast samples and grades as the "
            "worst band, so an empty dump would report the whole route critical."
        )

    pairs = _pair_with_route(FixtureSource(route_root).frames(), dump)

    out_path = Path(out_path) if out_path else route_root / "detections" / "lane.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for frame, raw in pairs:
            record = build_record(frame, raw, dump.meta)
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")
    return out_path
