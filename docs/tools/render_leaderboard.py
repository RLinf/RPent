# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render benchmark charts after checking every plotted score against both RST pages."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .benchmark_data import generated_tables_check, load_data, validate_data
else:
    from benchmark_data import generated_tables_check, load_data, validate_data


@dataclass(frozen=True)
class Entry:
    model: str
    score: float
    effort: str
    record_id: str
    rate: str
    kind: str


@dataclass(frozen=True)
class Panel:
    slug: str
    title: dict[str, str]
    subtitle: tuple[str, str]
    source_note: dict[str, str]
    maximum: int
    entries: tuple[Entry, ...]


DATA = load_data()
CONFIGURATIONS = {item["id"]: item for item in DATA["configurations"]}
RECORDS = {item["id"]: item for item in DATA["results"]}
VIEWS = {item["id"]: item for item in DATA["views"]}


def make_panel(view: dict, overview: bool = False) -> Panel:
    record_ids = view.get("overview_record_ids") if overview else None
    record_ids = record_ids if record_ids is not None else view["chart_record_ids"]
    entries = []
    for record_id in record_ids:
        record = RECORDS[record_id]
        if record["status"] != "reported":
            continue
        configuration = CONFIGURATIONS[record["configuration_id"]]
        entries.append(
            Entry(
                configuration["model"],
                float(record["rate"]),
                configuration["effort"] or "",
                record_id,
                record["rate"],
                configuration["kind"],
            )
        )
    return Panel(
        view["id"],
        view["title"],
        (view["subtitle"]["en"], view["subtitle"]["zh"]),
        view["source_note"],
        view["axis_max"],
        tuple(sorted(entries, key=lambda item: -item.score)),
    )


PANELS = tuple(
    make_panel(VIEWS[view_id], overview=True) for view_id in DATA["overview_views"]
)
ALL_PANELS = tuple(make_panel(view) for view in DATA["views"])

COLORS = {
    "light": {
        "background": "#FFFFFF",
        "text": "#182238",
        "muted": "#526078",
        "grid": "#E7EAF0",
    },
    "dark": {
        "background": "#141820",
        "text": "#EEF0F7",
        "muted": "#B9C1D1",
        "grid": "#353C4A",
    },
}
RPENT_COLOR = "#7842B1"
EXTERNAL_COLOR = "#AEB7C6"
DOCS = Path(__file__).resolve().parents[1]


def check_data() -> int:
    """Validate the canonical records and every generated bilingual result table."""
    validate_data(DATA)
    generated_tables_check(DATA)
    return sum(len(panel.entries) for panel in PANELS) * 2


def panel_description(panel: Panel) -> str:
    series = "; ".join(
        f"{'RPent / ' if entry.kind == 'rpent' else ''}{entry.model} {entry.effort}: {entry.rate}%"
        for entry in panel.entries
    )
    details = {
        "title": panel.title,
        "subtitle": panel.subtitle,
        "source_note": panel.source_note,
        "record_ids": [entry.record_id for entry in panel.entries],
    }
    return f"{series}; Y-axis: 0–{axis_maximum(panel)}%\n" + json.dumps(
        details, ensure_ascii=False, sort_keys=True
    )


def axis_maximum(panel: Panel) -> int:
    return panel.maximum


def axis_ticks(panel: Panel) -> list[int]:
    maximum = axis_maximum(panel)
    return sorted({*range(0, maximum + 1, 20), maximum})


def check_panel_geometry(tree: ET.ElementTree, panel: Panel, path: Path):
    """Check the rendered axis labels and bar heights without importing Matplotlib."""
    namespace = "{http://www.w3.org/2000/svg}"
    maximum = axis_maximum(panel)
    axes = tree.find(
        f".//{namespace}g[@id='leaderboard-axis-{panel.slug}-0-{maximum}']"
    )
    if axes is None:
        raise ValueError(f"{path}: missing expected 0–{maximum}% axis")
    for value in axis_ticks(panel):
        tick = axes.find(
            f".//{namespace}g[@id='leaderboard-tick-{panel.slug}-{value}']"
        )
        if tick is None or not any(
            child.tag is ET.Comment and child.text.strip() == f"{value}%"
            for child in tick.iter()
        ):
            raise ValueError(f"{path}: missing or incorrect {value}% axis label")
    bars = [
        group
        for group in axes.findall(f".//{namespace}g")
        if group.get("id", "").startswith(f"leaderboard-bar-{panel.slug}-")
    ]
    if len(bars) != len(panel.entries):
        raise ValueError(f"{path}: wrong number of bars in {panel.slug}")
    for index, entry in enumerate(panel.entries):
        bar = axes.find(
            f".//{namespace}g[@id='leaderboard-bar-{panel.slug}-{index}']/{namespace}path"
        )
        if bar is None:
            raise ValueError(f"{path}: missing bar for {entry.model}")
        clip_id = re.fullmatch(r"url\(#(.+)\)", bar.get("clip-path", ""))
        clip = (
            tree.find(f".//{namespace}clipPath[@id='{clip_id[1]}']/{namespace}rect")
            if clip_id
            else None
        )
        if clip is None:
            raise ValueError(f"{path}: missing rendered axis bounds")
        coordinates = [
            float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", bar.get("d", ""))
        ]
        if len(coordinates) != 8:
            raise ValueError(f"{path}: unexpected bar geometry for {entry.model}")
        y_coordinates = coordinates[1::2]
        height = float(clip.get("height"))
        baseline = float(clip.get("y")) + height
        actual_score = (max(y_coordinates) - min(y_coordinates)) / height * maximum
        color = RPENT_COLOR if entry.kind == "rpent" else EXTERNAL_COLOR
        if (
            not math.isclose(actual_score, entry.score, abs_tol=0.00001)
            or not math.isclose(max(y_coordinates), baseline, abs_tol=0.00001)
            or f"fill: {color.lower()}" not in bar.get("style", "")
        ):
            raise ValueError(
                f"{path}: incorrect height, baseline, or color for {entry.model}"
            )
        record = axes.find(f"{namespace}g[@data-record-id='{entry.record_id}']")
        if (
            record is None
            or record.find(
                f"{namespace}g[@id='leaderboard-bar-{panel.slug}-{index}']/{namespace}path"
            )
            is not bar
        ):
            raise ValueError(f"{path}: missing interactive record group")
        center = (min(coordinates[::2]) + max(coordinates[::2])) / 2
        if not math.isclose(
            float(record.get("data-center-x", "nan")), center, abs_tol=0.00001
        ):
            raise ValueError(f"{path}: incorrect interactive bar position")
        for component, expected in (
            ("value", [f"{entry.rate}%"]),
            ("model", model_label(entry).splitlines()),
        ):
            label = record.find(
                f"{namespace}g[@id='leaderboard-{component}-{panel.slug}-{index}']"
            )
            actual = (
                [
                    child.text.strip()
                    for child in label.iter()
                    if child.tag is ET.Comment
                ]
                if label is not None
                else []
            )
            if actual != expected:
                raise ValueError(f"{path}: mismatched interactive {component} label")
        if record.find(f".//{namespace}defs") is not None:
            raise ValueError(
                f"{path}: glyph definitions cannot depend on a visible bar"
            )


def check_assets(output: Path) -> int:
    """Reject stale SVG data, axis scales, or actual bar geometry in all figures."""
    count = 0
    for language in ("en", "zh"):
        for theme in COLORS:
            for panel in ALL_PANELS:
                path = output / f"{panel.slug}-{language}-{theme}.svg"
                tree = ET.parse(
                    path,
                    parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)),
                )
                description = tree.find(
                    ".//{http://purl.org/dc/elements/1.1/}description"
                )
                if description is None or description.text != panel_description(panel):
                    raise ValueError(
                        f"Stale chart asset: {path}; regenerate the figures"
                    )
                check_panel_geometry(tree, panel, path)
                check_font_glyphs(tree, language, path)
                count += 1
            path = output / f"leaderboard-{language}-{theme}.svg"
            tree = ET.parse(
                path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            )
            for panel in PANELS:
                check_panel_geometry(tree, panel, path)
            check_font_glyphs(tree, language, path)
            count += 1
    return count


def check_font_glyphs(tree: ET.ElementTree, language: str, path: Path):
    """Check the fonts actually embedded as SVG glyph paths, including mixed text."""
    fonts = {
        item.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#").rsplit("-", 1)[0]
        for item in tree.findall(".//{http://www.w3.org/2000/svg}use")
    }
    expected = {"TimesNewRomanPSMT", "TimesNewRomanPS-BoldMT"}
    if language == "zh":
        expected.add("KaiTi")
    if fonts != expected:
        raise ValueError(
            f"{path}: unexpected embedded fonts {fonts}; expected {expected}"
        )


def setup_matplotlib(times_font: Path, times_bold_font: Path, cjk_font: Path):
    import matplotlib
    from matplotlib import font_manager, ft2font

    matplotlib.use("Agg")
    for path, family, postscript in (
        (times_font, "Times New Roman", "TimesNewRomanPSMT"),
        (times_bold_font, "Times New Roman", "TimesNewRomanPS-BoldMT"),
        (cjk_font, "KaiTi", "KaiTi"),
    ):
        if path is None or not path.is_file():
            raise ValueError(
                "Rendering requires --times-font, --times-bold-font, and --cjk-font"
            )
        face = ft2font.FT2Font(str(path))
        if face.family_name != family or face.postscript_name != postscript:
            raise ValueError(
                f"{path}: expected genuine {family} / {postscript}, found {face.family_name} / {face.postscript_name}"
            )
        font_manager.fontManager.addfont(str(path))
    matplotlib.rcParams.update(
        {
            "font.family": ["Times New Roman", "KaiTi"],
            "svg.fonttype": "path",
            "svg.hashsalt": "rpent-benchmark-leaderboard-1",
            "axes.unicode_minus": False,
            "figure.dpi": 100,
            "savefig.dpi": 150,
        }
    )


def model_label(entry: Entry) -> str:
    if entry.effort:
        return f"{entry.model}\n{entry.effort}"
    return {
        "WorldDreamer": "World\nDreamer",
        "LingBot-VLA": "LingBot-\nVLA",
        "GR00T-N1.7": "GR00T-\nN1.7",
    }.get(entry.model, entry.model)


def draw_panel(figure, panel: Panel, language: str, theme: str, rectangle):
    left, bottom, width, height = rectangle
    colors = COLORS[theme]
    language_index = 0 if language == "en" else 1
    axes = figure.add_axes(
        (left + 0.105 * width, bottom + 0.26 * height, 0.875 * width, 0.47 * height)
    )
    axes.set_facecolor(colors["background"])
    maximum = axis_maximum(panel)
    axes.set_gid(f"leaderboard-axis-{panel.slug}-0-{maximum}")
    axes.set_ylim(0, maximum)
    axes.set_xlim(-0.62, len(panel.entries) - 0.38)
    ticks = axis_ticks(panel)
    axes.set_yticks(ticks, [f"{value}%" for value in ticks])
    for tick, value in zip(axes.get_yticklabels(), ticks, strict=True):
        tick.set_gid(f"leaderboard-tick-{panel.slug}-{value}")
    axes.tick_params(axis="y", colors=colors["muted"], labelsize=18, length=0, pad=6)
    axes.tick_params(axis="x", colors=colors["text"], labelsize=19, length=0, pad=13)
    axes.grid(axis="y", color=colors["grid"], linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "left", "right"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(colors["grid"])
    axes.set_xticks(
        range(len(panel.entries)), [model_label(entry) for entry in panel.entries]
    )
    for index, tick in enumerate(axes.get_xticklabels()):
        tick.set_gid(f"leaderboard-model-{panel.slug}-{index}")
    bars = axes.bar(
        range(len(panel.entries)),
        [entry.score for entry in panel.entries],
        # Keep physical bar widths equal in three-entry and five-entry panels.
        width=0.38 * (len(panel.entries) + 0.24) / 5.24,
        color=[
            RPENT_COLOR if entry.kind == "rpent" else EXTERNAL_COLOR
            for entry in panel.entries
        ],
        zorder=3,
    )
    for index, (bar, entry) in enumerate(zip(bars, panel.entries, strict=True)):
        bar.set_gid(f"leaderboard-bar-{panel.slug}-{index}")
        label = f"{entry.rate}%"
        annotation = axes.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=colors["text"],
            fontsize=23,
            fontweight="bold",
            annotation_clip=False,
        )
        annotation.set_gid(f"leaderboard-value-{panel.slug}-{index}")
    title = panel.title[language]
    for text, y, size, weight, color in (
        (title, 0.965, 26, "bold", colors["text"]),
        (
            panel.subtitle[language_index],
            0.875,
            19 if language == "en" else 21,
            "normal",
            colors["muted"],
        ),
    ):
        figure.text(
            left + 0.035 * width,
            bottom + y * height,
            text,
            fontsize=size,
            fontweight=weight,
            color=color,
            ha="left",
            va="top",
        )
    source = panel.source_note[language]
    figure.text(
        left + 0.105 * width,
        bottom + 0.025 * height,
        source,
        fontsize=16 if language == "en" else 18,
        color=colors["muted"],
        va="bottom",
    )


def audit_text_layout(figure, name: str) -> dict:
    """Measure all visible text boxes and reject clipping or intersecting labels."""
    from matplotlib.text import Text

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    boxes = []
    clipped = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bounds = artist.get_window_extent(renderer)
        item = {"text": artist.get_text(), "bounds": list(bounds.extents)}
        boxes.append(item)
        if (
            bounds.x0 < -0.5
            or bounds.y0 < -0.5
            or bounds.x1 > figure.bbox.width + 0.5
            or bounds.y1 > figure.bbox.height + 0.5
        ):
            clipped.append(item)
    overlaps = []
    for index, first in enumerate(boxes):
        x0, y0, x1, y1 = first["bounds"]
        for second in boxes[index + 1 :]:
            xx0, yy0, xx1, yy1 = second["bounds"]
            if min(x1, xx1) - max(x0, xx0) > 0.5 and min(y1, yy1) - max(y0, yy0) > 0.5:
                overlaps.append([first["text"], second["text"]])
    report = {
        "name": name,
        "text_count": len(boxes),
        "clipped": clipped,
        "overlaps": overlaps,
        "text_boxes": boxes,
    }
    if clipped or overlaps:
        raise ValueError(f"{name}: text clipping {clipped} or overlap {overlaps}")
    return report


def add_interactive_groups(path: Path, panels: tuple[Panel, ...]):
    """Keep bars and their outlined labels together for accessible SVG filtering."""
    namespaces = {
        "": "http://www.w3.org/2000/svg",
        "xlink": "http://www.w3.org/1999/xlink",
        "dc": "http://purl.org/dc/elements/1.1/",
        "cc": "http://creativecommons.org/ns#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    }
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    namespace = "{" + namespaces[""] + "}"
    tree = ET.parse(
        path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    )
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    definitions = root.find(f"{namespace}defs")
    for nested in list(root.iter(f"{namespace}defs")):
        if nested is definitions:
            continue
        for definition in list(nested):
            definitions.append(definition)
        parents[nested].remove(nested)
    for panel in panels:
        axis_id = f"leaderboard-axis-{panel.slug}-0-{axis_maximum(panel)}"
        axes = root.find(f".//{namespace}g[@id='{axis_id}']")
        for index, entry in enumerate(panel.entries):
            bar = axes.find(f"{namespace}g[@id='leaderboard-bar-{panel.slug}-{index}']")
            shape = bar.find(f"{namespace}path")
            coordinates = [
                float(value)
                for value in re.findall(r"-?\d+(?:\.\d+)?", shape.get("d", ""))
            ]
            center = (min(coordinates[::2]) + max(coordinates[::2])) / 2
            clip_id = re.fullmatch(r"url\(#(.+)\)", shape.get("clip-path", ""))[1]
            bounds = root.find(
                f".//{namespace}clipPath[@id='{clip_id}']/{namespace}rect"
            )
            for target in (axes, root) if len(panels) == 1 else (axes,):
                target.set("data-axis-left", bounds.get("x"))
                target.set("data-axis-width", bounds.get("width"))
                target.set("data-axis-max", str(axis_maximum(panel)))
            group = ET.Element(
                f"{namespace}g",
                {
                    "class": "rpent-bar-record",
                    "data-record-id": entry.record_id,
                    "data-center-x": f"{center:.6f}",
                },
            )
            position = list(axes).index(bar)
            value = axes.find(
                f"{namespace}g[@id='leaderboard-value-{panel.slug}-{index}']"
            )
            label = axes.find(
                f".//{namespace}g[@id='leaderboard-model-{panel.slug}-{index}']"
            )
            for artist in (bar, value, label):
                parents[artist].remove(artist)
                group.append(artist)
            axes.insert(position, group)
    root.set("role", "group")
    root.set("data-rpent-interactive", "1")
    path.write_text(
        ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n",
        encoding="utf-8",
    )


def save_figure(
    figure, output: Path, name: str, description: str, panels: tuple[Panel, ...]
):
    audit = audit_text_layout(figure, name)
    for extension in ("svg", "png"):
        metadata = {"Title": name, "Description": description}
        if extension == "svg":
            metadata.update({"Date": None, "Creator": "RPent leaderboard renderer"})
        else:
            metadata["Software"] = "RPent leaderboard renderer"
        path = output / f"{name}.{extension}"
        figure.savefig(path, metadata=metadata)
        if extension == "svg":
            add_interactive_groups(path, panels)
            # Matplotlib leaves spaces at the ends of path-coordinate lines.
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8"
            )
    return audit


def render(output: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output.mkdir(parents=True, exist_ok=True)
    audits = []
    for language in ("en", "zh"):
        for theme, colors in COLORS.items():
            for panel in ALL_PANELS:
                figure = plt.figure(figsize=(7, 5.4), facecolor=colors["background"])
                draw_panel(figure, panel, language, theme, (0, 0, 1, 1))
                description = panel_description(panel)
                audits.append(
                    save_figure(
                        figure,
                        output,
                        f"{panel.slug}-{language}-{theme}",
                        description,
                        (panel,),
                    )
                )
                plt.close(figure)
            figure = plt.figure(figsize=(21.8, 13.0), facecolor=colors["background"])
            title = (
                "RPent · Benchmark Leaderboard"
                if language == "en"
                else "RPent · 基准测试 Leaderboard"
            )
            figure.text(
                0.018,
                0.97,
                title,
                color=colors["text"],
                fontsize=36,
                weight="bold",
                va="top",
            )
            legend_labels = (
                "RPent",
                "External methods" if language == "en" else "外部方法",
            )
            figure.legend(
                handles=[Patch(color=RPENT_COLOR), Patch(color=EXTERNAL_COLOR)],
                labels=legend_labels,
                loc="upper right",
                bbox_to_anchor=(0.982, 0.97),
                borderaxespad=0,
                ncols=2,
                frameon=False,
                labelcolor=colors["text"],
                fontsize=19 if language == "en" else 21,
            )
            for index, panel in enumerate(PANELS):
                row, column = divmod(index, 3)
                draw_panel(
                    figure,
                    panel,
                    language,
                    theme,
                    (
                        (0.1 + column * 7.25) / 21.8,
                        (6.25 - row * 5.6) / 13.0,
                        7 / 21.8,
                        5.4 / 13.0,
                    ),
                )
            note = (
                "Success rate ↑ · Each axis starts at 0%; upper limits vary by panel. Rankings apply only to the displayed methods and coverage."
                if language == "en"
                else "成功率 ↑ · 各图纵轴从 0% 起，上限随面板变化；排名仅限图中方法及对应评测范围。"
            )
            figure.text(
                0.035,
                0.022,
                note,
                fontsize=19 if language == "en" else 21,
                color=colors["muted"],
            )
            description = (
                note
                + "\n"
                + "\n".join(
                    f"{panel.slug}: {panel_description(panel)}" for panel in PANELS
                )
            )
            audits.append(
                save_figure(
                    figure,
                    output,
                    f"leaderboard-{language}-{theme}",
                    description,
                    PANELS,
                )
            )
            plt.close(figure)
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check RST data without rendering or Matplotlib",
    )
    parser.add_argument("--output-dir", type=Path, default=DOCS / "_static/benchmarks")
    parser.add_argument(
        "--times-font", type=Path, help="Genuine Times New Roman regular TTF"
    )
    parser.add_argument(
        "--times-bold-font", type=Path, help="Genuine Times New Roman bold TTF"
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        help="Write complete text-box layout checks outside the source tree",
    )
    parser.add_argument(
        "--cjk-font",
        type=Path,
        help="Genuine KaiTi (楷体) simkai.ttf; substitutes are rejected",
    )
    args = parser.parse_args()
    checked = check_data()
    print(
        f"Validated {checked} chart scores in the overview and all result records against both RST pages."
    )
    if not args.check_data:
        setup_matplotlib(args.times_font, args.times_bold_font, args.cjk_font)
        audits = render(args.output_dir)
        if args.audit_report:
            args.audit_report.parent.mkdir(parents=True, exist_ok=True)
            args.audit_report.write_text(
                json.dumps(
                    {"passed": True, "figures": audits}, ensure_ascii=False, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            f"Rendered {(len(ALL_PANELS) + 1) * 8} SVG/PNG assets in {args.output_dir}"
        )
    checked_assets = check_assets(args.output_dir)
    print(
        f"Validated {checked_assets} panel SVGs (including combined figures): scores, axes, bar geometry, and genuine font glyphs."
    )


if __name__ == "__main__":
    main()
