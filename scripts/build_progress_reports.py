"""Build the ten mentor progress reports as editable Word files and PDFs."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "progress"
FIG = OUT / "figures"
FONT = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INK = (26, 29, 33)
MUTED = (90, 98, 108)
LINE = (197, 205, 214)
GREEN = (11, 107, 79)
GREEN_BG = (232, 243, 238)
RED = (155, 28, 28)
RED_BG = (250, 236, 236)
PAPER = (255, 255, 255)

AUTHOR = "Atul Kumar Gupta"
ROLL = "25MCSS06"
TITLE = "LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference"


def _font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(SANS_BOLD if bold else SANS, size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _center(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int, int, int], font, fill) -> None:
    x0, y0, x1, y1 = xy
    tw, th = _text_size(draw, text, font)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2), text, font=font, fill=fill)


def _box(draw, xy, text, fill=PAPER, outline=INK, size=22) -> None:
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=3)
    _center(draw, text, xy, _font(True, size), INK)


def _arrow(draw, x0, y, x1) -> None:
    draw.line((x0, y, x1 - 14, y), fill=INK, width=3)
    draw.polygon([(x1 - 16, y - 7), (x1, y), (x1 - 16, y + 7)], fill=INK)


def _save(name: str, image: Image.Image) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    image.save(path, "PNG")
    return path


def fig_split() -> Path:
    image = Image.new("RGB", (1100, 1400), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((80, 80, 1020, 560), radius=16, fill=GREEN_BG, outline=GREEN, width=5)
    _center(draw, "Flight controller", (80, 220, 1020, 320), _font(True, 40), INK)
    _center(draw, "PX4 keeps the aircraft flying", (80, 330, 1020, 410), _font(False, 28), MUTED)
    draw.line((550, 580, 550, 700), fill=INK, width=4)
    draw.polygon([(530, 690), (570, 690), (550, 720)], fill=INK)
    draw.rounded_rectangle((80, 760, 1020, 1240), radius=16, fill=GREEN_BG, outline=GREEN, width=5)
    _center(draw, "Companion computer", (80, 900, 1020, 1000), _font(True, 40), INK)
    _center(draw, "The camera network runs here", (80, 1020, 1020, 1100), _font(False, 28), MUTED)
    return _save("01_split.png", image)


def fig_rule() -> Path:
    image = Image.new("RGB", (1100, 1400), PAPER)
    draw = ImageDraw.Draw(image)
    labels = [
        ("Suggest", "name a plan"),
        ("Check", "accept or cut it"),
        ("Rewrite", "change the ONNX graph"),
        ("Measure", "time and numeric error"),
    ]
    y = 60
    for index, (title, subtitle) in enumerate(labels):
        draw.rounded_rectangle((120, y, 980, y + 240), radius=16, fill=GREEN_BG, outline=GREEN, width=5)
        _center(draw, title, (120, y + 40, 980, y + 130), _font(True, 40), INK)
        _center(draw, subtitle, (120, y + 130, 980, y + 200), _font(False, 26), MUTED)
        if index < 3:
            mid = y + 250
            draw.line((550, y + 246, 550, mid + 20), fill=INK, width=4)
            draw.polygon([(530, mid + 8), (570, mid + 8), (550, mid + 32)], fill=INK)
        y += 330
    return _save("02_rule.png", image)


def fig_fuse() -> Path:
    image = Image.new("RGB", (1400, 480), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((80, 30), "Before", font=_font(True, 26), fill=MUTED)
    draw.text((820, 30), "After graph_fuse", font=_font(True, 26), fill=MUTED)
    for x, labels in (
        (80, ["Conv", "Relu", "Flatten 64", "Gemm K = 64"]),
        (820, ["FusedConv", "Flatten 64", "Gemm K = 64"]),
    ):
        y = 90
        for label in labels:
            draw.rounded_rectangle((x, y, x + 460, y + 70), radius=10, fill=GREEN_BG, outline=GREEN, width=3)
            _center(draw, label, (x, y, x + 460, y + 70), _font(True, 24), INK)
            y += 90
    return _save("03_fuse.png", image)


def fig_zoo() -> Path:
    image = Image.new("RGB", (1400, 560), PAPER)
    draw = ImageDraw.Draw(image)
    names = [
        "yolov8n  detect",
        "yolo11n  detect",
        "yolov8n-pose",
        "yolov8n-seg",
        "ssdlite  detect",
        "mobilenet  classify",
        "midas  depth",
    ]
    for index, name in enumerate(names):
        col, row = index % 4, index // 4
        x, y = 40 + col * 330, 30 + row * 90
        if index == 6:
            x, y = 40, 210
        draw.rounded_rectangle((x, y, x + 300, y + 70), radius=10, fill=GREEN_BG, outline=GREEN, width=3)
        _center(draw, name, (x, y, x + 300, y + 70), _font(True, 18), INK)
    draw.rounded_rectangle((430, 360, 970, 470), radius=12, fill=INK, outline=INK, width=3)
    _center(draw, "one compile loop", (430, 360, 970, 470), _font(True, 28), PAPER)
    draw.line((190, 280, 700, 360), fill=INK, width=3)
    return _save("04_zoo.png", image)


def fig_gate() -> Path:
    image = Image.new("RGB", (1400, 460), PAPER)
    draw = ImageDraw.Draw(image)
    for index in range(3):
        x = 40 + index * 280
        draw.rounded_rectangle((x, 40, x + 240, 140), radius=10, fill=GREEN_BG, outline=GREEN, width=3)
        _center(draw, f"Try {index + 1}", (x, 40, x + 240, 140), _font(True, 26), INK)
        if index < 2:
            _arrow(draw, x + 246, 90, x + 274)
    draw.rounded_rectangle((900, 40, 1320, 160), radius=10, fill=GREEN_BG, outline=GREEN, width=3)
    _center(draw, "Accepted plan", (900, 40, 1320, 160), _font(True, 26), INK)
    draw.rounded_rectangle((900, 260, 1320, 380), radius=10, fill=RED_BG, outline=RED, width=3)
    _center(draw, "Heuristic fallback", (900, 260, 1320, 380), _font(True, 24), INK)
    draw.line((880, 90, 900, 100), fill=INK, width=3)
    draw.line((760, 140, 760, 320), fill=INK, width=3)
    draw.line((760, 320, 900, 320), fill=INK, width=3)
    return _save("05_gate.png", image)


def fig_bars() -> Path:
    image = Image.new("RGB", (1400, 480), PAPER)
    draw = ImageDraw.Draw(image)
    rows = [
        (220, "preset", GREEN),
        (140, "chosen", GREEN),
        (100, "faster, failed check", RED),
    ]
    y = 50
    for width, label, color in rows:
        draw.rectangle((280, y, 280 + width * 4, y + 70), fill=color)
        draw.text((40, y + 18), label, font=_font(True, 24), fill=INK)
        y += 120
    draw.text((280, 410), "shorter bar = lower p50    red = not chosen", font=_font(False, 22), fill=MUTED)
    return _save("06_bars.png", image)


def fig_planes() -> Path:
    image = Image.new("RGB", (1400, 420), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 80, 460, 320), radius=12, fill=GREEN_BG, outline=GREEN, width=4)
    _center(draw, "Compile", (40, 140, 460, 210), _font(True, 30), INK)
    _center(draw, "plan, rewrite, measure", (40, 210, 460, 270), _font(False, 20), MUTED)
    draw.rounded_rectangle((540, 140, 860, 260), radius=12, fill=INK, outline=INK, width=3)
    _center(draw, "ONNX + sidecar", (540, 140, 860, 260), _font(True, 24), PAPER)
    draw.rounded_rectangle((940, 80, 1360, 320), radius=12, fill=GREEN_BG, outline=GREEN, width=4)
    _center(draw, "Deploy", (940, 140, 1360, 210), _font(True, 30), INK)
    _center(draw, "Gazebo camera, ROS node", (940, 210, 1360, 270), _font(False, 20), MUTED)
    _arrow(draw, 470, 200, 530)
    _arrow(draw, 870, 200, 930)
    return _save("07_planes.png", image)


def fig_hosts() -> Path:
    image = Image.new("RGB", (1400, 480), PAPER)
    draw = ImageDraw.Draw(image)
    hosts = ["Groq", "OpenAI", "Gemini", "DeepSeek", "Anthropic", "local"]
    for index, name in enumerate(hosts):
        x = 40 + (index % 3) * 250
        y = 30 + (index // 3) * 90
        draw.rounded_rectangle((x, y, x + 220, y + 70), radius=10, fill=GREEN_BG, outline=GREEN, width=3)
        _center(draw, name, (x, y, x + 220, y + 70), _font(True, 22), INK)
    draw.rounded_rectangle((860, 80, 1340, 200), radius=12, fill=INK, outline=INK, width=3)
    _center(draw, "one HTTP client", (860, 80, 1340, 200), _font(True, 28), PAPER)
    draw.rounded_rectangle((860, 280, 1340, 400), radius=12, fill=GREEN_BG, outline=GREEN, width=4)
    _center(draw, "one extra try, then measure", (860, 280, 1340, 400), _font(True, 22), INK)
    draw.line((1100, 210, 1100, 270), fill=INK, width=3)
    draw.polygon([(1093, 268), (1107, 268), (1100, 286)], fill=INK)
    return _save("08_hosts.png", image)


def fig_page() -> Path:
    image = Image.new("RGB", (900, 720), PAPER)
    draw = ImageDraw.Draw(image)
    bands = [
        "Host: this machine, no NVIDIA",
        "Graph notes",
        "Plan and attempts",
        "Ranking and chosen file",
        "Zoo table and remeasure",
    ]
    y = 30
    for band in bands:
        draw.rounded_rectangle((40, y, 860, y + 110), radius=12, fill=GREEN_BG, outline=GREEN, width=3)
        _center(draw, band, (40, y, 860, y + 110), _font(True, 26), INK)
        y += 130
    return _save("09_page.png", image)


def fig_claim() -> Path:
    image = Image.new("RGB", (1400, 520), PAPER)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 660, 480), radius=12, fill=GREEN_BG, outline=GREEN, width=4)
    draw.rounded_rectangle((740, 40, 1360, 480), radius=12, fill=RED_BG, outline=RED, width=4)
    _center(draw, "Can say", (40, 60, 660, 130), _font(True, 30), INK)
    _center(draw, "Cannot say yet", (740, 60, 1360, 130), _font(True, 30), INK)
    left = [
        "Allowlisted plan",
        "Real ONNX rewrite",
        "Measured CPU choice",
        "One extra try",
        "ROS only loads the file",
    ]
    right = ["TensorRT speed", "Board power", "Labelled camera accuracy"]
    y = 160
    for line in left:
        draw.text((80, y), line, font=_font(False, 24), fill=INK)
        y += 52
    y = 180
    for line in right:
        draw.text((780, y), line, font=_font(False, 24), fill=INK)
        y += 64
    return _save("10_claim.png", image)


Report = dict


def reports() -> list[Report]:
    return [
        {
            "num": 1,
            "slug": "companion",
            "heading": "Why the companion computer is a compile problem",
            "intro": [
                "This is the first progress note.",
                "A small drone splits the work. PX4 on the flight controller keeps the aircraft in the air. A Linux computer beside it has to run the camera network, and it has to keep up with the frames.",
            ],
            "body": [
                "A compile recipe is the set of choices that turn a trained network into a file that can run: which joins of operations to allow, whether weights are 8-bit, and how many CPU threads to use. People still pick that recipe by habit. A recipe that is fast on a desktop GPU is often slow, or numerically wrong, on a small ARM CPU.",
                "The lab stack had to be named before any compiler was written.",
            ],
            "bullets": [
                "The companion computer runs the camera network. The flight controller does not.",
                "PX4 is the flight stack, including in simulation.",
                "Gazebo is the world the aircraft flies in.",
                "ROS 2 carries messages between programs.",
                "ONNX stores the network as a graph of operations.",
            ],
            "after": [
                "A camera that sends a frame every 33 ms does not leave much time. If the network, and the compile choices around it, use more than that, frames are dropped. That is why the recipe is part of the thesis, not a setup detail.",
            ],
            "caption": "Figure 1. Flight stays on PX4. The camera network stays on the companion computer.",
            "figure": fig_split,
            "outro": "Flight and vision are separate jobs. The next note asks who is allowed to choose the compile recipe.",
        },
        {
            "num": 2,
            "slug": "rule",
            "heading": "The language model may only name a plan",
            "intro": [
                "Report 1 separated flight from vision.",
                "This note fixes who may choose the compile recipe. A language model, asked to make a network fast, will invent passes that do not exist and quote speeds it never measured.",
            ],
            "body": [
                "The model may only name a plan from a fixed list. A checker accepts or cuts that plan. An engine then rewrites the ONNX graph. A profiler measures time and compares outputs with the untouched graph. The model does not edit the file, and it does not fly the aircraft.",
                "The plan is checked as JSON before any rewrite. A sentence that contains a made-up speed is stripped. The stored record always says that no frame rate was claimed.",
            ],
            "bullets": [],
            "caption": "Figure 1. The model suggests. The rest of the system checks, rewrites, and measures.",
            "figure": fig_rule,
            "outro": "The model is a planner, not the compiler. The next note shows that the rewrite changes a real graph.",
        },
        {
            "num": 3,
            "slug": "fixture",
            "heading": "A tiny network proves the rewrite",
            "intro": [
                "Report 2 said an engine rewrites the graph.",
                "This note shows one rewrite, on a network small enough to inspect by hand.",
            ],
            "body": [
                "The fixture is one image of size 1×1×8×8. Flatten turns that map into 64 values. The Gemm that follows must use width 64. Width 16 is kept as a test that must fail.",
                "The pass graph_fuse joins Conv and Relu. On ONNX Runtime for the CPU, the Relu node disappears and a FusedConv node remains. That change is in the graph itself.",
                "Both facts are locked by tests. A Gemm width of 16 must not be treated as a successful compile. graph_fuse on this fixture must leave one FusedConv and no Relu.",
            ],
            "bullets": [],
            "caption": "Figure 1. Conv and Relu become one FusedConv. The Flatten width stays 64.",
            "figure": fig_fuse,
            "outro": "The rewrite is visible on this fixture. The next note runs the same commands on the networks a companion computer uses.",
        },
        {
            "num": 4,
            "slug": "zoo",
            "heading": "One loop, seven companion models",
            "intro": [
                "Report 3 used a toy graph.",
                "This note runs the same compile commands on seven camera networks. The compiler does not contain a special case for any one of them.",
            ],
            "body": [
                "Each model is one record: a task, a file, and an export script.",
            ],
            "bullets": [
                "yolov8n and yolo11n detect people and vehicles.",
                "yolov8n-pose detects body keypoints.",
                "yolov8n-seg produces masks.",
                "ssdlite_mobilenetv3 is a lighter detector.",
                "mobilenetv3_small is a small classifier.",
                "midas_small estimates depth from one camera.",
            ],
            "after": [
                "Ultralytics YOLO graphs use SiLU, which is Conv, then Sigmoid, then Mul. The CPU fusion pass only joins Conv and Relu, so those YOLO graphs do not get smaller. MobileNet and MiDaS do get smaller where Conv is followed by batch-norm or Relu. That difference is a result.",
                "There is no fused CPU kernel for SiLU in this project, and none will be invented to make the node count fall. The list of models lives in experiments/zoo.yaml. Adding a model means a new record and an export script, not a new branch in the compiler.",
            ],
            "caption": "Figure 1. Seven models enter one compile loop.",
            "figure": fig_zoo,
            "outro": "One loop covers the zoo. The next note is the gate that can refuse a plan before any rewrite.",
        },
        {
            "num": 5,
            "slug": "gate",
            "heading": "A plan can be refused",
            "intro": [
                "Report 4 put seven models through one loop.",
                "This note is only the gate in front of the engine.",
            ],
            "body": [
                "A plan is JSON. It holds at most eight steps. Each step name must come from a closed list. The plan may also set how hard ONNX Runtime optimizes the graph, how many threads to use, and whether execution is sequential or parallel.",
                "An unknown name rejects the whole plan. The engine never sees it. A known name that does not match this graph, such as batch-norm fusion when the graph has no batch norm, is dropped. The rest of the plan continues.",
                "The checker allows three tries. If the reply is still illegal, an offline heuristic plan is used. That result is labelled as a fallback.",
                "On this CPU, a thread count above the number of cores is cut down to that number. Parallel execution is left as the plan asked for it. The checker does not rename the plan. It only removes or clamps what cannot run.",
            ],
            "bullets": [],
            "caption": "Figure 1. Three tries. Then an accepted plan, or the heuristic fallback.",
            "figure": fig_gate,
            "outro": "Illegal plans stop or shrink before the rewrite. The next note is how a winner is chosen after measurement.",
        },
        {
            "num": 6,
            "slug": "winner",
            "heading": "The winner is the fastest plan that passed",
            "intro": [
                "Report 5 said which plans are legal.",
                "This note says which legal plan is kept.",
            ],
            "body": [
                "Several legal plans are compiled on the same machine: fixed presets, the heuristic, and the model's plan. Each run records whether compile succeeded, whether the outputs still match the untouched graph, and the latency percentile p50.",
                "The chosen plan is the one with the lowest p50 among the plans that passed the numeric check. A plan that is faster but fails the check stays in the table. It is not chosen. If the model's plan has no rank, that is the reason, not a missing run.",
                "p50 is the median latency of repeated runs on this machine. It moves when the QEMU load moves, so two runs from different sessions are not one experiment. The numeric check compares the rewritten network with the original FP32 graph.",
            ],
            "bullets": [],
            "caption": "Figure 1. The short green bar can win. The red bar is faster and still rejected.",
            "figure": fig_bars,
            "outro": "Speed without a passed check is not a win. The next note is where the chosen file is loaded.",
        },
        {
            "num": 7,
            "slug": "deploy",
            "heading": "The aircraft only loads the file",
            "intro": [
                "Report 6 ends with an ONNX file and a sidecar of session options.",
                "This note is the program that loads that file on the simulated aircraft.",
            ],
            "body": [
                "Gazebo publishes the camera on the ROS topic /nnc/camera/image_raw. The simulated airframe is nnc_x500_cam. PX4 SITL flies beside it. The ROS node inference_node loads the artifact and runs it on those frames.",
                "The node does not send flight setpoints. The compiler does not read the flight controller. The aircraft is a place to run the file.",
                "The sidecar next to the ONNX file stores the plan and the session options, so the node uses the same thread count that was measured. Camera frames can be written to disk with record_frames.py. That recording is a separate experiment from the compile table.",
            ],
            "bullets": [],
            "caption": "Figure 1. The compile side and the aircraft share only the artifact.",
            "figure": fig_planes,
            "outro": "Deployment loads the measured file and does not fly. The next note lets a remote model name the plan.",
        },
        {
            "num": 8,
            "slug": "hosted",
            "heading": "A hosted model, one revision",
            "intro": [
                "Reports 5 and 6 already had a planner.",
                "This note says that planner can be a model on a remote host, without a second client.",
            ],
            "body": [
                "There is one HTTP client. Groq is one named host. OpenAI, Anthropic, Gemini, DeepSeek, and other hosts that speak the same chat API are further rows. A new host is a URL and a model id. The API key stays outside git.",
                "After a plan has been measured, there is at most one extra call. If compile or the numeric check failed, that call is named llm_revised. If the plan passed but was strictly slower than the best legal plan, the call is named llm_improved. A rate limit is stored as the text LLM HTTP 429. The extra plan is allowed to lose.",
                "A slow reasoning model needs a longer timeout and sometimes no temperature field. Those are settings in the environment, not a new program. The same three tries from the gate still apply on that extra call. Only one extra compile is run.",
            ],
            "bullets": [],
            "caption": "Figure 1. Many hosts, one client, then one extra measured try.",
            "figure": fig_hosts,
            "outro": "A remote model can propose a plan, and the record can show that it was wrong. The next note is the page used to run and show this loop.",
        },
        {
            "num": 9,
            "slug": "page",
            "heading": "A page that runs the loop",
            "intro": [
                "Reports 3 to 8 were command-line steps.",
                "This note is a page on this machine that calls those same functions.",
            ],
            "body": [
                "The page shows the host, the graph notes, the plan and its attempts, the ranking, a table across the seven models, and a fresh timing of the chosen file. The operator picks a host and a model id. If the key is missing, the page says key not set. It does not ask for the key.",
                "The seven ONNX files are stored in experiments/models.",
                "The page listens only on this machine, at 127.0.0.1. One job runs at a time. If the console process is down, the buttons become usable again and the bar says the console is not reachable. Analyze, plan, optimize, export, and a remeasure of the chosen file all call the same Python functions as the command line.",
            ],
            "bullets": [],
            "caption": "Figure 1. The page is those five bands, in that order.",
            "figure": fig_page,
            "outro": "The loop can be run and shown from one page. The last note states what the thesis can claim.",
        },
        {
            "num": 10,
            "slug": "claim",
            "heading": "What can be claimed",
            "intro": [
                "Reports 1 to 9 are the path from the problem to a measured file on a simulated camera.",
                "This note adds no new tool. It states the claim.",
            ],
            "body": [
                "The work can say that a language model names an allowlisted plan, that a checker can refuse it, that an engine rewrites ONNX, that a profiler measures, that the kept plan is the fastest one that passed, that one extra try is allowed after measurement, and that ROS only loads the file.",
                "It cannot yet say TensorRT speed, board power, or accuracy on labelled images. Those need a board with an NVIDIA GPU. This machine is QEMU on ARM and has none. Latency from this machine must not be compared with a Jetson.",
                "Every stored run sets fps_claimed to false. When a throughput number appears, it is 1000 divided by the mean latency in milliseconds. It is not a camera frame rate, and it is not a claim about a Jetson.",
            ],
            "bullets": [],
            "caption": "Figure 1. The left column is the thesis. The right column waits for a board.",
            "figure": fig_claim,
            "outro": "The next measurement, on a real board, is TensorRT and power. These ten reports are not that result.",
        },
    ]


def _set_run_font(run, name: str, size: int, bold: bool = False, color: RGBColor | None = None) -> None:
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if color is not None:
        run.font.color.rgb = color


def write_docx(item: Report, figure: Path) -> Path:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Liberation Serif"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Liberation Serif")

    def add(text: str, *, size=11, bold=False, space_after=8, center=False) -> None:
        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(space_after)
        para.paragraph_format.space_before = Pt(2)
        para.paragraph_format.line_spacing = 1.15
        if center:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = para.add_run(text)
        _set_run_font(run, "Liberation Serif", 12 if size == 11 else size, bold)

    add(f"Progress report {item['num']} of 10", size=11, bold=True, space_after=2)
    add(TITLE, size=12, space_after=2)
    add(f"{AUTHOR}  ·  {ROLL}", size=11, space_after=10)
    add(item["heading"], size=16, bold=True, space_after=10)
    for paragraph in item["intro"]:
        add(paragraph)
    for paragraph in item["body"]:
        add(paragraph)
    for bullet in item["bullets"]:
        para = doc.add_paragraph(style="List Bullet")
        para.paragraph_format.space_after = Pt(2)
        run = para.add_run(bullet)
        _set_run_font(run, "Liberation Serif", 11)
    for paragraph in item.get("after", []):
        add(paragraph)
    doc.add_page_break()
    doc.add_picture(str(figure), width=Inches(6.2))
    add(item["caption"], size=10, space_after=10)
    add(item["outro"])
    path = OUT / f"{item['num']:02d}_{item['slug']}.docx"
    doc.save(path)
    return path


class ReportPDF(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Serif", size=9)
        self.set_text_color(90, 98, 108)
        self.cell(0, 8, f"{AUTHOR}  ·  {ROLL}  ·  {self.page_no()}", align="C")


def write_pdf(item: Report, figure: Path) -> Path:
    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_font("Serif", "", FONT)
    pdf.add_font("Serif", "B", FONT_BOLD)
    pdf.add_page()
    pdf.set_text_color(26, 29, 33)

    def para(text: str, *, size: int = 12, bold: bool = False, height: float = 6.3) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Serif", "B" if bold else "", size)
        pdf.multi_cell(pdf.epw, height, text)
        pdf.ln(2.2)

    para(f"Progress report {item['num']} of 10", bold=True, height=6)
    para(TITLE, height=6)
    para(f"{AUTHOR}  ·  {ROLL}", height=6)
    pdf.ln(1)
    para(item["heading"], size=16, bold=True, height=8)
    pdf.ln(1)

    for paragraph in item["intro"] + item["body"]:
        para(paragraph)
    for bullet in item["bullets"]:
        para(f"- {bullet}")
    for paragraph in item.get("after", []):
        para(paragraph)
    pdf.add_page()
    pdf.image(str(figure), w=pdf.epw)
    pdf.ln(3)
    para(item["caption"], size=10)
    para(item["outro"])
    path = OUT / f"{item['num']:02d}_{item['slug']}.pdf"
    pdf.output(str(path))
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for item in reports():
        figure = item["figure"]()
        docx = write_docx(item, figure)
        pdf = write_pdf(item, figure)
        print(docx.name, pdf.name)


if __name__ == "__main__":
    main()
