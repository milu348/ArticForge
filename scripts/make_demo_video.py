#!/usr/bin/env python3
"""Generate the Chinese-caption ArticForge demo video (no screen recording).

Composes title/problem/flow cards, the five delivered assets, joint motions,
force-calibration data and the Panda-manipulation clip into one MP4.

Usage:
  .venv/bin/python scripts/make_demo_video.py \
      --out docs/report-assets/ArticForge-demo.mp4
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 1280, 720, 24
BG = (13, 18, 30)
ACCENT = (72, 207, 194)
WHITE = (238, 242, 248)
GREY = (150, 162, 180)
FONT_CANDIDATES = [
    "/home/milu/.local/share/fonts/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def find_font():
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    for root in (Path.home() / ".local/share/fonts", Path("/usr/share/fonts")):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in (".otf", ".ttf", ".ttc") and any(
                    k in p.name.lower() for k in ("noto", "cjk", "wqy", "hei")):
                return str(p)
    raise RuntimeError("未找到中文字体（Noto Sans CJK / WenQuanYi 等），无法渲染字幕")


FONT = find_font()


def font(size):
    return ImageFont.truetype(FONT, size)


def base_frame():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(BG[0] + 8 * (1 - t)),
                                       int(BG[1] + 10 * (1 - t)),
                                       int(BG[2] + 14 * (1 - t))))
    return img


def centered(d, text, y, size, fill=WHITE, f=None):
    f = f or font(size)
    box = d.textbbox((0, 0), text, font=f)
    x = (W - (box[2] - box[0])) // 2
    d.text((x, y), text, font=f, fill=fill)


def fade(img, frame_idx, total, fade_n):
    """Blend the whole frame to background at the segment edges."""
    if fade_n <= 0:
        return np.asarray(img)
    a = 1.0
    if frame_idx < fade_n:
        a = frame_idx / fade_n
    elif frame_idx >= total - fade_n:
        a = (total - 1 - frame_idx) / fade_n
    if a >= 0.999:
        return np.asarray(img)
    arr = np.asarray(img).astype(np.float32)
    bg = np.array(BG, dtype=np.float32)
    return np.clip(arr * a + bg * (1 - a), 0, 255).astype(np.uint8)


def emit(writer, img, i, n, fade_n=6):
    writer.append_data(fade(img, i, n, fade_n))


def card(lines, duration=4.0):
    """Full-screen title/problem card; lines = [(text, size, color, gap)]."""
    n = int(duration * FPS)
    for i in range(n):
        img = base_frame()
        d = ImageDraw.Draw(img)
        y = 150
        for text, size, color, gap in lines:
            centered(d, text, y, size, color)
            y += size + gap
        d.line([(W // 2 - 90, y + 20), (W // 2 + 90, y + 20)], fill=ACCENT, width=3)
        emit(writer_global, img, i, n, 8)


def flow_card(duration=3.5):
    n = int(duration * FPS)
    steps = ["① 文生 3D", "② 关节化", "③ 自动适配", "④ 凸包碰撞", "⑤ 力标定", "⑥ 双引擎验证"]
    for i in range(n):
        img = base_frame()
        d = ImageDraw.Draw(img)
        centered(d, "从一句话到可训练资产", 110, 46, WHITE)
        for j, s in enumerate(steps):
            row, col = divmod(j, 3)
            x = 120 + col * 355
            y = 250 + row * 150
            d.rounded_rectangle([x, y, x + 310, y + 110], radius=14,
                                fill=(24, 34, 52), outline=ACCENT, width=2)
            f = font(30)
            box = d.textbbox((0, 0), s, font=f)
            d.text((x + (310 - (box[2] - box[0])) // 2, y + 34), s, font=f, fill=WHITE)
            if col < 2:
                d.text((x + 322, y + 36), "→", font=font(34), fill=ACCENT)
        emit(writer_global, img, i, n, 8)


def terminal_card(duration=8.0):
    """Terminal-style display of the pipeline commands (DISPLAY ONLY — no API calls)."""
    lines = [
        ("# 云端 LUX3D（约 41 积分）", GREY),
        ('$ python pipeline/1_text_to_3d.py $QUOTE_ID --asset side-table-01 --prompt "木质边桌"', WHITE),
        ("$ python skills/aholo-lux3d/.../asset_uploader.py upload mesh.glb --region cn", WHITE),
        ("$ python pipeline/2_articulation_submit.py --asset side-table-01 --glb-url $URL", WHITE),
        ("", GREY),
        ("# 本地流水线 MuJoCo（0 积分）", GREY),
        ("$ python pipeline/3_adapt_urdf.py --asset side-table-01", WHITE),
        ("$ python pipeline/4_make_collision.py --asset side-table-01", WHITE),
        ("$ python pipeline/5_filter_hulls.py --asset side-table-01", WHITE),
        ("$ python pipeline/6_force_probe.py --asset side-table-01", WHITE),
        ("$ python pipeline/7_physics_video.py --asset side-table-01", WHITE),
        ("$ python pipeline/8_render_video.py --asset side-table-01", WHITE),
    ]
    n = int(duration * FPS)
    f = font(21)
    for i in range(n):
        img = base_frame()
        d = ImageDraw.Draw(img)
        # terminal window
        d.rounded_rectangle([70, 90, 1210, 625], radius=12,
                            fill=(8, 12, 22), outline=(52, 70, 96), width=2)
        d.rectangle([70, 90, 1210, 134], fill=(22, 32, 50))
        for k, c in enumerate([(224, 96, 96), (224, 182, 84), (96, 190, 120)]):
            d.ellipse([94 + k * 26, 104, 108 + k * 26, 118], fill=c)
        d.text((200, 103), "ArticForge 流水线  ·  命令演示（不实际调用 API）",
               font=font(19), fill=GREY)
        # finish revealing at ~70% of the segment so all commands stay readable
        shown = int(min(len(lines), max(1, (i + 1) / (n * 0.7) * len(lines))))
        for j, (text, color) in enumerate(lines[:shown]):
            y = 156 + j * 36
            if text.startswith("$"):
                d.text((100, y), text, font=f, fill=color)
                d.rectangle([88, y + 2, 91, y + 24], fill=ACCENT)
            else:
                d.text((100, y), text, font=f, fill=color)
        d.rectangle([0, H - 92, W, H], fill=(8, 12, 22))
        centered(d, "端到端流水线：云端 2 步 + 本地 6 步 · 单件约 41 积分 / 10 分钟",
                 H - 76, 30, WHITE)
        emit(writer_global, img, i, n, 6)


def show_image(path, duration, caption, zoom=0.0, label=None):
    n = int(duration * FPS)
    src = Image.open(path).convert("RGB")
    for i in range(n):
        t = i / max(n - 1, 1)
        img = base_frame()
        scale = 1.0 + zoom * t
        sw, sh = int(W * 0.82 * scale), int(H * 0.72 * scale)
        thumb = src.resize((sw, sh), Image.LANCZOS)
        x, y = (W - sw) // 2, (H - sh) // 2 - 20
        img.paste(thumb, (x, y))
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 92, W, H], fill=(8, 12, 22))
        centered(d, caption, H - 76, 32, WHITE)
        if label:
            d.rounded_rectangle([40, 40, 40 + 14 * len(label) + 30, 92], radius=10,
                                fill=(20, 30, 46), outline=ACCENT, width=2)
            d.text((55, 52), label, font=font(26), fill=ACCENT)
        emit(writer_global, img, i, n, 6)


def asset_grid(duration=3.0):
    n = int(duration * FPS)
    assets = [("side-table-01", "边桌"), ("wardrobe-01", "衣柜"),
              ("nightstand-02", "床头柜"), ("oven-mini-01", "迷你烤箱"),
              ("toolbox-01", "工具箱")]
    imgs = []
    for name, label in assets:
        im = Image.open(ROOT / "docs" / "report-assets" / f"{name}.png").convert("RGB")
        imgs.append((im.resize((360, 203), Image.LANCZOS), label))
    for i in range(n):
        img = base_frame()
        d = ImageDraw.Draw(img)
        centered(d, "五件一句话生成的关节物体资产", 60, 42, WHITE)
        for k, (im, label) in enumerate(imgs):
            row, col = divmod(k, 3)
            x = 60 + col * 395
            y = 160 + row * 260
            img.paste(im, (x, y))
            d.rectangle([x, y + 203, x + 360, y + 245], fill=(20, 30, 46))
            f = font(24)
            box = d.textbbox((0, 0), label, font=f)
            d.text((x + (360 - (box[2] - box[0])) // 2, y + 209), label, font=f, fill=ACCENT)
        emit(writer_global, img, i, n, 8)


def video_clip(path, duration, caption, max_frames=None):
    reader = __import__("imageio.v2", fromlist=["get_reader"]).get_reader(path)
    meta = reader.get_meta_data()
    sw, sh = meta["size"]
    n = int(duration * FPS)
    frames = []
    nf = meta.get("nframes", None)
    if nf is None or not np.isfinite(nf):
        nf = 10 ** 9
    limit = min(max_frames or 10 ** 9, int(nf))
    for idx in np.linspace(0, min(limit - 1, 200), n).astype(int):
        try:
            frames.append(Image.fromarray(reader.get_data(int(idx))).convert("RGB"))
        except Exception:
            break
    reader.close()
    if not frames:
        return
    for i, src in enumerate(frames):
        img = base_frame()
        # letterbox preserving aspect ratio
        factor = min(W / sw, (H - 100) / sh)
        nw, nh = int(sw * factor), int(sh * factor)
        thumb = src.resize((nw, nh), Image.LANCZOS)
        img.paste(thumb, ((W - nw) // 2, (H - 100 - nh) // 2))
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 92, W, H], fill=(8, 12, 22))
        centered(d, caption, H - 76, 32, WHITE)
        emit(writer_global, img, i, len(frames), 6)


def metrics_card(duration=5.0):
    n = int(duration * FPS)
    items = [("5 件", "真实资产交付"), ("¥4 / 10 分钟", "单件成本"),
             ("16/16", "离线回归"), ("22/22", "跨引擎冒烟")]
    for i in range(n):
        img = base_frame()
        d = ImageDraw.Draw(img)
        centered(d, "ArticForge", 100, 64, WHITE)
        centered(d, "一句话生成可训练关节物体资产", 195, 32, ACCENT)
        for k, (big, small) in enumerate(items):
            x = 90 + (k % 2) * 570
            y = 300 + (k // 2) * 150
            d.rounded_rectangle([x, y, x + 500, y + 125], radius=16,
                                fill=(24, 34, 52), outline=ACCENT, width=2)
            d.text((x + 28, y + 18), big, font=font(40), fill=WHITE)
            d.text((x + 28, y + 76), small, font=font(24), fill=GREY)
        centered(d, "github.com/milu348/ArticForge", 620, 28, GREY)
        emit(writer_global, img, i, n, 10)


writer_global = None


def main():
    global writer_global
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/report-assets/ArticForge-demo.mp4")
    args = ap.parse_args()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    import imageio
    writer_global = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8,
                                       macro_block_size=1)

    # 1. 片头
    card([("ArticForge", 78, WHITE, 24),
          ("一句话生成可训练关节物体资产", 40, ACCENT, 40),
          ("文本 → 照片级 3D → 带力标定的 URDF → 可训练仿真", 28, GREY, 0)], 4.5)

    # 2. 痛点
    card([("具身智能的资产瓶颈", 52, WHITE, 50),
          ("手工资产：标注贵、品类少、无法按需扩展", 30, GREY, 22),
          ("LLM 规则生成：几何粗糙、没有物理参数", 30, GREY, 0)], 4.0)

    # 3. 流水线
    flow_card(4.0)
    terminal_card(8.0)   # 仅展示命令，不调用 LUX3D API

    # 4. 五件资产
    asset_grid(3.5)
    for name, label in [("side-table-01", "边桌 · 抽屉"),
                        ("wardrobe-01", "衣柜 · 对开门"),
                        ("nightstand-02", "床头柜 · 抽屉+门"),
                        ("oven-mini-01", "烤箱 · 下翻门"),
                        ("toolbox-01", "工具箱 · 托盘")]:
        show_image(ROOT / "docs" / "report-assets" / f"{name}.png", 1.6,
                   f"{label}  ·  真实可开合关节", zoom=0.05, label=label)

    # 5. 关节开合（运动学演示）
    video_clip(ROOT / "assets/side-table-01/side-table-01_demo.mp4", 6.0,
               "关节开合：抽屉拉出 → 回关（MuJoCo 运动学演示）")

    # 6. 力驱动物理仿真
    video_clip(ROOT / "assets/side-table-01/side-table-01_physics.mp4", 5.0,
               "力驱动仿真：摩擦、限位、接触真实生效")

    # 7. 力标定
    show_image(ROOT / "assets/side-table-01/force_curves.png", 4.0,
               "逐关节力标定：启动力 1.35N · 全程 33.3cm · 行程 100%")

    # 8. Panda 机械臂
    video_clip(ROOT / "docs/report-assets/panda-close-drawer-offscreen.mp4", 5.8,
               "ManiSkill3 / SAPIEN：Panda 机械臂推关抽屉")

    # 9. 片尾
    metrics_card(5.0)

    writer_global.close()
    print(f"demo video: {out}")


if __name__ == "__main__":
    main()
