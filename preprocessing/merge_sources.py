import os
import re
import numpy as np
import cv2
import cairosvg
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Paths to datasets (adjust as needed)
sketch_dir = "../dataset/sketch_dataset"
normalmap_dir = "../dataset/normal_map_dataset"

# Pick a sample (for demo, just pick the first matching pair)
sketch_files = sorted([f for f in os.listdir(sketch_dir) if f.endswith('.svg')])
normalmap_files = sorted([f for f in os.listdir(normalmap_dir) if f.endswith('.png')])

# Find a matching normal map for each sketch (by basename, ignoring extension)
import re
sketch_to_nm = {}
for sketch_file in sketch_files:
    # Match sketch and normal map by shared prefix before "_sketch_var" and "_nm_var"
    sketch_pattern = re.compile(r"^(.*)_sketch_var(\d+)\.svg$")
    m = sketch_pattern.match(sketch_file)
    if m:
        base = m.group(1)
        var_part = m.group(2)
        expected_nm = f"{base}_nm_var{var_part}.png"
        if expected_nm in normalmap_files:
            sketch_to_nm[sketch_file] = expected_nm

print(f"Found {len(sketch_to_nm)} sketch/normalmap pairs:")
for k, v in sketch_to_nm.items():
    print(f"  {k} <-> {v}")

if not sketch_to_nm:
    raise Exception("No matching sketch/normal map pairs found.")


def load_hatching_texture(path):
    tex = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tex is None:
        raise RuntimeError(f'Не удалось загрузить текстуру: {path}')
    if tex.ndim == 3 and tex.shape[2] == 4:
        bgr = tex[:, :, :3]
        a   = tex[:, :, 3].astype(np.float32) / 255.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Композитим на белый фон через альфу (прозрачное -> белое)
        gray = gray * a + (1.0 - a)
        return (gray * 255.0).astype(np.uint8)
    elif tex.ndim == 3:
        return cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
    else:
        return tex
    
hatching_tex_path = os.path.join('..', 'preprocessing', 'hatching_texture.png')
hatching_tex = load_hatching_texture(hatching_tex_path)
if hatching_tex is None:
    raise RuntimeError(f'Не удалось загрузить текстуру: {hatching_tex_path}')

# ==== Параметры пайплайна ====
SCALE_FACTOR   = 1.02
OFFSET_XY      = (-10, -10)       # (dx, dy)
SKETCH_ALPHA   = 0.85             # сила линий эскиза поверх
R_THRESH       = 200              # порог для каналов normal map
SHADE_STRENGTH = (1, 0.3, 0.0)  # веса вклада каналов (R,G,B) в серую тень
# ---------------------------------------------------------------

def _strip_namespace(svg_text: str) -> str:
    return svg_text.replace('xmlns="http://www.w3.org/2000/svg"', '')

def _remove_fill_from_style_attr(style: str) -> str:
    # убираем fill:... и fill-opacity:... из inline-style
    style = re.sub(r'fill\s*:\s*[^;]+;?', '', style, flags=re.IGNORECASE)
    style = re.sub(r'fill-opacity\s*:\s*[^;]+;?', '', style, flags=re.IGNORECASE)
    # подчистить возможные ;; и пробелы
    style = re.sub(r';{2,}', ';', style).strip().strip(';').strip()
    return style

def render_svg_to_bgra(svg_path, W, H):
    """
    Растеризует SVG в PNG с прозрачным фоном.
    - Удаляет фоновые <rect> (включая width/height=100%).
    - Удаляет ВСЕ заливки у элементов (оставляет только stroke).
    - Чистит fill в <style> и style="...".
    """
    with open(svg_path, 'r', encoding='utf-8') as f:
        svg_text = f.read()

    svg_text = _strip_namespace(svg_text)
    root = ET.fromstring(svg_text)

    # 1) подчистить <style> внутри svg (убрать fill в CSS)
    for style_node in root.findall('.//style'):
        if style_node.text:
            cleaned = re.sub(r'fill\s*:\s*[^;}\n]+;?', '', style_node.text, flags=re.IGNORECASE)
            cleaned = re.sub(r'fill-opacity\s*:\s*[^;}\n]+;?', '', cleaned, flags=re.IGNORECASE)
            style_node.text = cleaned

    # 2) удалить фоновые прямоугольники
    for rect in list(root.findall('.//rect')):
        w = rect.attrib.get('width', '').strip()
        h = rect.attrib.get('height', '').strip()
        has_fill = 'fill' in rect.attrib or ('style' in rect.attrib and re.search(r'fill\s*:', rect.attrib['style'], re.I))
        covers_all = (w in ('100%', '100') and h in ('100%', '100')) or (rect.attrib.get('x','0')=='0' and rect.attrib.get('y','0')=='0' and has_fill)
        if has_fill and (covers_all or True):  # чаще фоновый — любой залитый rect
            try:
                parent = root
                # найти реального родителя (ElementTree не хранит ссылку — делаем простой обход)
                stack = [root]
                while stack:
                    node = stack.pop()
                    for ch in list(node):
                        if ch is rect:
                            parent = node
                        stack.append(ch)
                parent.remove(rect)
            except Exception:
                pass  # если не удалось — просто оставим

    # 3) рекурсивно убрать все заливки у остальных элементов
    def strip_fills(node: ET.Element):
        # убираем fill атрибуты
        if 'fill' in node.attrib:
            node.set('fill', 'none')
        if 'fill-opacity' in node.attrib:
            node.set('fill-opacity', '0')
        # вычищаем fill из style
        if 'style' in node.attrib:
            style_clean = _remove_fill_from_style_attr(node.attrib['style'])
            if style_clean:
                node.set('style', style_clean)
            else:
                del node.attrib['style']
        for ch in list(node):
            strip_fills(ch)

    strip_fills(root)

    svg_clean = ET.tostring(root, encoding='utf-8').decode('utf-8')

    # 4) Растеризация с прозрачным фоном
    png_bytes = cairosvg.svg2png(
        bytestring=svg_clean.encode('utf-8'),
        output_width=W, output_height=H,
        background_color='rgba(0,0,0,0)'
    )
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError('Failed to rasterize SVG')
    if img.shape[2] == 3:  # добавить альфу, если нет
        a = np.full(img.shape[:2], 255, np.uint8)
        img = np.dstack([img, a])
    return img

def warp_bgra(bgra, M, size):
    W, H = size
    rgb = bgra[:, :, :3]
    a   = bgra[:, :, 3]
    rgb_w = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    a_w   = cv2.warpAffine(a,   M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.dstack([rgb_w, a_w])

def alpha_over(base_bgr, fg_bgra, fg_alpha_scale=1.0):
    base = base_bgr.astype(np.float32)
    fg_rgb = fg_bgra[:, :, :3].astype(np.float32)
    a = (fg_bgra[:, :, 3].astype(np.float32) / 255.0) * float(fg_alpha_scale)
    out = base * (1.0 - a[..., None]) + fg_rgb * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def compute_hatching_mask(
    nm, thresh, strength, hatching_tex, *,
    scale=0.1,
    angles_deg=(-45.0, 45.0, 0.0),
    angle_spread=40.0,
    jitter_deg=6.0,
    scale_jitter=0.2,
    noise_strength=0.18,
    seed=None
):
    rng = np.random.default_rng(seed)

    # ----- веса каналов -----
    R = nm[:, :, 2].astype(np.float32)
    G = nm[:, :, 1].astype(np.float32)
    B = nm[:, :, 0].astype(np.float32)

    def norm(ch):
        return np.clip((ch - thresh) / max(1.0, (255.0 - thresh)), 0.0, 1.0)

    if np.isscalar(strength):
        sR = sG = sB = float(strength)
    else:
        sR, sG, sB = map(float, strength)

    wR = sR * norm(R)
    wG = sG * norm(G)
    wB = sB * norm(B)
    shade_norm = np.clip(wR + wG + wB, 0.0, 1.0)

    wsum = wR + wG + wB + 1e-6
    wR_n, wG_n, wB_n = wR/wsum, wG/wsum, wB/wsum

    H, W = nm.shape[:2]

    # ---- helpers ----
    def _tiled(tex, target_w, target_h, base_scale):
        th, tw = tex.shape[:2]
        s = max(1e-6, base_scale * (1.0 + rng.uniform(-scale_jitter, scale_jitter)))
        new_w = max(1, int(tw * s))
        new_h = max(1, int(th * s))
        small = cv2.resize(tex, (new_w, new_h), interpolation=cv2.INTER_AREA)
        reps_x = int(np.ceil(target_w / new_w)) + 1
        reps_y = int(np.ceil(target_h / new_h)) + 1
        tiled = np.tile(small, (reps_y, reps_x))
        return tiled[:target_h, :target_w]

    def _rotated_tiled(base_tex, angle_deg):
        # Сначала тайлим на весь кадр
        tiled = _tiled(base_tex, W, H, scale).astype(np.float32) / 255.0
        # Затем крутим всю плоскость с заворачиванием границ (без белых углов)
        ang = float(angle_deg) + rng.normal(0.0, jitter_deg)
        M = cv2.getRotationMatrix2D((W*0.5, H*0.5), ang, 1.0)
        # Случайный сдвиг для «texture paint»
        M[0, 2] += rng.uniform(-W*0.5, W*0.5)
        M[1, 2] += rng.uniform(-H*0.5, H*0.5)
        return cv2.warpAffine(
            tiled, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP   # <- главное изменение
        )

    half = angle_spread * 0.5
    ax_lo, ay_lo, az_lo = [a - half for a in angles_deg]
    ax_hi, ay_hi, az_hi = [a + half for a in angles_deg]

    # две ориентации для каждой оси
    tex_x_lo = _rotated_tiled(hatching_tex, ax_lo)
    tex_x_hi = _rotated_tiled(hatching_tex, ax_hi)
    tex_y_lo = _rotated_tiled(hatching_tex, ay_lo)
    tex_y_hi = _rotated_tiled(hatching_tex, ay_hi)
    tex_z_lo = _rotated_tiled(hatching_tex, az_lo)
    tex_z_hi = _rotated_tiled(hatching_tex, az_hi)

    # интерполяция угла по силе тени
    tex_x = tex_x_lo * (1.0 - shade_norm) + tex_x_hi * shade_norm
    tex_y = tex_y_lo * (1.0 - shade_norm) + tex_y_hi * shade_norm
    tex_z = tex_z_lo * (1.0 - shade_norm) + tex_z_hi * shade_norm

    # смешение по направлениям нормали
    hatch = wR_n * tex_x + wG_n * tex_y + wB_n * tex_z  # 0..1

    # «hand-drawn» зерно
    if noise_strength > 0:
        noise = rng.random((H, W)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.2, sigmaY=1.2)
        hatch = np.clip(hatch * (1.0 - 0.5*noise_strength + noise_strength*noise), 0.0, 1.0)

    # чем больше shade_norm — тем темнее (сильнее проявляем штрихи)
    out_gray = 1.0 - shade_norm * (1.0 - hatch)
    out_gray = np.clip(out_gray * 255.0, 0, 255).astype(np.uint8)
    return cv2.merge([out_gray, out_gray, out_gray])

# ==== Основной код ====
if not sketch_to_nm:
    raise RuntimeError('No sketch/normal-map pairs detected')

for idx, (sketch_name, nm_name) in enumerate(list(sketch_to_nm.items())[0:5]):
    svg_path = os.path.join(sketch_dir, sketch_name)
    nm_path  = os.path.join(normalmap_dir, nm_name)

    nm = cv2.imread(nm_path, cv2.IMREAD_COLOR)
    if nm is None:
        raise RuntimeError(f'Cannot read normal map: {nm_path}')
    H, W = nm.shape[:2]

    sketch_bgra = render_svg_to_bgra(svg_path, W, H)

    M = np.array([[SCALE_FACTOR, 0.0, OFFSET_XY[0]],
                  [0.0, SCALE_FACTOR, OFFSET_XY[1]]], dtype=np.float32)
    sketch_bgra = warp_bgra(sketch_bgra, M, (W, H))

    base_bgr = compute_hatching_mask(
        nm, R_THRESH, SHADE_STRENGTH,
        hatching_tex=hatching_tex,
        scale=0.3,
        angles_deg=(-5, 5, 0),   # X->R, Y->G, Z->B
        angle_spread=40,           # насколько «крутить» при strength→1
        jitter_deg=6,
        scale_jitter=0.1,
        noise_strength=0.2,
        seed=42
    )

    out = alpha_over(base_bgr, sketch_bgra, fg_alpha_scale=SKETCH_ALPHA)
    # Сохраняем результат в ../dataset/shadowed_sketches/
    out_dir = os.path.join('..', 'dataset', 'shadowed_sketches')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(sketch_name)[0] + '.png')
    cv2.imwrite(out_path, out)
    print(f"Saved: {out_path}")