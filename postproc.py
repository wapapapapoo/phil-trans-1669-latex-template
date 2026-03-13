import os
import random
import numpy as np
import moderngl
import fitz
import math

from PIL import Image, ImageDraw

# -------------------------
# output folder
# -------------------------

os.makedirs("dist", exist_ok=True)

# -------------------------
# open pdf
# -------------------------

pdf = fitz.open("main.pdf")

dpi = 600
zoom = dpi / 72
matrix = fitz.Matrix(zoom, zoom)

# -------------------------
# create GL context
# -------------------------

ctx = moderngl.create_standalone_context()

# -------------------------
# shaders
# -------------------------

diff_prog = ctx.program(
vertex_shader="""
#version 330
in vec2 pos;
out vec2 uv;
void main(){
    uv = (pos+1.0)*0.5;
    gl_Position = vec4(pos,0,1);
}
""",
fragment_shader="""
#version 330

uniform sampler2D tex;
uniform vec2 resolution;
uniform float diffusion;

uniform float noiseLowScale;
uniform float noiseHighScale;

uniform float noiseLowAmp;
uniform float noiseHighAmp;

uniform float noiseOffset;

uniform vec2 seedLow;
uniform vec2 seedHigh;

in vec2 uv;
out vec4 fragColor;

float hash(vec2 p){
    p = fract(p*vec2(123.34,456.21));
    p += dot(p,p+45.32);
    return fract(p.x*p.y);
}

float noise(vec2 p){
    vec2 i=floor(p);
    vec2 f=fract(p);

    float a=hash(i);
    float b=hash(i+vec2(1,0));
    float c=hash(i+vec2(0,1));
    float d=hash(i+vec2(1,1));

    vec2 u=f*f*(3.0-2.0*f);

    return mix(a,b,u.x)
         + (c-a)*u.y*(1.0-u.x)
         + (d-b)*u.x*u.y;
}

void main(){

    vec2 t = 1.0 / resolution;

    float c  = texture(tex, uv).r;

    float n1 = texture(tex, uv+vec2(t.x,0)).r;
    float n2 = texture(tex, uv-vec2(t.x,0)).r;
    float n3 = texture(tex, uv+vec2(0,t.y)).r;
    float n4 = texture(tex, uv-vec2(0,t.y)).r;

    float d1 = texture(tex, uv+vec2(t.x,t.y)).r;
    float d2 = texture(tex, uv+vec2(-t.x,t.y)).r;
    float d3 = texture(tex, uv+vec2(t.x,-t.y)).r;
    float d4 = texture(tex, uv+vec2(-t.x,-t.y)).r;

    float lap =
        0.5*(n1+n2+n3+n4)
      + 0.25*(d1+d2+d3+d4)
      - 3.0*c;

    float low  = noise(uv * noiseLowScale  + seedLow)  * noiseLowAmp;
    float high = noise(uv * noiseHighScale + seedHigh) * noiseHighAmp;

    float fiber = low + high + noiseOffset;

    float D = diffusion * (0.5 + fiber);

    float result = c + D * lap;

    result = clamp(result,0.0,1.0);

    fragColor = vec4(result,result,result,1.0);
}
"""
)

th_prog = ctx.program(
vertex_shader="""
#version 330
in vec2 pos;
out vec2 uv;
void main(){
    uv = (pos+1.0)*0.5;
    gl_Position = vec4(pos,0,1);
}
""",
fragment_shader="""
#version 330
uniform sampler2D tex;
in vec2 uv;
out vec4 fragColor;
void main(){
    float v = texture(tex, uv).r;
    float th = 0.45;
    float k = 20.0;
    float ink = 1.0 / (1.0 + exp(-k*(v-th)));
    fragColor = vec4(ink,ink,ink,1.0);
}
"""
)

quad = np.array([
-1,-1,
1,-1,
-1,1,
1,1
], dtype="f4")

vbo = ctx.buffer(quad.tobytes())

diff_vao = ctx.simple_vertex_array(diff_prog, vbo, "pos")
th_vao   = ctx.simple_vertex_array(th_prog, vbo, "pos")

# -------------------------
# process each page
# -------------------------

transforms = []

for page_index, page in enumerate(pdf):
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)

    img = Image.frombytes("L", [pix.width, pix.height], pix.samples)

    w, h = img.size

    # -------------------------
    # scan border
    # -------------------------

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0,0,w-1,h-1],
        outline=random.randint(0,40),
        width=2
    )

    # -------------------------
    # padding
    # -------------------------

    pad = 48

    padded = Image.new("L", (w + pad*2, h + pad*2), 255)
    padded.paste(img, (pad, pad))

    img = padded
    w, h = img.size

    # -------------------------
    # rotation
    # -------------------------

    angle = random.gauss(0, 1.)

    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=255
    )

    # -------------------------
    # translation
    # -------------------------

    dx = random.randint(-32,32)
    dy = random.randint(-32,32)

    canvas = Image.new("L",(w,h),255)
    canvas.paste(img,(dx,dy))

    img = canvas

    # -------------------------
    # simulation data
    # -------------------------

    data = np.array(img,dtype=np.float32)/255.0
    data = 1.0 - data

    # save transform arg
    transforms.append((angle, dx, dy, pad))

    # -------------------------
    # textures
    # -------------------------

    texA = ctx.texture((w,h),1,data.astype("f4").tobytes(),dtype="f4")
    texB = ctx.texture((w,h),1,dtype="f4")

    texA.filter = (moderngl.NEAREST,moderngl.NEAREST)
    texB.filter = (moderngl.NEAREST,moderngl.NEAREST)

    fboA = ctx.framebuffer(color_attachments=[texA])
    fboB = ctx.framebuffer(color_attachments=[texB])

    # uniforms

    diff_prog["resolution"].value = (w,h)
    diff_prog["diffusion"].value = 0.2

    diff_prog["noiseLowScale"].value  = 2.0
    diff_prog["noiseHighScale"].value = 120.0

    diff_prog["noiseLowAmp"].value  = 2.4
    diff_prog["noiseHighAmp"].value = 1.0

    diff_prog["noiseOffset"].value = -1.0

    diff_prog["seedLow"].value  = (12.3,45.6)
    diff_prog["seedHigh"].value = (78.9,10.1)

    # diffusion

    iterations = 15

    for i in range(iterations):

        texA.use()
        fboB.use()

        diff_vao.render(moderngl.TRIANGLE_STRIP)

        texA, texB = texB, texA
        fboA, fboB = fboB, fboA

    # threshold pass

    texA.use()
    fboB.use()

    th_vao.render(moderngl.TRIANGLE_STRIP)

    texA, texB = texB, texA

    # read result

    raw = texA.read()
    result = np.frombuffer(raw,dtype=np.float32).reshape(h,w)

    result = 1.0 - result
    result = (result*255).astype(np.uint8)

    out_path = f"dist/page_{page_index+1}.png"

    Image.fromarray(result).save(out_path)

import math

# -------------------------
# build searchable pdf
# -------------------------

out_pdf = fitz.open()

def transform_point(x, y, cx, cy, angle_deg, dx, dy, pad):
    # padding
    x += pad
    y += pad

    # rotate around center
    rad = math.radians(angle_deg)
    x0 = x - cx
    y0 = y - cy

    xr = x0*math.cos(rad) - y0*math.sin(rad)
    yr = x0*math.sin(rad) + y0*math.cos(rad)

    x = xr + cx
    y = yr + cy

    # translation
    x += dx
    y += dy

    return x, y





# -------------------------
# build searchable pdf
# -------------------------

out_pdf = fitz.open()

for page_index, page in enumerate(pdf):

    angle, dx, dy, pad = transforms[page_index]

    img_path = f"dist/page_{page_index+1}.png"

    img = Image.open(img_path)
    w, h = img.size

    page_out = out_pdf.new_page(width=w, height=h)

    page_out.insert_image(
        fitz.Rect(0,0,w,h),
        filename=img_path
    )

    zoom = dpi / 72

    text_dict = page.get_text("dict")

    for block in text_dict["blocks"]:

        if block["type"] != 0:
            continue

        for line in block["lines"]:

            spans = line["spans"]

            if not spans:
                continue

            # ---------- reconstruct full line text ----------
            parts = []
            prev_x1 = None

            for s in spans:

                t = s["text"]

                if prev_x1 is not None:
                    gap = s["bbox"][0] - prev_x1

                    # heuristic: detect word break
                    if gap > s["size"] * 0.25:
                        parts.append(" ")

                parts.append(t)

                prev_x1 = s["bbox"][2]

            text = "".join(parts)

            # ---------- line anchor ----------
            x0, y0, x1, y1 = spans[0]["bbox"]

            x = x0 * zoom + pad
            y = y1 * zoom + pad

            page_out.insert_text(
                (x, y),
                text,
                fontsize=spans[0]["size"] * zoom,
                render_mode=3
            )

out_pdf.save("dist/output.pdf")
