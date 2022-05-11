"""
https://www.shadertoy.com/view/Ms2SD1

"Seascape" by Alexander Alekseev aka TDM - 2014
License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
"""
import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu)

W, H = 800, 640
img = ti.Vector.field(3, ti.f32, shape=(W, H))

NUM_STEPS = 8
EPSILON	= 1e-3
EPSILON_NRM = 0.1 / W
AA = 1

ITER_GEOMETRY = 3
ITER_FRAGMENT = 5
SEA_HEIGHT = 0.6
SEA_CHOPPY = 4.0
SEA_SPEED = 0.8
SEA_FREQ = 0.16
SEA_BASE = vec3(0.0, 0.09, 0.18)
SEA_WATER_COLOR = vec3(0.8, 0.9, 0.6) * 0.6
octave_m = mat2([[1.6, 1.2], [-1.2, 1.6]])


@ti.func
def fromEuler(ang):
    a1 = vec2(sin(ang.x), cos(ang.x))
    a2 = vec2(sin(ang.y), cos(ang.y))
    a3 = vec2(sin(ang.z), cos(ang.z))
    return mat3([
        [
            a1.y *a3.y + a1.x * a2.x * a3.x,
            a1.y *a2.x * a3.x + a3.y * a1.x,
            -a2.y * a3.x
        ],
        [
            -a2.y * a1.x, a1.y * a2.y, a2.x
        ],
        [
            a3.y * a1.x * a2.x + a1.y * a3.x,
            a1.x * a3.x - a1.y * a3.y * a2.x,
            a2.y * a3.y
        ]
    ])


@ti.func
def hash21(p):
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123)


@ti.func
def noise(p):
    ip = floor(p)
    fp = fract(p)
    u = fp * fp * (3.0 - 2.0 * fp)
    a = hash21(ip + vec2(0, 0))
    b = hash21(ip + vec2(1, 0))
    c = hash21(ip + vec2(0, 1))
    d = hash21(ip + vec2(1, 1))
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2 - 1


@ti.func
def diffuse(nor, lig, exponent):
    return pow(dot(nor, lig) * 0.4 + 0.6, exponent)


@ti.func
def specular(nor, lig, e, s):
    nrm = (s + 8.0) / (pi * 8.0)
    return pow(max(dot(reflect(e, nor), lig), 0), s) * nrm


@ti.func
def getSkyColor(e):
    e.y = (max(e.y, 0.0) * 0.8 + 0.2) * 0.8
    return vec3(
        pow(1.0 - e.y, 2.0),
        1.0 - e.y,
        0.6 + (1.0 - e.y) * 0.4
    ) * 1.1


@ti.func
def sea_octave(uv, choppy):
    uv += noise(uv)
    wv = 1.0 - abs(sin(uv))
    swv = abs(cos(uv))
    wv = mix(wv, swv, wv)
    return pow(1 - pow(wv.x * wv.y, 0.65), choppy)


@ti.func
def seatime(time):
    return 1.0 + time * SEA_SPEED


@ti.func
def map(p, time):
    time = seatime(time)
    freq = SEA_FREQ
    amp = SEA_HEIGHT
    choppy = SEA_CHOPPY
    uv = p.xz
    uv.x *= 0.75

    d = h = 0.0
    for i in range(ITER_GEOMETRY):
        d = sea_octave((uv + time) * freq, choppy)
        d += sea_octave((uv - time) * freq, choppy)
        h += d * amp
        uv = octave_m @ uv
        freq *= 1.9
        amp *= 0.22
        choppy = mix(choppy, 1.0, 0.2)

    return p.y - h


@ti.func
def map_detailed(p, time):
    time = seatime(time)
    freq = SEA_FREQ
    amp = SEA_HEIGHT
    choppy = SEA_CHOPPY
    uv = p.xz
    uv.x *= 0.75

    d = h = 0.0
    for i in range(ITER_FRAGMENT):
        d = sea_octave((uv + time) * freq, choppy)
        d += sea_octave((uv - time) * freq, choppy)
        h += d * amp
        uv = octave_m @ uv
        freq *= 1.9
        amp *= 0.22
        choppy = mix(choppy, 1.0, 0.2)

    return p.y - h


@ti.func
def getSeaColor(p, n, l, eye, dist):
    fresnel = clamp(1.0 - dot(n, -eye), 0.0, 1.0)
    fresnel = pow(fresnel, 3.0) * 0.5
    reflected = getSkyColor(reflect(eye, n))
    refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12
    color = mix(refracted, reflected, fresnel)
    atten = max(1.0 - dot(dist, dist) * 0.001, 0.0)
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten
    color += vec3(specular(n, l, eye, 60.0))
    return color


@ti.func
def getNormal(p, eps, time):
    n = vec3(0)
    n.y = map_detailed(p, time)
    n.x = map_detailed(vec3(p.x + eps, p.y, p.z), time) - n.y
    n.z = map_detailed(vec3(p.x, p.y, p.z + eps), time) - n.y
    n.y = eps
    return normalize(n)


@ti.func
def heightMapTracing(ro, rd, time):
    p = vec3(0)
    tm = 0.0
    tx = 1000.0
    hx = map(ro + rd * tx, time)
    if hx > 0.0:
        p = ro + rd * tx

    else:
        hm = map(ro + rd * tm, time)
        tmid = 0.0
        for i in range(NUM_STEPS):
            tmid = mix(tm, tx, hm / (hm - hx))
            p = ro + rd * tmid
            hmid = map(p, time)
            if hmid < 0.0:
                tx = tmid
                hx = hmid
            else:
                tm = tmid
                hm = hmid
    return p


@ti.func
def getPixel(coord, time):
    uv = 2 * coord / vec2(W, H) - 1.0
    uv.x *= W / H
    ang = vec3(sin(time * 3) * 0.1, sin(time) * 0.2 + 0.3, time)
    ro = vec3(0.0, 3.5, time * 5.0)
    rd = normalize(vec3(uv.xy, -2.0))
    rd.z += length(uv) * 0.14
    rd = fromEuler(ang) @ normalize(rd)

    p = heightMapTracing(ro, rd, time)
    dist = p - ro
    n = getNormal(p, dot(dist, dist) * EPSILON_NRM, time)
    light = normalize(vec3(0.0, 1.0, 0.8))

    return mix(
        getSkyColor(rd),
        getSeaColor(p, n, light, rd, dist),
        pow(smoothstep(0, -0.02, rd.y), 0.2)
    )


@ti.kernel
def step(t: float, mx: float, my: float):
    time = t * 0.3 + mx * 0.01
    for i, j in img:
        color = vec3(0)
        for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
            uv = vec2(i, j) + vec2(dx, dy) / 3
            color += getPixel(uv, time)

        color /= 9.0
        color = pow(color, 0.65)
        img[i, j] = color


def main():
    gui = ti.GUI('Seascape', res=(W, H))
    t = 0.0
    while gui.running:
        mouse_x = mouse_y = 0.0
        gui.get_event(ti.GUI.PRESS)
        if gui.is_pressed(ti.GUI.LMB):
            mouse_x, mouse_y = gui.get_cursor_pos()

        if gui.is_pressed(ti.GUI.ESCAPE):
            gui.running = False

        if gui.is_pressed('p'):
            gui.set_image(img)
            gui.show('screenshot.png')

        step(t, mouse_x, mouse_y)
        t += 0.03
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
