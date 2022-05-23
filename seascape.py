"""
https://www.shadertoy.com/view/Ms2SD1

"Seascape" by Alexander Alekseev aka TDM - 2014
License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
"""
import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

W, H = 800, 640
iResolution = tm.vec2(W, H)
iTime = ti.field(float, shape=())
iMouse = ti.Vector.field(2, float, shape=())

img = ti.Vector.field(3, float, shape=(W, H))

NUM_STEPS = 8
EPSILON = 1e-3
EPSILON_NRM = 0.1 / W
AA = 1

ITER_GEOMETRY = 3
ITER_FRAGMENT = 5
SEA_HEIGHT = 0.6
SEA_CHOPPY = 4.0
SEA_SPEED = 0.8
SEA_FREQ = 0.16
SEA_BASE = tm.vec3(0.0, 0.09, 0.18)
SEA_WATER_COLOR = tm.vec3(0.8, 0.9, 0.6) * 0.6
octave_m = tm.mat2([[1.6, -1.2], [1.2, 1.6]])



def init():
    iTime[None] = 0.0
    iMouse[None] = (0, 0)
    return time.perf_counter()


@ti.func
def fromEuler(ang):
    a1 = tm.vec2(tm.sin(ang.x), tm.cos(ang.x))
    a2 = tm.vec2(tm.sin(ang.y), tm.cos(ang.y))
    a3 = tm.vec2(tm.sin(ang.z), tm.cos(ang.z))
    return tm.mat3([
        [
            a1.y * a3.y + a1.x * a2.x * a3.x,
            a1.y * a2.x * a3.x + a3.y * a1.x,
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
    return tm.fract(tm.sin(tm.dot(p, tm.vec2(127.1, 311.7))) * 43758.5453123)


@ti.func
def noise(p):
    ip = tm.floor(p)
    fp = tm.fract(p)
    u = fp * fp * (3.0 - 2.0 * fp)
    a = hash21(ip + tm.vec2(0, 0))
    b = hash21(ip + tm.vec2(1, 0))
    c = hash21(ip + tm.vec2(0, 1))
    d = hash21(ip + tm.vec2(1, 1))
    return tm.mix(tm.mix(a, b, u.x), tm.mix(c, d, u.x), u.y) * 2 - 1


@ti.func
def diffuse(nor, lig, exponent):
    return tm.pow(tm.dot(nor, lig) * 0.4 + 0.6, exponent)


@ti.func
def specular(nor, lig, e, s):
    nrm = (s + 8.0) / (tm.pi * 8.0)
    return tm.pow(tm.max(tm.dot(tm.reflect(e, nor), lig), 0), s) * nrm


@ti.func
def getSkyColor(e):
    e.y = (tm.max(e.y, 0.0) * 0.8 + 0.2) * 0.8
    return tm.vec3(
        tm.pow(1.0 - e.y, 2.0),
        1.0 - e.y,
        0.6 + (1.0 - e.y) * 0.4
    ) * 1.1


@ti.func
def sea_octave(uv, choppy):
    uv += noise(uv)
    wv = 1.0 - abs(tm.sin(uv))
    swv = abs(tm.cos(uv))
    wv = tm.mix(wv, swv, wv)
    return tm.pow(1 - tm.pow(wv.x * wv.y, 0.65), choppy)


@ti.func
def seatime():
    return 1.0 + iTime[None] * SEA_SPEED


@ti.func
def map(p):
    time = seatime()
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
        choppy = tm.mix(choppy, 1.0, 0.2)

    return p.y - h


@ti.func
def map_detailed(p):
    time = seatime()
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
        choppy = tm.mix(choppy, 1.0, 0.2)

    return p.y - h


@ti.func
def getSeaColor(p, n, l, eye, dist):
    fresnel = tm.clamp(1.0 - tm.dot(n, -eye), 0.0, 1.0)
    fresnel = tm.pow(fresnel, 3.0) * 0.5
    reflected = getSkyColor(tm.reflect(eye, n))
    refracted = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12
    color = tm.mix(refracted, reflected, fresnel)
    atten = tm.max(1.0 - tm.dot(dist, dist) * 0.001, 0.0)
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten
    color += tm.vec3(specular(n, l, eye, 60.0))
    return color


@ti.func
def getNormal(p, eps):
    n = tm.vec3(0)
    n.y = map_detailed(p)
    n.x = map_detailed(tm.vec3(p.x + eps, p.y, p.z)) - n.y
    n.z = map_detailed(tm.vec3(p.x, p.y, p.z + eps)) - n.y
    n.y = eps
    return tm.normalize(n)


@ti.func
def heightMapTracing(ro, rd):
    p = tm.vec3(0)
    tmin = 0.0
    tmax = 1000.0
    hmax = map(ro + rd * tmax)
    if hmax > 0.0:
        p = ro + rd * tmax

    else:
        hmin = map(ro + rd * tmin)
        tmid = 0.0
        for i in range(NUM_STEPS):
            tmid = tm.mix(tmin, tmax, hmin / (hmin - hmax))
            p = ro + rd * tmid
            hmid = map(p)
            if hmid < 0.0:
                tmax = tmid
                hmax = hmid
            else:
                tmin = tmid
                hmin = hmid
    return p


@ti.func
def getPixel(coord, time):
    uv = 2 * coord / iResolution - 1.0
    uv.x *= W / H
    ang = tm.vec3(tm.sin(time * 3) * 0.1, tm.sin(time) * 0.2 + 0.3, time)
    ro = tm.vec3(0.0, 3.5, time * 5.0)
    rd = tm.normalize(tm.vec3(uv.xy, -2.0))
    rd.z += tm.length(uv) * 0.14
    rd = fromEuler(ang) @ tm.normalize(rd)

    p = heightMapTracing(ro, rd)
    dist = p - ro
    n = getNormal(p, tm.dot(dist, dist) * EPSILON_NRM)
    light = tm.normalize(tm.vec3(0.0, 1.0, 0.8))

    return tm.mix(
        getSkyColor(rd),
        getSeaColor(p, n, light, rd, dist),
        tm.pow(tm.smoothstep(0, -0.02, rd.y), 0.2)
    )


@ti.kernel
def step():
    time = iTime[None] * 0.3 + iMouse[None].x * 0.01
    for i, j in img:
        color = tm.vec3(0)
        for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
            uv = tm.vec2(i, j) + tm.vec2(dx, dy) / 3
            color += getPixel(uv, time)

        color /= 9.0
        color = tm.pow(color, 0.65)
        img[i, j] = color


def main():
    t0 = init()
    gui = ti.GUI('Seascape', res=(W, H), fast_gui=True)
    while gui.running:
        gui.get_event(ti.GUI.PRESS)
        if gui.is_pressed(ti.GUI.LMB):
            mouse_x, mouse_y = gui.get_cursor_pos()
            iMouse[None] = tm.vec2(mouse_x, mouse_y) * iResolution

        if gui.is_pressed(ti.GUI.ESCAPE):
            gui.running = False

        if gui.is_pressed('p'):
            gui.set_image(img)
            gui.show('screenshot.png')

        step()
        iTime[None] = time.perf_counter() - t0
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
