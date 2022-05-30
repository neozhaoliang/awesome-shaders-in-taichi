import taichi as ti
import taichi.math as tm
from PIL import Image
import numpy as np
ti.init(arch=ti.vulkan)

kMatGround = 0
kMatPlasticRed = 1
kMatWood = 2
kMatLead = 3
IOR = 1.584
flip = 1.0

W, H = 800, 450
iResolution = tm.vec2(W, H)
texture_res = tm.ivec2(64, 64)
material = ti.Vector.field(2, int, shape=(W, H))
iChannel0 = ti.Vector.field(4, float, shape=(W, H))
iChannel1 = ti.Vector.field(4, float, shape=texture_res)
img = ti.Vector.field(4, float, shape=(W, H))


def load_texture(texture, image_file):
    """Load a background image to a Taichi field.
    """
    img = Image.open(image_file).resize(texture_res).convert("RGBA")
    data = np.asarray(img).astype(np.float32) / 255.0
    texture.from_numpy(data)


@ti.kernel
def init():
    for i, j in material:
        material[i, j][0] = 0
        material[i, j][1] = 1

    for i, j in img:
        img[i, j] = 0, 0, 0, 0

    for i, j in iChannel0:
        iChannel0[i, j] = 0, 0, 0, 0

    for i, j in iChannel1:
        iChannel1[i, j] = tm.vec4(ti.random())


@ti.func
def texture(tex, uv):
    p = tm.mod(uv * texture_res, texture_res)
    ip = int(p)
    fp = p - ip
    i0, j0 = ip
    i1 = min(texture_res.x, i0 + 1)
    j1 = min(texture_res.y, j0 + 1)
    a = tex[i0, j0]
    b = tex[i1, j0]
    c = tex[i0, j1]
    d = tex[i1, j1]
    return tm.mix(tm.mix(a, b, fp.x), tm.mix(c, d, fp.x), fp.y)


@ti.func
def hash2():
    return tm.vec2(ti.random(), ti.random())


@ti.func
def sdBox(p, b, r=0):
    q = abs(p) - (b - r)
    return tm.length(max(q, 0)) + min(max(q.x, q.y), 0) - r


@ti.func
def sdRoundedCylinder(p, radius, halfHeight, bevel):
    p2 = tm.vec2(tm.length(p.xz), p.y)
    return sdBox(p2, tm.vec2(radius, halfHeight), bevel)


@ti.func
def hexa(p, r1, r2):
    ang = tm.pi / 3
    x = (r1 - r2) / tm.tan(ang) + r2
    v = tm.vec2(x, r1)
    rot = tm.rot2(-ang)
    hex1 = sdBox(p, v, r2)
    p.xy = rot @ p.xy
    hex2 = sdBox(p, v, r2)
    p.xy = rot @ p.xy
    hex3 = sdBox(p, v, r2)
    return min(hex1, min(hex2, hex3))


@ti.func
def sdCone(p, c):
    q = tm.vec2(tm.length(p.xz), -p.y)
    d = tm.length(q - c * max(tm.dot(q, c), 0))
    sign = -1.0 if q.x * c.y - q.y * c.x < 0.0 else 1.0
    return d * sign


@ti.func
def scene(p, fragCoord):
    ground = p.y - (tm.cos(min(100, p.z) * 0.03) - 1) * 2
    p.y -= 4
    p.xz = tm.rot2(2.1) @ p.xz
    p.z -= 10
    a = 0.23
    cone = sdCone(p.xzy, tm.vec2(tm.sin(a), tm.cos(a)))
    paintShell = hexa(p.xy, 4, 1)
    woodCenter = hexa(p.xy, 3.95, 0.95)
    leadCore = tm.length(p.xy) - 1.1

    paintShell = max(paintShell, max(cone, -woodCenter))
    woodCenter = max(woodCenter, cone)

    woodCenter = max(woodCenter, 0.01 - leadCore)
    if material[fragCoord][1]:
        woodCenter += (texture(iChannel1, p.xy * 0.25).r - 0.4) * 0.1
    leadCore = max(leadCore, cone)
    leadCore = max(leadCore, p.z + 1)
    leadCore = min(leadCore, tm.length(p + tm.vec3(0,0,1.05)) - tm.sin(a))
    leadCore += (texture(iChannel1, tm.vec2(tm.atan2(p.x, p.z) * 0.5)).r - 0.3) * 0.05

    best = ground
    best = min(best, paintShell)
    best = min(best, woodCenter)
    best = min(best, leadCore)

    if best == ground:
        material[fragCoord][0] = kMatGround
    elif best == leadCore:
        material[fragCoord][0] = kMatLead
    elif best == woodCenter:
        material[fragCoord][0] = kMatWood
    elif best == paintShell:
        material[fragCoord][0] = kMatPlasticRed
    return best


@ti.func
def ortho(a):
    return tm.cross(tm.vec3(-1), a)


@ti.func
def getSampleBiased(dir, power):
    dir = tm.normalize(dir)
    o1 = tm.normalize(ortho(dir))
    o2 = tm.normalize(tm.cross(dir, o1))
    r = hash2()
    r.x *= 2 * tm.pi
    r.y = tm.pow(r.y, 1 / (power + 1.0))
    oneminus = tm.sqrt(1 - r.y * r.y)
    return tm.cos(r.x) * oneminus * o1 + tm.sin(r.x) * oneminus * o2 + r.y * dir


@ti.func
def getConeSample(dir, extent):
    dir = tm.normalize(dir)
    o1 = tm.normalize(ortho(dir))
    o2 = tm.normalize(tm.cross(dir, o1))
    r = hash2()
    r.x *= 2 * tm.pi
    r.y = 1 - r.y * extent
    oneminus = tm.sqrt(1 - r.y * r.y)
    return tm.cos(r.x) * oneminus * o1 + tm.sin(r.x) * oneminus * o2 + r.y * dir


@ti.func
def sky(sunDir, viewDir):
    softlight = max(0, tm.dot(tm.normalize(sunDir * tm.vec3(-1, 1, -1)), viewDir) + 0.2)
    keylight = tm.pow(max(0, tm.dot(sunDir, viewDir) - 0.5), 3)
    return tm.vec3(softlight * 0.015 + keylight * 10) * 1.5


@ti.func
def trace5(cam, dir, nearClip, h: ti.template(), n: ti.template(), k: ti.template(), fragCoord):
    t = nearClip
    for _ in range(100):
        k = scene(cam + dir * t, fragCoord) * flip
        if abs(k) < 0.001:
            break
        t += k

    h = cam + dir * t

    result = False
    if abs(k) < 0.001:
        o = tm.vec2(.001, 0)
        n = tm.normalize(tm.vec3(
            scene(h + o.xyy, fragCoord) - k,
            scene(h + o.yxy, fragCoord) - k,
            scene(h + o.yyx, fragCoord) - k
        )) * flip
        result = True
    return result


@ti.func
def floorPattern(uv):
    kUnit1 = 10.0
    kUnit2 = 5.0
    kUnit3 = 1.0
    kThick1 = 0.10
    kThick2 = 0.05
    kThick3 = 0.03

    uv1 = abs(tm.mod(uv, kUnit1) - kUnit1 * 0.5)
    uv2 = abs(tm.mod(uv, kUnit2) - kUnit2 * 0.5)
    uv3 = abs(tm.mod(uv, kUnit3) - kUnit3 * 0.5)
    lines1 = -max(uv1.x, uv1.y) + kUnit1 * 0.5 - kThick1
    lines2 = -max(uv2.x, uv2.y) + kUnit2 * 0.5 - kThick2
    lines3 = -max(uv3.x, uv3.y) + kUnit3 * 0.5 - kThick3
    return min(lines1, min(lines2, lines3))


@ti.func
def trace2(cam, dir, nearClip, fragCoord):
    sunDirection = tm.normalize(tm.vec3(-1., .8, -.7))
    accum = tm.vec3(1)
    dosky = True
    result = tm.vec3(0)
    for bounce in range(10):
        h = n = tm.vec3(0)
        k = 0.0
        clip = nearClip if bounce == 0 else 0.0
        if trace5(cam, dir, clip, h, n, k, fragCoord):
            cam = h + n * 0.01
            if material[fragCoord][0] == kMatGround:
                dir = getSampleBiased(n, 1)
                accum *= tm.mix(tm.vec3(.25, .3, .35), tm.vec3(.8), tm.step(0, floorPattern(h.xz)))
                if bounce == 0:
                    material[fragCoord][1] = 0

            elif material[fragCoord][0] == kMatWood:
                dir = getSampleBiased(n, 1)
                col = tm.vec3(211, 183, 155) / 255
                accum *= col * col *col

            elif material[fragCoord][0] == kMatPlasticRed:
                fresnel = tm.pow(1 - min(.99, tm.dot(-dir, n)), 5)
                fresnel = tm.mix(.04, 1., fresnel)
                if ti.random() < fresnel:
                    dir = tm.reflect(dir, n)
                else:
                    dir = getSampleBiased(n, 1)
                    accum *= tm.vec3(180, 2, 1) / 255.

            elif material[fragCoord][0] == kMatLead:
                fresnel = tm.pow(1 - min(.99, tm.dot(-dir, n)), 5)
                fresnel = tm.mix(.04, 1., fresnel)
                dir = getConeSample(tm.reflect(dir, n), 0.3)
                accum *= .05

        elif abs(k) > 0.1:
            result = sky(sunDirection, dir) * accum
            dosky = False
            break

        else:
            break

    if dosky:
        result = sky(sunDirection, dir) * accum
    return result


@ti.func
def bokeh():
    a = hash2()
    a.x = a.x * 3 - 1
    a -= tm.step(1, a.x + a.y)
    a.x += a.y * 0.5
    a.y *= tm.sqrt(0.75)
    return a


@ti.kernel
def renderBuffer():
    for i, j in iChannel0:
        fragCoord = tm.ivec2(i, j)
        fragColor = iChannel0[fragCoord]
        uv = (tm.vec2(i, j) + hash2() - 0.5) / iResolution - 0.5
        aspect = iResolution.x / iResolution.y
        uv.x *= aspect
        uv *= max(1, (16 / 9) / aspect)
        camPos = tm.vec3(140, 60, 60) * 1.5
        lookAt = tm.vec3(0, 4, 0)
        focusDistance = tm.distance(camPos, lookAt) * 0.99
        apertureRadius = tm.vec2(3)
        cam = tm.vec3(0)
        dir = tm.normalize(tm.vec3(uv, 6.5))
        bokehJitter = bokeh()
        cam.xy += bokehJitter * apertureRadius
        dir.xy -= bokehJitter * apertureRadius * dir.z / focusDistance

        lookDir = lookAt - camPos
        pitch = -tm.atan2(lookDir.y, tm.length(lookDir.xz))
        yaw = -tm.atan2(lookDir.x, lookDir.z)
        cam.yz = tm.rot2(pitch) @ cam.yz
        dir.yz = tm.rot2(pitch) @ dir.yz
        cam.xz = tm.rot2(yaw) @ cam.xz
        dir.xz = tm.rot2(yaw) @ dir.xz
        cam += camPos
        pixel = trace2(cam, dir, tm.length(camPos) * 0.7, fragCoord)
        if pixel.x >= 0.0:
            fragColor += tm.vec4(pixel, 1)
        else:
            fragColor += tm.vec4(0)
        iChannel0[i, j] = fragColor


@ti.kernel
def renderImage():
    for i, j in img:
        uv = tm.vec2(i, j) /iResolution
        tex = iChannel0[tm.ivec2(i, j)]
        color = tex.rgb / tex.a
        uv -= 0.5
        color *= 1 - tm.dot(uv, uv) * 0.1

        color *= 3.5
        color = tm.mix(color, 1 - tm.exp(color * -2), 0.5)
        color = tm.pow(color, tm.vec3(1, 1.02, 1.05))
        color = tm.pow(color, tm.vec3(.45))
        color += (tm.vec3(ti.random(), ti.random(), ti.random()) - 0.5) * 0.01

        uv *= iResolution.xy / iResolution.yx
        color *= tm.step(abs(uv.y), 0.5 / (16 / 9))
        color *= tm.step(abs(uv.x), 0.5 * (16 / 9))
        img[i, j] = tm.vec4(color, 1)



def main():
    gui = ti.ui.Window('Creative Block', res=(W, H))
    canvas = gui.get_canvas()
    init()
    #load_texture(iChannel1, "./noise_gray_64x64.png")
    while gui.running:
        gui.get_event(ti.ui.PRESS)
        if gui.is_pressed(ti.ui.ESCAPE):
            gui.running = False

        if gui.is_pressed('p'):
            canvas.set_image(img)
            gui.write_image('screenshot.png')

        renderBuffer()
        renderImage()
        canvas.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
