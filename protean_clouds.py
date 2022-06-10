# https://www.shadertoy.com/view/3l23Rh
import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan)

W, H = 960, 640
MAXITER = 130
iResolution = tm.vec2(W, H)
iTime = ti.field(float, shape=())
iMouse = ti.Vector.field(2, float, shape=())

img = ti.Vector.field(3, float, shape=(W, H))

prm1 = ti.field(float, shape=())
bsMo = ti.Vector.field(2, float, shape=())

m3 = tm.mat3([[0.33338, 0.56034, -0.71817],
              [-0.87887, 0.32651, -0.15323],
              [0.15162, 0.69596, 0.61339]]) * 1.93


def init():
    iTime[None] = 0.0
    iMouse[None] = (0, 0)
    prm1[None] = 0
    bsMo[None] = (0, 0)
    return time.perf_counter()


@ti.func
def linstep(mn, mx, x):
    return tm.clamp((x - mn) / (mx - mn), 0, 1)


@ti.func
def disp(t):
    return tm.vec2(tm.sin(t * 0.22), tm.cos(t * 0.175)) * 2


@ti.func
def map(p):
    p2 = p
    p2.xy -= disp(p.z).xy
    m = tm.rot2(-tm.sin(p.z + iTime[None]) * (0.1 + prm1[None] * 0.05) - iTime[None] * 0.09)
    p.xy = m @ p.xy
    cl = tm.dot(p2.xy, p2.xy)
    d = 0.
    p *= .61
    z = 1.
    trk = 1.
    dspAmp = 0.1 + prm1[None] * 0.2
    for _ in range(5):
        p += tm.sin(p.zxy * 0.75 * trk + iTime[None] * trk * 0.8) * dspAmp
        d -= abs(tm.dot(tm.cos(p), tm.sin(p.yzx))*z)
        z *= 0.57
        trk *= 1.4
        p = m3.transpose() @ p

    d = abs(d + prm1[None] * 3.) + prm1[None] * 0.3 - 2.5 + bsMo[None].y
    return tm.vec2(d + cl * 0.2 + 0.25, cl)


@ti.func
def render(ro, rd, time):
    rez = tm.vec4(0)
    ldst = 8.
    lpos = tm.vec3(disp(time + ldst) * 0.5, time + ldst)
    t = 1.5
    fogT = 0.
    for _ in range(MAXITER):
        if rez.a > 0.99:
            break

        pos = ro + t*rd
        mpv = map(pos)
        den = tm.clamp(mpv.x - 0.3, 0, 1) * 1.12
        dn = tm.clamp((mpv.x + 2.), 0, 3)
        col = tm.vec4(0)
        if mpv.x > 0.6:
            col = tm.vec4(
                tm.sin(tm.vec3(5, 0.4, 0.2) + mpv.y * 0.1 + tm.sin(pos.z * 0.4) * 0.5 + 1.8) * 0.5 + 0.5, 0.08
            )
            col *= den * den * den
            col.rgb *= linstep(4, -2.5, mpv.x) * 2.3
            dif = tm.clamp((den - map(pos + 0.8).x) / 9, 0.001, 1)
            dif += tm.clamp((den - map(pos + 0.35).x) / 2.5, 0.001, 1)
            col.xyz *= den * (tm.vec3(0.005, 0.045, 0.075) + 1.5 * tm.vec3(0.033, 0.07, 0.03) * dif)

        fogC = tm.exp(t*0.2 - 2.2)
        col += tm.vec4(0.06, 0.11, 0.11, 0.1) * tm.clamp(fogC - fogT, 0, 1)
        fogT = fogC
        rez = rez + col * (1 - rez.a)
        t += tm.clamp(0.5 - dn * dn * 0.05, 0.09, 0.3)
    return tm.clamp(rez, 0, 1)


@ti.func
def getsat(c):
    mi = tm.min(tm.min(c.x, c.y), c.z)
    ma = tm.max(tm.max(c.x, c.y), c.z)
    return (ma - mi) / (ma + 1e-7)


@ti.func
def iLerp(a, b, x):
    ic = tm.mix(a, b, x) + tm.vec3(1e-6, 0, 0)
    sd = abs(getsat(ic) - tm.mix(getsat(a), getsat(b), x))
    dir = tm.normalize(tm.vec3(2 * ic.x - ic.y - ic.z,
                         2 * ic.y - ic.x - ic.z,
                         2 * ic.z - ic.y - ic.x))
    lgt = tm.dot(tm.vec3(1), ic)
    ff = tm.dot(dir, tm.normalize(ic))
    ic += 1.5 * dir * sd * ff * lgt
    return tm.clamp(ic, 0, 1)


@ti.kernel
def step():
    for i, j in img:
        fragCoord = tm.vec2(i, j)
        q = fragCoord / iResolution
        p = (fragCoord - 0.5 * iResolution) / iResolution.y
        bsMo[None] = (iMouse[None] - 0.5 * iResolution) / iResolution.y

        time = iTime[None] * 3
        ro = tm.vec3(0, 0, time)
        ro += tm.vec3(tm.sin(time) * 0.5, 0, 0)

        dspAmp = .85
        ro.xy += disp(ro.z) * dspAmp
        tgtDst = 3.5

        target = tm.normalize(ro - tm.vec3(disp(time + tgtDst) * dspAmp, time + tgtDst))
        ro.x -= bsMo[None].x * 2
        rightdir = tm.normalize(tm.cross(target, tm.vec3(0, 1, 0)))
        updir = tm.normalize(tm.cross(rightdir, target))
        rightdir = tm.normalize(tm.cross(updir, target))
        rd = tm.normalize((p.x * rightdir + p.y * updir) - target)
        rd.xy = tm.rot2(disp(time + 3.5).x*0.2 - bsMo[None].x) @ rd.xy
        prm1[None] = tm.smoothstep(-0.4, 0.4, tm.sin(iTime[None]*0.3))
        scn = render(ro, rd, time)

        col = scn.rgb
        col = iLerp(col.bgr, col.rgb, tm.clamp(1 - prm1[None], 0.05, 1))
        col = tm.pow(col, tm.vec3(.55, 0.65, 0.6)) * tm.vec3(1, 0.97, 0.9)
        col *= tm.pow(16.0 * q.x * q.y * (1 - q.x) * (1 - q.y), 0.12) * 0.7 + 0.3
        img[i, j] = col


def main():
    t0 = init()
    gui = ti.ui.Window('Protean Clouds', res=(W, H))
    canvas = gui.get_canvas()
    while gui.running:
        gui.get_event(ti.ui.PRESS)
        if gui.is_pressed(ti.ui.LMB):
            mouse_x, mouse_y = gui.get_cursor_pos()
            iMouse[None] = tm.vec2(mouse_x, mouse_y) * iResolution

        if gui.is_pressed(ti.ui.ESCAPE):
            gui.running = False

        if gui.is_pressed('p'):
            canvas.set_image(img)
            gui.show('screenshot.png')

        step()
        iTime[None] = time.perf_counter() - t0
        canvas.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
