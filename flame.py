# https://www.shadertoy.com/view/MdX3zr
import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan)

W, H = 800, 640
iResolution = tm.vec2(W, H)
iTime = ti.field(float, shape=())
iMouse = ti.Vector.field(2, float, shape=())

img = ti.Vector.field(4, float, shape=(W, H))


def init():
    iTime[None] = 0.0
    iMouse[None] = (0, 0)
    return time.perf_counter()


@ti.func
def noise(p):
    i = tm.floor(p)
    a = tm.dot(i, tm.vec3(1., 57., 21.)) + tm.vec4(0., 57., 21., 78.)
    f = tm.cos((p - i) * tm.acos(-1)) * (-0.5) + 0.5
    a = tm.mix(tm.sin(tm.cos(a) * a), tm.sin(tm.cos(1 + a) * (1 + a)), f.x)
    a.xy = tm.mix(a.xz, a.yw, f.y)
    return tm.mix(a.x, a.y, f.z)


@ti.func
def sphere(p, spr):
    return tm.length(spr.xyz - p) - spr.w


@ti.func
def flame(p):
    d = sphere(p * tm.vec3(1, 0.5, 1.0), tm.vec4(.0,-1.,.0,1.))
    return d + (noise(p + tm.vec3(.0, iTime[None] * 2, 0)) + noise(p * 3) * 0.5) * 0.25 * (p.y)


@ti.func
def scene(p):
    return tm.min(100 - tm.length(p), abs(flame(p)))


@ti.func
def raymarch(org, dir):
    glow = 0.0
    eps = 0.02
    p = org
    glowed = False
    for i in range(64):
        d = scene(p) + eps
        p += d * dir
        if d > eps:
            if flame(p) < 0:
                glowed = True
            if glowed:
                glow = float(i) / 64
    return tm.vec4(p, glow)


@ti.kernel
def step():
    for i, j in img:
        v = 2 * tm.vec2(i, j) / iResolution - 1
        v.x *= W / H
        org = tm.vec3(0., -2., 4.)
        dir = tm.normalize(tm.vec3(v.x*1.6, -v.y, -1.5))
        p = raymarch(org, dir)
        glow = p.w
        col = tm.mix(tm.vec4(1, 0.5, 0.1, 1),
                     tm.vec4(0.1, 0.5, 1, 1),
                     p.y * 0.02 + 0.4)
        col = tm.mix(tm.vec4(0), col, tm.pow(glow * 2, 4))
        img[i, j] = col


def main():
    t0 = init()
    gui = ti.ui.Window('Flame', res=(W, H))
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
