# https://www.shadertoy.com/view/ttdfz2
import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

W, H = 800, 640
iResolution = tm.vec2(W, H)
iTime = ti.field(ti.f32, shape=())
iMouse = ti.Vector.field(2, ti.f32, shape=())
img = ti.Vector.field(3, ti.f32, shape=(W, H))


def init():
    iTime[None] = 0.0
    iMouse[None] = (0, 0)
    return time.perf_counter()


@ti.func
def hue(x):
    return tm.cos(x * 6.3 + tm.vec3(0, 23, 21)) * 0.5 + 0.5


@ti.kernel
def step():
    for i, j in img:
        F = tm.vec3(i, j, 0)
        col = tm.vec3(0)
        for k in range(1, 100):
            p = F.z * tm.vec3((F.xy - 0.5 * iResolution) / H, 1)
            p.z -= 1.0
            p = tm.rotate3d(p, tm.normalize(tm.vec3(1, 3, 3)), iTime[None] * 0.2)
            s = 3.0
            for _ in range(5):
                s *= (e := 1.0 / tm.min(tm.dot(p, p), 1))
                p = abs(p) * e - 1.5

            F.z += (e := tm.length(p.xy) / s)
            col += (tm.mix(tm.vec3(1), hue(F.z), 0.6) * 0.8 / float(k)) * float(e < 0.001)

        img[i, j] = col


def main():
    t0 = init()
    gui = ti.GUI('Fractal Stamens', res=(W, H))
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
