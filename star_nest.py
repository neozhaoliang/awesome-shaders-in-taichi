# https://www.shadertoy.com/view/XlfGRj
import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

W, H = 800, 640
iResolution = tm.vec2(W, H)
iTime = ti.field(ti.f32, shape=())
iMouse = ti.Vector.field(2, ti.f32, shape=())

img = ti.Vector.field(3, ti.f32, shape=(W, H))

iterations = 17
formuparam = 0.53

volsteps = 20
stepsize = 0.1

zoom = 0.800
tile = 0.850
speed = 0.010

brightness = 0.0015
darkmatter = 0.300
distfading = 0.730
saturation = 0.850


def init():
    iTime[None] = 0.0
    iMouse[None] = (0, 0)
    return time.perf_counter()


@ti.kernel
def step():
    for i, j in img:
        uv = tm.vec2(i, j) / iResolution - 0.5
        uv.y *= H / W
        dir = tm.vec3(uv * zoom, 1)
        time = iTime[None] * speed + 0.25

        a1 = 0.5 + iMouse[None].x * 2
        a2 = 0.8 + iMouse[None].y * 2
        R1 = tm.rot2(-a1)
        R2 = tm.rot2(-a2)
        dir.xz = R1 @ dir.xz
        dir.xy = R2 @ dir.xy
        fr = tm.vec3(1, 0.5, 0.5)
        fr += tm.vec3(2*time, time, -2)
        fr.xz = R1 @ fr.xz
        fr.xy = R2 @ fr.xy

        s, fade = 0.1, 1.0
        v = tm.vec3(0)
        for r in range(volsteps):
            p = fr + s * dir * 0.5
            p = abs(tm.vec3(tile) - tm.mod(p, tm.vec3(tile * 2)))
            pa = a = 0.0
            for _ in range(iterations):
                p = abs(p) / tm.dot(p, p) - formuparam
                a += abs(tm.length(p) - pa)
                pa = tm.length(p)

            dm = max(0, darkmatter - a * a * 0.001)
            a *= a*a
            if r > 6:
                fade *= 1 - dm

            v += fade
            v += tm.vec3(s, s*s, s*s*s*s) * a * fade * brightness
            fade *= distfading
            s += stepsize

        color = tm.mix(tm.vec3(tm.length(v)), v, saturation)
        img[i, j] = color * 0.01


def main():
    t0 = init()
    gui = ti.GUI('Star Nest', res=(W, H), fast_gui=True)
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
