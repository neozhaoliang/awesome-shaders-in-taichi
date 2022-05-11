# https://www.shadertoy.com/view/XlfGRj
import taichi as ti
from taichi.math import *

ti.init(arch=ti.gpu)

W, H = 800, 640
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


@ti.kernel
def step(t: float, mx: float, my: float):
    for i, j in img:
        uv = vec2(i, j) / H - 0.5
        dir = vec3(uv * zoom, 1)
        time = t * speed + 0.25

        a1 = 0.5 + mx * 2
        a2 = 0.8 + my * 2
        R1 = rot2(-a1)
        R2 = rot2(-a2)
        dir.xz = R1 @ dir.xz
        dir.xy = R2 @ dir.xy
        fr = vec3(1, 0.5, 0.5)
        fr += vec3(2*time, time, -2)
        fr.xz = R1 @ fr.xz
        fr.xy = R2 @ fr.xy

        s, fade = 0.1, 1.0
        v = vec3(0)
        for r in range(volsteps):
            p = fr + s * dir * 0.5
            p = abs(vec3(tile) - mod(p, vec3(tile * 2)))
            pa = a = 0.0
            for _ in range(iterations):
                p = abs(p) / dot(p, p) - formuparam
                a += abs(length(p) - pa)
                pa = length(p)

            dm = max(0, darkmatter - a * a * 0.001)
            a *= a*a
            if r > 6:
                fade *= 1 - dm

            v += fade
            v += vec3(s, s*s, s*s*s*s) * a * fade * brightness
            fade *= distfading
            s += stepsize

        color = mix(vec3(length(v)), v, saturation)
        img[i, j] = color * 0.01


def main():
    gui = ti.GUI('universe', res=(W, H))
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
        t += 0.01
        gui.set_image(img)
        gui.show()


if __name__ == '__main__':
    main()
