from time import perf_counter
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)
res = (800, 640)
img = ti.Vector.field(3, ti.f32, shape=res)

@ti.kernel
def step(t: ti.f32):
    for i, j in img:
        FC = tm.vec3(i, j, 0)
        img[i, j] = tm.vec3(0)
        for k in range(1, 100):
            p = FC.z * tm.vec3((FC.xy - 0.5 * tm.vec2(res)) / res[1], 1)
            p.z -= 1.0
            p = tm.rotate3d(p, tm.normalize(tm.vec3(1, 3, 3)), t * 0.2)
            s = 3.0
            for _ in range(5):
                s *= (e := 1.0 / tm.min(tm.dot(p, p), 1))
                p = abs(p) * e - 1.5
            FC.z += (e := tm.length(p.xy) / s)
            img[i, j] += (tm.cos(FC.z * 6.3 + tm.vec3(0, 23, 21)) * 0.24 + 0.56) * float(e < 0.001) / k

t0 = perf_counter()
gui = ti.GUI('Fractal Stamens', res=res)
while gui.running:
    step(perf_counter() - t0)
    gui.set_image(img)
    gui.show()
