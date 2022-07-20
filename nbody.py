import time
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan)

res = (800, 640)  # window resolution
img = ti.Vector.field(3, float, shape=res)  # pixel colors
iTime = ti.field(float, shape=())  # time elapsed up to current step
aspect = res[0] / res[1]
start_time = None
substeps = 10
wrap_boundary = True  # use periodic boundary
xmax = 1.0  # max distance in x-axis

N = 200  # number of particles
G = 1  # gravitational constant 6.67408e-11, using 1 for simplicity 
pos = ti.Vector.field(2, float, N)  # particle positions, N x 2d vectors (x, y)
vel = ti.Vector.field(2, float, N)  # particle velocities N x 2d vectors (vx, vy)
force = ti.Vector.field(2, float, N)  # particle forces N x 2d vectors (Fx, Fy)
galaxy_size = 0.5  # galaxy size
mass = 1  # particle mass
init_vel = 120  # initial velocity
lum_density = 40000.0
star_size = 2.0

@ti.func
def hash11(x):
    return tm.fract(tm.sin(x * 33.) * 43758.5453)

@ti.func
def hsv2rgb(hsv):
    hsv.yz = tm.clamp(hsv.yz, 0, 1)
    v = tm.vec3(0, 2, 1) / 3
    return hsv.z * (0.63 * hsv.y * (tm.cos(2 * tm.pi * (hsv.x + v)) - 1) + 1)

@ti.func
def get_particle_color(ind, lum):
    sat = tm.mix(0.5, 0.9, hash11(ind)) * 0.45 / lum
    hue = tm.mix(-0.2, 0.2, hash11(ind + 13)) + 0.75 * iTime[None]
    return hsv2rgb(tm.vec3(hue, sat, lum))
    
@ti.func
def draw_particles(uv):
    starhv = tm.vec2(9, 0.32)
    stardiag = tm.vec2(13, 0.61)
    col = tm.vec3(0)
    for i in range(N):
        p = uv - pos[i]
        q = 0.707 * tm.vec2(tm.dot(p, tm.vec2(1)), tm.dot(p, tm.vec2(1, -1)))
        dists = tm.vec4(
            tm.length(p * starhv),
            tm.length(p * starhv.yx),
            tm.length(q * stardiag),
            tm.length(q * stardiag.yx)
        ) * star_size + tm.vec4(0.015, tm.vec3(0.01))
        lum0 = tm.mix(0.1, 3.2, hash11(i + 20))
        lum1 = tm.dot(tm.vec4(0.65, 0.65, 0.2, 0.2), 1 / dists) + 1 / (p.norm() * star_size + 0.015)
        lum = lum0 * pow(lum1, 2.2) / lum_density
        col += get_particle_color(i, lum)
    return col

@ti.kernel
def init_particles():
    for i in range(N):
        theta = ti.random() * 2 * tm.pi
        r = tm.sqrt(ti.random() * 0.9 + 0.1) * xmax * galaxy_size
        offset = r * tm.vdir(theta)
        pos[i] = offset
        vel[i] = tm.vec2(-offset.y, offset.x) * init_vel

@ti.kernel
def compute_force():
    for i in range(N):
        force[i] = [0, 0]  # clear forces
    # compute gravitational force
    for i in range(N):
        for j in range(N):
            if i != j:
                diff = pos[i] - pos[j]
                r = diff.norm(1e-5)
                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * mass * mass * diff / r**3
                # assign to each particle
                force[i] += f

@ti.kernel
def update_pos_vel():
    dt = 1e-4 / substeps
    for i in range(N):
        vel[i] += dt * force[i] / mass
        pos[i] += dt * vel[i]
        if ti.static(wrap_boundary):
            if abs(pos[i].x) > xmax:
                pos[i].x = -pos[i].x
            if abs(pos[i].y) > xmax / aspect:
                pos[i].y = -pos[i].y

def init():
    global start_time
    start_time = time.perf_counter()
    iTime[None] = 0
    init_particles()

def update_time():
    iTime[None] = time.perf_counter() - start_time

@ti.kernel
def render():
    for i, j in img:
        uv = 2 * tm.vec2(i, j) / res - 1
        uv *= xmax
        uv.y /= aspect
        img[i, j] = img[i, j] * 0.75 + draw_particles(uv) * 0.9

gui = ti.ui.Window("N-body problem", res=res)
canvas = gui.get_canvas()
init()
while gui.running:
    if gui.is_pressed("r"):
        init()

    if gui.is_pressed("p"):
        canvas.set_image(img)
        gui.write_image("screenshot.png")
        gui.show()

    for _ in range(substeps):
        update_time()
        compute_force()
        update_pos_vel()

    render()
    canvas.set_image(img)
    gui.show()
