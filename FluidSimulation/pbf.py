import taichi as ti
import random
import numpy as np

ti.init(arch=ti.gpu)

# global constant
RATIO = 10
MAX_PARTICLES = 30 * 30
dt = 1e-2
g = [0, -9.8 * dt]
damp = 1e-3
h = 3
PI = 3.1415926535
poly6_coef = 315 / 64 / PI / h**9
spiky_grad_coef = -45 / PI / h**6
rho0 = 0.24  # scaling coefficient?? / set as almost initial pho 
epsilon = 10
total_time = 0

# global variables
p = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
p_star = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
v = ti.Vector.field(2, dtype=ti.f32, shape=MAX_PARTICLES)
lbda = ti.field(ti.f32, shape=MAX_PARTICLES)
neighbours = ti.field(ti.i8, shape=(MAX_PARTICLES, MAX_PARTICLES))
p_grid = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_PARTICLES))
bounds = ti.field(ti.f32, shape=2)
direction = ti.field(ti.f32, shape=())

@ti.kernel
def update_neighbour():
    for i in range(MAX_PARTICLES):
        for j in range(i):
            if (p[j] - p[i]).norm() < h:
                neighbours[i, j] = 1
                neighbours[j, i] = 1
            else:
                neighbours[i, j] = 0
                neighbours[j, i] = 0
            neighbours[i, i] = 0


@ti.kernel
def compute_lambda():
    for i in range(MAX_PARTICLES):
        rhoi = 0.0
        denominator = 0.0
        sum_grad_pk_W = ti.Vector([0.0, 0.0])
        for k in range(MAX_PARTICLES):
            if neighbours[i, k] == 1:
                rhoi += W(p[i] - p[k])
                grad_pk_Ci = grad_W(p[i] - p[k])  # for neigbours
                sum_grad_pk_W += grad_pk_Ci
                denominator += grad_pk_Ci.norm_sqr()
        denominator += sum_grad_pk_W.norm_sqr()
        #if 100 < i < 200 and i % 10 == 0:
        #    print("rhoi", rhoi, "\tdnmt", denominator)
        Ci = rhoi - rho0
        lbda[i] = -Ci / (denominator + epsilon)

@ti.kernel
def collide_boundary():
    # collide with bounds
    for i in range(MAX_PARTICLES):
        if p_star[i].y < 0:
            p[i].y = p_star[i].y
            p_star[i].y = ti.random() * 0.1
            #v[i].y *= -0.5
        if p_star[i].x < bounds[0] / RATIO:
            p[i].x = p_star[i].x
            p_star[i].x = ti.random() * 0.1
            #v[i].x *= -0.5
        if p_star[i].x > bounds[1] / RATIO:
            p[i].x = p_star[i].x
            p_star[i].x = bounds[1] / RATIO - ti.random() * 0.1
            #v[i].x *= -0.5


@ti.kernel
def update_delta_p():
    for i in range(MAX_PARTICLES):
        threshold = 5
        delta_pi = ti.Vector([0.0, 0.0])
        for j in range(MAX_PARTICLES):
            if neighbours[i, j] == 1:
                delta_pi += (lbda[i] + lbda[j] + scorr(i, j)) * grad_W(p[i] - p[j])
        delta_pi /= rho0
        if (delta_pi[0] > threshold):  # insure numerical stability but i don't think it's a good way
            delta_pi[0] = threshold
        if (delta_pi[1] > threshold):
            delta_pi[1] = threshold
        p_star[i] += delta_pi


@ti.func
def grad_W(r):  # spiky
    return spiky_grad_coef * (h - r.norm())**2 * r.normalized()


@ti.func
def W(r):  # poly 6
    return poly6_coef * (h**2 - r.norm_sqr())**3


@ti.func
def scorr(i, j):
    dq = ti.Vector([0, 0.2 * h])
    return -0.02 * (W(p[i] - p[j]) / W(dq))**4

@ti.kernel
def external_dynamic():
    for i in range(MAX_PARTICLES):
        v[i] *= ti.exp(-damp * dt)
        v[i] += g
        p[i] = p[i] + v[i] * dt
        p_star[i] = p[i]

@ti.kernel
def update_velocity():
    for i in range(MAX_PARTICLES):
        v[i] += (p_star[i] - p[i]) / dt * 0.1
        p[i] = p_star[i]

@ti.kernel
def move_bound():
    speed = 1
    if bounds[1] <= 600:
        direction[None] = 1
    if bounds[1] >= 1000:
        direction[None] = -1
    bounds[1] += direction[None] * speed

def update(total_time):
    # external force & damping
    external_dynamic()

    # solver iteration
    for _ in range(3): # to auto-paralize 
        update_neighbour()
        collide_boundary()
        # constraints update
        compute_lambda() 
        update_delta_p()
    if total_time >= 20:
        move_bound()
    collide_boundary()
    update_velocity()

#main function start

#initialize
for i in range(MAX_PARTICLES):
    bounds[0] = 0
    bounds[1] = 1000
    direction[None] = -1
    #print(i//30 * 5 + 500, i % 30 * 5 + 500)
    p[i] = ti.Vector([
        i // 30 * 12 + 400 + (random.random() - 0.5), i % 30 * 12 + 400 +
        (random.random() - 0.5)
    ]) / RATIO
    #v[i] = ti.Vector([(random.random() - 0.5) * 10, (random.random()) - 0.5 * 10]) / RATIO
    #p[i, 1] = i % 30 * 5 + 500

# draw GUI
gui = ti.GUI('Position based fluid',
             res=(1000, 600),
             background_color=0xdddddd)
while True:
    for _ in range(5):
        update(total_time)
        total_time += dt
    gui.circles(p.to_numpy() * RATIO / np.array([1000, 600]), color=0x1E90FF, radius=5)
    gui.show()
