import taichi as ti
ti.init(arch=ti.gpu)

res = 500
dt = 0.04
fade_coef = 0.9999

canvas = ti.field(float, shape=(10 * res, 10 * res))
p = ti.field(float, shape=(res, res))
p_b = ti.field(float, shape=(res, res))
r = ti.field(float, shape=(res, res))
m = ti.field(float, shape=(res, res))
u_a = ti.field(float, shape=(res + 1, res))
v_a = ti.field(float, shape=(res, res + 1))
u_b = ti.field(float, shape=(res + 1, res))
v_b = ti.field(float, shape=(res, res + 1))
d = ti.field(float, shape=(res, res))  # all between 0 to 1
r_norm_last = ti.field(float, shape=())
beta = ti.field(float, shape=())

class Pair: # swap to save memory cost
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def swap(self):
        self.a, self.b = self.b, self.a

@ti.kernel
def debug():
    #print(bilinear_interpolate_p(1.5, 0.6))
    #print(bilinear_interpolate_d(0.4, 0.6))
    #print(bilinear_interpolate_u(1, 0))
    x = 1.2
    y = 2.3
    x_, y_ = float(x), float(y)
    i, j = ti.floor(x_), ti.floor(y_ - 0.5)
    m, n = x_ - i, y_ - 0.5 - j
    print(_idx(i, j, d, res, res))
    #print(bilinear_interpolate_v(1, 2, v_pair.a))

def conjugate_gradient():
    compute_r0_m0(u_pair.b, v_pair.b)
    for _ in range(10):
        ret = descent(_)
        if ret == 1:
            break
        update_m()

@ti.kernel
def compute_r0_m0(u: ti.template(), v: ti.template()):  #compute r0, m0
    r_norm_last[None] = 0.0
    for i, j in ti.ndrange(res, res):
        div_V = u[i + 1, j] + v[i, j + 1] - u[i, j] - v[i, j]
        r[i, j] = div_V / dt - A_mul_x(i, j, p)
        m[i, j] = r[i, j]
        r_norm_last[None] += r[i, j] * r[i, j]
    r_norm_last[None] /= (res * res) 

@ti.kernel
def descent(t: ti.i32) -> ti.i32:

    ret = 0 # 0 for continue 1 for stop
    rT_r = 0.0 # compute nominator
    mT_A_m = 0.0  # compute denominator
    
    for i, j in ti.ndrange(res, res):
        rT_r += r[i, j] * r[i, j]
        mT_A_m += m[i, j] * A_mul_x(i, j, m)
    alpha = rT_r / mT_A_m

    #update p / r
    r_norm = 0.0
    for i, j in ti.ndrange(res, res):
        p[i, j] += alpha * m[i, j]
        r[i, j] -= alpha * A_mul_x(i, j, m)
        r_norm += r[i, j]* r[i, j]
    
    # determine ending
    r_norm = r_norm / res / res
    print("iteration round:", t, "\taverage normed residual:", r_norm)
    if r_norm <= 1e-2: # should determine
        ret = 1
    else:
        beta[None] = r_norm / r_norm_last[None]
        r_norm_last[None] = r_norm

    # update m is postponed since this can't be parralled here
    return ret

@ti.kernel
def update_m():
    for i, j in ti.ndrange(res, res):
        m[i, j] = r[i, j] + beta[None] * m[i, j]

@ti.func
def A_mul_x(i, j, x):  # multiply A with x at i, j entry
    # using safe indexing
    return (4 * _idx(i, j, x, res, res) - _idx(i - 1, j, x, res, res) - _idx(i, j - 1, x, res, res) - _idx(i + 1, j, x, res, res) - _idx(i, j + 1, x, res, res))

@ti.func
def bilinear_interpolate_d(x, y, vec):
    x_, y_ = float(x), float(y)
    i, j = ti.floor(x_-0.5), ti.floor(y_-0.5)
    m, n = x_-0.5-i, y_-0.5-j
    return _bilinear(m, n, _idx(i, j, vec, res, res), _idx(i, j + 1, vec, res, res), _idx(i + 1, j, vec, res, res), _idx(i + 1, j + 1, vec, res, res))

@ti.func
def bilinear_interpolate_v(x, y, vec):
    x_, y_ = float(x), float(y)
    i, j = ti.floor(x_- 0.5), ti.floor(y_)
    m, n = x_ -0.5 - i, y_ - j
    return _bilinear(m, n, _idx(i, j, vec, res, res + 1), _idx(i, j + 1, vec, res, res + 1), _idx(i + 1, j, vec, res, res + 1), _idx(i + 1, j + 1, vec, res, res + 1))

@ti.func
def bilinear_interpolate_u(x, y, vec):
    x_, y_ = float(x), float(y)
    i, j = ti.floor(x_), ti.floor(y_ - 0.5)
    m, n = x_ - i, y_ - j - 0.5 
    return _bilinear(m, n, _idx(i, j, vec, res + 1, res), _idx(i, j + 1, vec, res + 1, res), _idx(i + 1, j, vec, res + 1, res), _idx(i + 1, j + 1, vec, res + 1, res))

@ti.func
def _idx(i, j, vec, i_bound, j_bound):  # safe indexing with boundary check
    ret = 0.0
    if i < 0 or j < 0 or i >= i_bound or j >= j_bound:
        ret = 0.0
    else:
        ret = vec[int(i), int(j)]
    return ret

@ti.func
def _bilinear(m, n, left_bottom, left_top, right_bottom, right_top):
    return (1-n)*((1-m) * left_bottom + m * right_bottom) + n * ((1-m) * left_top + m * right_top)

@ti.kernel
def advect(u: ti.template(), u_new: ti.template(), v: ti.template(), v_new: ti.template()):  # advect u & v field
    # semi lagrangian
    V_u = ti.Vector([0.0, 0.0])
    V_v = ti.Vector([0.0, 0.0])
    for i, j in ti.ndrange((1, res-1), (0, res)):  # updating u while ignoring boundaries
        v_origin = bilinear_interpolate_v(i, j+0.5, v)
        V_u = BFECC(i, j + 0.5, u[i,j], v_origin, u, v)
        u_new[i, j] = V_u[0]
    
    for i, j in ti.ndrange((0, res), (1, res-1)):
        u_origin = bilinear_interpolate_u(i+0.5, j, u)
        V_v = BFECC(i+0.5, j, u_origin, v[i,j], u, v)
        v_new[i, j] = V_v[1]


        
@ti.kernel
def advect_dyes(u: ti.template(), v: ti.template(), d: ti.template()):
    for i, j in ti.ndrange(res, res):
        u_origin = bilinear_interpolate_u(i+0.5, j+0.5, u)
        v_origin = bilinear_interpolate_v(i+0.5, j+0.5, v)
        x_mid = i+0.5 - u_origin * dt * 0.5 #mid point
        y_mid = j+0.5 - v_origin * dt * 0.5
        u_mid = bilinear_interpolate_u(x_mid, y_mid, u)
        v_mid = bilinear_interpolate_v(x_mid, y_mid, v)
        x_ = i+0.5 - u_mid * dt
        y_ = j + 0.5 - v_mid * dt
        V_star = ti.Vector([bilinear_interpolate_u(x_, y_, u), bilinear_interpolate_v(x_, y_, v)]) 
        d_star = bilinear_interpolate_d(x_, y_, d)
        d_sstar = bilinear_interpolate_d(x_ + V_star.x * dt, y_ + V_star.y * dt, d)
        d[i, j] = fade_coef * (d_star + 0.5 * (d_sstar - d[i, j]))
        # d_pos = ti.Vector([float(i+0.5), float(j+0.5)]) - V_origin * dt

@ti.func
def BFECC(x, y, u_origin, v_origin, u, v):
    # first semi-lagrangian
    x_mid = x - u_origin * dt * 0.5 #mid point
    y_mid = y - v_origin * dt * 0.5
    u_mid = bilinear_interpolate_u(x_mid, y_mid, u)
    v_mid = bilinear_interpolate_v(x_mid, y_mid, v)
    x_ = x - u_mid * dt
    y_ = y - v_mid * dt
    V_star = ti.Vector([bilinear_interpolate_u(x_, y_, u), bilinear_interpolate_v(x_, y_, v)]) 
    # V_star
    x_mid_s = x + V_star.x * dt * 0.5 #mid point
    y_mid_s = y + V_star.y * dt * 0.5
    u_mid_s = bilinear_interpolate_u(x_mid_s, y_mid_s, u)
    v_mid_s = bilinear_interpolate_v(x_mid_s, y_mid_s, v)
    x_s = x + u_mid_s * dt
    y_s = y + v_mid_s * dt
    V_sstar = ti.Vector([bilinear_interpolate_u(x_s, y_s, u), bilinear_interpolate_v(x_s, y_s, v)])
    
    V_corr = 0.5 * (V_sstar + ti.Vector([u_origin, v_origin]))
    return semi_lagrangian(x, y, V_corr.x, V_corr.y, u, v)

@ti.func     
def semi_lagrangian(x, y, u_origin, v_origin, u, v): #maccormack later
    x_mid = x - u_origin * dt * 0.5 #mid point
    y_mid = y - v_origin * dt * 0.5
    u_mid = bilinear_interpolate_u(x_mid, y_mid, u)
    v_mid = bilinear_interpolate_v(x_mid, y_mid, v)
    x_ = x - u_mid * dt
    y_ = y - v_mid * dt
    return ti.Vector([bilinear_interpolate_u(x_, y_, u), bilinear_interpolate_v(x_, y_, v)]) 

@ti.kernel
def gravity(v: ti.template()): #not useful ?
    for i, j in ti.ndrange((0, res), (1, res)):
        v[i, j] -= 0.01 * dt

@ti.kernel
def apply_pressure(u: ti.template(), v: ti.template(), p: ti.template()):
    for i, j in ti.ndrange((0, res), (1, res)):
        v[i, j] -= dt * (p[i, j] - p[i, j - 1])
    for i, j in ti.ndrange((1, res), (0, res)): 
        u[i, j] -= dt * (p[i, j] - p[i - 1, j])

@ti.kernel
def apply_boundary(u: ti.template(), v: ti.template(), d: ti.template()):
    for j in range(int(res * 0.5), int(res * 0.55)):
        for i in range(int(10)):
            u[i, j] = 10
            u[i, j] = 10
            d[i, j] = 1
    v[res-1, 0] = 0
    #for i in range(res):
    #    v[res-1, i] +=  u[res-1, i] * -0.5
    #    v[res-1, i+1] = u[res-1, i] * 0.5


@ti.kernel
def Jacobi(it_n: ti.i32, p: ti.template(), p_new: ti.template(), u: ti.template(), v: ti.template()):
    norm = 0.0
    for i, j in ti.ndrange(res, res):
        div_V = u[i + 1, j] + v[i, j + 1] - u[i, j] - v[i, j]
        new_val = (-div_V / dt + _idx(i, j + 1, p, res, res) + _idx(i + 1, j, p, res, res) + _idx(i - 1, j, p, res, res) + _idx(i, j - 1, p, res, res)) * 0.25
        norm += (new_val - p[i, j]) * (new_val - p[i, j])
        p_new[i, j] = new_val
    #print("iteraiton:", it_n, "\tnorm", norm / res / res)
    
@ti.kernel
def paint():
    for i, j in ti.ndrange(res, res):
        for i_, j_ in ti.ndrange(10, 10):
            canvas[10 *i + i_, 10 *j+j_] = d[i, j]

def update():
    advect(u_pair.a, u_pair.b, v_pair.a, v_pair.b)
    advect_dyes(u_pair.a, v_pair.a, d)
    gravity(v_pair.b)
    apply_boundary(u_pair.b, v_pair.b, d)
    #conjugate_gradient()
    for i in range(20):
        Jacobi(i, p_pair.a, p_pair.b, u_pair.b, v_pair.b)
        p_pair.swap()
    apply_pressure(u_pair.b, v_pair.b, p_pair.a)  # pressure is always swaped to p_pair.a
    u_pair.swap()
    v_pair.swap()

#main funciton starts here
#initizliaztion 
#for i in range(200, 300):
#    for j in range(300, 350):
#        d[i, j] = 1
u_pair = Pair(u_a, u_b)
v_pair = Pair(v_a, v_b)
p_pair = Pair(p  , p_b)

#for i in range(100): # for debug only 
#    for j in range(100):
#        p[i, j] = d[i, j] = (i + j)
#        u_pair.a[i, j] = v_pair.a[i, j] = -i

# visulization and control logic
gui = ti.GUI('Euler fluid', res=(res, res), background_color=0xdddddd)
#debug()
while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            update()
            #print(d)
    for _ in range(5):
        update()
    #paint()
    #gui.set_image(canvas)
    gui.set_image(d.to_numpy())
    #gui.set_image(u_a.to_numpy())
    gui.show()