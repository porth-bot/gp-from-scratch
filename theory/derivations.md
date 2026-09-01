# Derivations

Hand-derived math behind the code, section-numbered to match the docstring
cross-references (`gp/gp.py` cites Sec. 1 and Sec. 2; `gp/ntk.py` cites
Sec. 6 and Sec. 7). Notation follows the source: `K` is a covariance
(Gram) matrix, `K_y = K + sigma^2 I` is the observation covariance,
`alpha = K_y^{-1} y`, and every positive hyperparameter is optimized through
its log, `theta = log psi`.

Everything here is checked numerically somewhere in `tests/`: kernel
gradients against central finite differences (`tests/test_kernels.py`), the
LML gradient against FD (`tests/test_gp.py`), and the arc-cosine / NTK
identities against the finite-width Monte-Carlo network (`tests/test_ntk.py`).

Contents:

1. Gaussian conditioning via the Schur complement
2. The log marginal likelihood and its gradient (trace identity)
3. Cholesky numerics
4. Kernel gradients in log-parameter space (the gradient table)
5. The Matern smoothness ladder and the periodic kernel (MacKay's warping)
6. Arc-cosine kernels (polar integral for kappa1, orthant probability for kappa0)
7. NNGP, NTK, and linearized gradient descent (the geometric series)
8. Bochner's theorem and random Fourier features (the `n^3 -> n D^2` trade)
9. The variational sparse GP (Titsias): inducing points and the free-energy bound
10. The Laplace approximation: a likelihood that is not Gaussian (and why it is
    not a bound in either direction)
11. Exercises

---

## 1. Gaussian conditioning via the Schur complement

Everything the regressor does at prediction time is one application of the
Gaussian conditioning formula. Let a joint Gaussian be partitioned into
blocks `a` (the quantity we want) and `b` (the quantity we observe):

```
[a]      ( [mu_a]   [ A    C  ] )
[b] ~ N  ( [mu_b] , [ C^T  B  ] ).
```

**Claim.** `a | b` is Gaussian with

```
mean(a | b) = mu_a + C B^{-1} (b - mu_b)
cov (a | b) = A - C B^{-1} C^T   =: S     (the Schur complement of B).
```

**Derivation (block LDL / completing the square).** The covariance factors as

```
[ A    C  ]   [ I   C B^{-1} ] [ S   0 ] [ I        0 ]
[ C^T  B  ] = [ 0   I        ] [ 0   B ] [ B^{-1}C^T I ],
```

with `S = A - C B^{-1} C^T`. This is just block Gaussian elimination:
the middle factor is block-diagonal, and one checks the product by
multiplying it out. Inverting a product of triangular/diagonal factors is
easy (invert each, reverse the order), giving the precision matrix

```
[ A    C  ]^{-1}   [ I         0 ] [ S^{-1}  0    ] [ I  -C B^{-1} ]
[ C^T  B  ]      = [ -B^{-1}C^T I ] [ 0       B^{-1}] [ 0   I        ].
```

Write the joint log-density's exponent, `-1/2 (z - mu)^T Prec (z - mu)` with
`z = (a, b)`. Substituting the factored precision above, the cross terms
telescope and the exponent splits into exactly two pieces:

```
-1/2 (a - mu_a - C B^{-1}(b - mu_b))^T S^{-1} (a - mu_a - C B^{-1}(b - mu_b))
-1/2 (b - mu_b)^T B^{-1} (b - mu_b).
```

The second piece is the marginal `p(b)`; the first, read as a function of
`a`, is a Gaussian in `a` with mean `mu_a + C B^{-1}(b - mu_b)` and
covariance `S`. That is the claim.

**Map to the GP.** With a zero-mean prior `f ~ GP(0, k)` and
`y = f(X) + eps`, `eps ~ N(0, sigma^2 I)`, the test outputs `f* = f(X*)` and
the observations `y` are jointly Gaussian. Reading off the blocks
(`a = f*`, `b = y`):

```
A = K**  = k(X*, X*)
C = K*   = k(X*, X)
B = K_y  = k(X, X) + sigma^2 I.
```

So

```
mean(f* | y) = K* K_y^{-1} y                (mu_a = mu_b = 0)
cov (f* | y) = K** - K* K_y^{-1} K*^T.
```

This is exactly `GPRegressor.predict`: `mean = Ks.T @ alpha` with
`alpha = K_y^{-1} y`, and `var = diag(K**) - sum(v^2)` with
`v = L^{-1} Ks` so that `sum(v^2, axis=0) = diag(K*^T K_y^{-1} K*)`. Passing
`include_noise=True` adds `sigma^2` to predict a new **observation** `y*`
rather than the latent `f*` (it puts the likelihood noise back on top of the
conditioned latent variance).

---

## 2. The log marginal likelihood and its gradient (trace identity)

Marginalizing the latent `f` (a Gaussian integral) gives
`y ~ N(0, K_y)`, hence the **evidence**

```
log p(y | theta) = -1/2 y^T K_y^{-1} y  -  1/2 log|K_y|  -  n/2 log(2 pi).
```

The three terms are, in order, **data fit**, **complexity penalty**, and a
normalizing constant. The complexity term `-1/2 log|K_y|` is the automatic
Occam's razor: a kernel flexible enough to fit any dataset spreads its prior
mass thinly, inflates `|K_y|`, and pays for it here. ML-II (empirical Bayes)
maximizes this over `theta`.

**Gradient.** Two matrix-calculus identities do all the work. For a matrix
`M(theta)` that is invertible and depends smoothly on a scalar `theta`:

```
d/dtheta (M^{-1})   = -M^{-1} (dM/dtheta) M^{-1}       (differentiate M M^{-1} = I)
d/dtheta log|M|     =  tr(M^{-1} dM/dtheta)            (Jacobi's formula).
```

Apply to each term of the LML with `M = K_y` and write `alpha = K_y^{-1} y`:

```
d/dtheta_j [ -1/2 y^T K_y^{-1} y ]
    = -1/2 y^T ( -K_y^{-1} (dK_y/dtheta_j) K_y^{-1} ) y
    = +1/2 alpha^T (dK_y/dtheta_j) alpha

d/dtheta_j [ -1/2 log|K_y| ]
    = -1/2 tr( K_y^{-1} dK_y/dtheta_j ).
```

Both are traces of `(something) x dK_y/dtheta_j`: use `x^T A x = tr(A x x^T)`
on the first. Combining,

```
d/dtheta_j log p(y | theta) = 1/2 tr[ ( alpha alpha^T - K_y^{-1} ) dK_y/dtheta_j ].
```

This is `lml_and_grad`: it forms `A = alpha alpha^T - K_y^{-1}` once, then
each parameter gradient is `1/2 * sum(A * dK_j)` (elementwise, which equals
`1/2 tr(A dK_j)` because both `A` and `dK_j` are symmetric).

**The noise gradient.** Noise enters as `theta_noise = log sigma^2` with
`dK_y / d(log sigma^2) = sigma^2 I`. Plugging `dK_y = sigma^2 I` into the
trace formula:

```
d/d(log sigma^2) log p = 1/2 sigma^2 tr( alpha alpha^T - K_y^{-1} )
                       = 1/2 sigma^2 ( alpha^T alpha - tr K_y^{-1} ),
```

which is the last line of `lml_and_grad` (`0.5 * noise_var * trace(A)`).

**Optimization.** The evidence is maximized by Adam ascent in
log-hyperparameter space (`gp/optimize.py`, Kingma & Ba 2015). Log-space
makes the steps scale-free (Sec. 4) and keeps every hyperparameter positive
without projection. ML-II surfaces are multimodal in general; random
restarts are the standard mitigation and are used where it matters.

---

## 3. Cholesky numerics

`K_y` is symmetric positive definite (SPD): `K` is PSD as a Gram matrix of a
valid kernel, and `+ sigma^2 I` (with `sigma^2 > 0`) makes it strictly PD.
For SPD matrices the Cholesky factorization `K_y = L L^T`, `L` lower
triangular with positive diagonal, is the right tool:

- **It exists and is unique** for SPD `K_y`, and computing it *is* the
  numerical test of positive-definiteness (it fails exactly when `K_y` is not
  PD).
- **Cost `n^3 / 3`** — half of an LU factorization, because symmetry is
  exploited.
- **Backward stable** without pivoting (the SPD structure guarantees the
  factor cannot blow up).
- **The log-determinant is free:** `|K_y| = |L|^2 = (prod_i L_ii)^2`, so

  ```
  log|K_y| = 2 sum_i log L_ii,
  ```

  used directly in the LML — no separate, less-stable determinant call.

Linear solves never form `K_y^{-1}`. Given `L`, solve `L L^T x = b` as two
triangular back-substitutions (`_chol_solve`):

```
alpha = L^{-T} (L^{-1} y)          two O(n^2) solves
v     = L^{-1} K*,   var = diag(K**) - colsum(v^2).
```

**Jitter.** Finitely sampled smooth kernels are numerically only
semidefinite: nearby inputs produce nearly identical rows, so the smallest
eigenvalue of `K` can dip below zero in floating point and break the
Cholesky. Adding `1e-10 * mean(diag(K))` to the diagonal restores strict
positive-definiteness at a level far below any statistical noise. (The `sqdist`
helper does the analogous clip in distance space, `max(., 0)`, to kill the
tiny negatives that `||a||^2 + ||b||^2 - 2 a.b` produces before a `sqrt`.)

---

## 4. Kernel gradients in log-parameter space (the gradient table)

Every kernel is parameterized by `theta = log psi` for its positive
hyperparameters `psi`. The one identity used everywhere is the log-space
chain rule:

```
d/d(log psi) = psi * d/dpsi.
```

Two consequences: (i) an unconstrained step in `theta` can never make
`psi = exp(theta)` non-positive, and (ii) the effective step is
*multiplicative* — a `0.1` step in `theta` means "change `psi` by ~10%"
whatever its magnitude, so a lengthscale of `0.01` and one of `100` get
comparable learning rates. This is why the whole optimization runs in
`theta`.

Below, `r = ||x - x'||`, `d2 = r^2`, `s2` is the signal variance, `l` the
lengthscale. Each row is verified against central FD in
`tests/test_kernels.py`.

**RBF**, `k = s2 * exp(-r^2 / (2 l^2))`:

```
dk/d(log s2) = s2 * (k / s2)              = k
dk/d(log l)  = l  * (k * r^2 / l^3)       = k * r^2 / l^2.
```

**Matern**, with `a = sqrt(2 nu) r / l`, so `da/d(log l) = l * (-a/l) = -a`.
The `s2` gradient is always `k` (as for RBF); for `log l`, use
`dk/d(log l) = (dk/da)(da/d(log l)) = -a * dk/da`:

```
nu = 0.5:  k = s2 e^{-a},                 dk/da = -s2 e^{-a}
           => dk/d(log l) = s2 * a e^{-a}

nu = 1.5:  k = s2 (1 + a) e^{-a},         dk/da = -s2 a e^{-a}
           => dk/d(log l) = s2 * a^2 e^{-a}

nu = 2.5:  k = s2 (1 + a + a^2/3) e^{-a}, dk/da = -s2 (a/3)(1 + a) e^{-a}
           => dk/d(log l) = s2 * (a^2 (1 + a) / 3) e^{-a}.
```

(For the `nu = 2.5` case: differentiating the polynomial gives `1 + 2a/3`,
and subtracting the polynomial itself from the `e^{-a}` derivative leaves
`-(a/3)(1 + a)`.)

**Periodic** (derivation of the kernel itself in Sec. 5),
`k = s2 * exp(-2 sin^2(pi r / p) / l^2)`. Let `phi = pi r / p`:

```
dk/d(log s2) = k

dk/d(log l)  = l * k * 4 sin^2(phi) / l^3   = k * 4 sin^2(pi r / p) / l^2

dk/d(log p): d/dp [-2 sin^2(phi)/l^2] = (-2/l^2) * 2 sin(phi)cos(phi) * dphi/dp,
             dphi/dp = -pi r / p^2,   2 sin(phi)cos(phi) = sin(2 phi),
             => dk/dp = k * (2 pi r / (l^2 p^2)) sin(2 pi r / p),
             => dk/d(log p) = p * dk/dp = k * (2 pi r / (p l^2)) sin(2 pi r / p).
```

**Composition.** `Sum` gradients concatenate (linearity: `d(k1+k2) = dk1 +
dk2`). `Product` uses the product rule elementwise:
`d(k1 k2) = dk1 * k2 + k1 * dk2`, so each `k1`-gradient is multiplied by the
matrix `k2` and vice versa. A **fixed** (frozen) parameter is simply dropped
from `theta`, `grads`, and `n_params` via a boolean mask, so the optimizer
never sees it — used to pin the CO2 seasonal period to exactly one year
(freezing it also removes the enormous periodic log-period gradient near a
phase mismatch, which otherwise destabilizes Adam).

---

## 5. The Matern smoothness ladder and the periodic kernel

### 5.1 Matern smoothness ladder

The general Matern kernel is

```
k_nu(r) = s2 * 2^{1-nu} / Gamma(nu) * (sqrt(2 nu) r / l)^nu * K_nu(sqrt(2 nu) r / l),
```

where `K_nu` is the modified Bessel function of the second kind. The only
property that matters for modeling is **sample-path smoothness**: a
stationary GP with Matern-`nu` covariance is `ceil(nu) - 1` times
mean-square differentiable. This gives a tunable ladder from rough to smooth:

```
nu = 1/2 : k = s2 e^{-a}                     continuous, nowhere differentiable
                                             (Ornstein-Uhlenbeck / exponential)
nu = 3/2 : k = s2 (1 + a) e^{-a}             once  MS-differentiable
nu = 5/2 : k = s2 (1 + a + a^2/3) e^{-a}     twice MS-differentiable
nu -> inf: k -> s2 exp(-r^2 / 2 l^2)         the RBF (infinitely smooth).
```

with `a = sqrt(2 nu) r / l`. The half-integer values are the ones with an
elementary closed form (the Bessel function collapses to
`exp x polynomial`), which is why the code restricts to
`nu in {1/2, 3/2, 5/2}`. The practical point: RBF's `C^infinity` paths are
often *too* smooth for physical data, and the Matern ladder lets you dial in
exactly how many derivatives the prior should believe in. This ladder is
visible directly in prior sample paths — `experiments/prior_samples.py` draws
from each kernel at a fixed lengthscale (figure in the README): the `nu=1/2`
draws are visibly jagged and the RBF draws are glassy-smooth.

The `nu -> inf` limit is worth seeing: with `a = sqrt(2 nu) r / l`, the
leading behavior of the normalized Bessel form is
`exp(-a^2 / (4 nu)) = exp(-r^2 / (2 l^2))`, independent of `nu` — the RBF.

### 5.2 Periodic kernel via MacKay's warping

To build an *exactly periodic* prior, MacKay's trick is to warp the input
onto a circle and apply an RBF there. Map `x` to

```
u(x) = ( cos(2 pi x / p),  sin(2 pi x / p) ) in R^2,
```

so that `x` and `x + p` land on the same point — periodicity is now built
into the geometry. The squared chord distance on the circle is

```
||u(x) - u(x')||^2 = 2 - 2 cos(2 pi (x - x') / p) = 4 sin^2(pi r / p),
```

using `1 - cos(2 t) = 2 sin^2(t)`. Feeding this into an RBF of lengthscale
`l`, `s2 exp(-||Delta u||^2 / (2 l^2))`, gives MacKay's periodic kernel

```
k(r) = s2 * exp( -2 sin^2(pi r / p) / l^2 ),
```

exactly periodic with period `p`, signal variance `s2`, and `l` controlling
the within-period wiggliness. Its gradients are in Sec. 4. In the CO2 model
this kernel supplies the annual cycle, multiplied by an RBF envelope so the
seasonal amplitude can drift slowly across decades.

---

## 6. Arc-cosine kernels (polar integral for kappa1, orthant probability for kappa0)

The infinite-width limits of the one-hidden-layer ReLU network reduce to two
Gaussian expectations over a random hidden weight `w ~ N(0, I)`. Both depend
on the inputs only through the two norms and the angle between them. Write
augmented inputs `u = x~`, `v = x~'` (the code appends a 1 for the bias), and
let `theta` be the angle between `u` and `v` (here `theta` is a geometric
angle, not a hyperparameter). These are the arc-cosine kernels of order 1
and 0 (Cho & Saul 2009).

**Reduction to a plane.** Both expectations involve `w` only through the
projections `w.u` and `w.v`. Since `w ~ N(0, I)` is isotropic, we can rotate
coordinates so all of `w`'s mass outside the 2D plane spanned by `u, v`
integrates out to 1, leaving a 2D standard Gaussian on that plane. Write its
polar coordinates `(rho, phi)`: density `(1 / 2 pi) e^{-rho^2/2}`, area
element `rho d(rho) d(phi)`. Put `u` along angle `0` and `v` at angle
`theta`, so `w.u = ||u|| rho cos(phi)` and `w.v = ||v|| rho cos(phi - theta)`.

### 6.1 kappa0 as an orthant probability

```
kappa0(u, v) = E[ 1(w.u > 0) 1(w.v > 0) ].
```

This is the probability that a random direction lies in **both** half-planes
`{w.u > 0}` and `{w.v > 0}`. Each half-plane is an arc of angular width
`pi` (a half-circle); their normals differ by `theta`, so their
intersection is an arc of width `pi - theta`. A uniform random direction
(the angle `phi` is uniform for an isotropic Gaussian) lands there with
probability

```
kappa0 = (pi - theta) / (2 pi).
```

(This is exactly the `arccos` law: `theta = arccos(u.v / (||u|| ||v||))`,
hence the name "arc-cosine kernel".)

### 6.2 kappa1 via a polar integral

```
kappa1(u, v) = E[ relu(w.u) relu(w.v) ].
```

The product is nonzero only where **both** projections are positive, i.e.
`cos(phi) > 0` and `cos(phi - theta) > 0`, which for `0 <= theta <= pi` is
the arc `phi in (theta - pi/2, pi/2)`. On that arc,

```
kappa1 = (1/2 pi) INT_arc  [ ||u|| rho cos(phi) ][ ||v|| rho cos(phi - theta) ]
                            e^{-rho^2/2} rho d(rho) d(phi)
       = (||u|| ||v|| / 2 pi) * [ INT_0^inf rho^3 e^{-rho^2/2} d(rho) ]
                              * [ INT_arc cos(phi) cos(phi - theta) d(phi) ].
```

**Radial integral.** Substitute `t = rho^2 / 2`:

```
INT_0^inf rho^3 e^{-rho^2/2} d(rho) = INT_0^inf 2 t e^{-t} d(t) = 2.
```

**Angular integral.** Use `cos(phi) cos(phi - theta) = 1/2[ cos(theta) +
cos(2 phi - theta) ]` and integrate over the arc of length `pi - theta`:

```
INT_{theta - pi/2}^{pi/2} 1/2[ cos(theta) + cos(2 phi - theta) ] d(phi)
   = 1/2 [ (pi - theta) cos(theta) + sin(theta) ],
```

where the second term comes from
`1/2 [ sin(2 phi - theta) ]` evaluated at the endpoints:
`sin(pi - theta) - sin(theta - pi) = sin(theta) - (-sin(theta)) = 2 sin(theta)`,
times the `1/2` out front gives `sin(theta)`.

Putting the pieces together (`(1/2 pi) * 2 * angular`):

```
kappa1(u, v) = (||u|| ||v|| / 2 pi) ( sin(theta) + (pi - theta) cos(theta) ),
```

exactly the `kappa1` in `gp/ntk.py`. The `_angles` helper computes
`cos(theta)` (clipped to `[-1, 1]`) and the norms; `kappa0`/`kappa1` apply
these closed forms.

---

## 7. NNGP, NTK, and linearized gradient descent

### 7.1 The network and its two infinite-width limits

The width-`m` network (NTK parameterization, Jacot et al. 2018) is

```
f(x) = sqrt(2/m) * sum_{i=1}^m a_i relu(w_i . x~),   x~ = (x, 1),
```

with `w_i ~ N(0, I_2)` and `a_i ~ N(0, 1)` at init, **both** layers trained.
The `sqrt(2/m)` is in the *function*, not the initialization — that is what
keeps outputs `O(1)` and the tangent kernel deterministic as `m -> inf`.

**NNGP kernel (covariance at initialization).** The `a_i` are independent,
zero-mean, unit-variance, and independent of the `w_i`, so

```
E_init[ f(x) f(x') ] = (2/m) sum_i E[a_i^2] E[ relu(w_i.x~) relu(w_i.x~') ]
                     = (2/m) * m * kappa1(x~, x~')
                     = 2 kappa1(x~, x~')   =  nngp_kernel.
```

**NTK.** The tangent kernel is the parameter-gradient inner product,
`Theta(x, x') = <grad_params f(x), grad_params f(x')>`. Split by layer:

```
df/da_i = sqrt(2/m) relu(w_i.x~)
df/dw_i = sqrt(2/m) a_i 1(w_i.x~ > 0) x~.
```

The `a`-gradients contribute
`(2/m) sum_i relu(w_i.x~) relu(w_i.x~') -> 2 kappa1`. The `W`-gradients
contribute `(2/m) sum_i a_i^2 1(w_i.x~>0) 1(w_i.x~'>0) (x~ . x~')`, and with
`E[a_i^2] = 1` and `E[1(.)1(.)] = kappa0` this tends to
`2 kappa0(x~, x~') (x~ . x~')`. Hence

```
Theta(x, x') = 2 kappa1(x~, x~') + 2 kappa0(x~, x~') (x~ . x~')   =  ntk_kernel.
```

The finite-width network computes exactly these Gram matrices
(`TwoLayerReLU.empirical_ntk`); the tests confirm they converge to the
closed forms as `m` grows.

### 7.2 Linearized gradient descent as a geometric series

Train with full-batch GD on `L = 1/2 ||f(X) - y||^2`. In the linearized
(constant-NTK) regime the function's values on any set move by the NTK times
the residual. Let `r_k = f_k(X) - y` be the train residual after step `k`.
One GD step gives

```
r_{k+1} = (I - lr * Theta) r_k       =>      r_k = (I - lr Theta)^k r_0,
```

where `Theta = Theta(X, X)` is the (symmetric PSD) train NTK. For a **test**
set `X*`, each step moves predictions by `-lr Theta(X*, X) r_k`, so

```
f_k(X*) = f_0(X*) - lr Theta(X*, X) sum_{j=0}^{k-1} (I - lr Theta)^j r_0.
```

The matrix geometric series telescopes (as `(I - M) sum_{j<k} M^j = I -
M^k`, here `M = I - lr Theta` so `lr Theta * sum = I - (I - lr Theta)^k`):

```
sum_{j=0}^{k-1} (I - lr Theta)^j = (lr Theta)^{-1} ( I - (I - lr Theta)^k ).
```

Substituting and cancelling the `lr Theta` (and flipping the sign by writing
`y - f_0` instead of `r_0 = f_0 - y`):

```
f_k(X*) = f_0(X*) + Theta(X*, X) Theta^{-1} ( I - (I - lr Theta)^k ) ( y - f_0(X) ).
```

This is `gd_prediction`, computed through the eigendecomposition
`Theta = Q diag(lam) Q^T`: then `(I - lr Theta)^k = Q diag((1 - lr lam)^k)
Q^T`, and `Theta^{-1}(I - (I - lr Theta)^k) = Q diag((1 - decay)/lam) Q^T`
with `decay = (1 - lr lam)^k`. So the `k`-th power costs one `eigh`, not `k`
matmuls.

**Convergence.** Each mode contracts by `1 - lr lam_i`; the series converges
iff `|1 - lr lam_i| < 1` for all `i`, i.e. `lr < 2 / lam_max` (asserted in
the code). As `k -> inf`, `decay -> 0` and

```
f_inf(X*) = f_0(X*) + Theta(X*, X) Theta^{-1} ( y - f_0(X) ),
```

which is **ridgeless kernel regression** with the NTK, plus the transient
carried by the random initialization `f_0`.

### 7.3 The width subtlety: covariance exact at any width, Gaussianity at 1/m

A point that is easy to state wrongly. The NNGP **covariance**
`E_init[f(x) f(x')] = 2 kappa1` holds *exactly at every finite width* `m` —
it is just the expectation of a sum of `m` i.i.d. terms scaled by `2/m`, no
limit required. What needs `m -> inf` is the **Gaussianity** of `f`: the
output is a sum of `m` i.i.d. contributions, so by the CLT it is only
*approximately* Gaussian at finite width, and its non-Gaussian structure
(the connected 4-point cumulant, which measures joint-Gaussianity failure)
decays as `1/m`. So:

- first two moments (mean 0, covariance `2 kappa1`): exact at any width;
- the *distribution* being a Gaussian process: the large-width statement,
  with `O(1/m)` corrections (Lee et al. 2018).

The same distinction governs the NTK: at finite width the empirical tangent
kernel is a random matrix that both differs from the analytic `Theta` by
`O(1/sqrt(m))` and drifts during training; the clean geometric series above
is the infinite-width idealization the finite-width experiments are measured
against.

---

## 8. Bochner's theorem and random Fourier features

Sections 1–3 build the exact GP, and every one of its costs is a cost of the
`n x n` Gram matrix: `n^2` to store, `n^3/3` to factorize. Random Fourier
features (Rahimi & Recht 2007, implemented in `gp/rff.py`) buy a way out by
writing the kernel as an *explicit* inner product in a finite feature space,
after which the model is ordinary Bayesian linear regression. The reason a
Fourier basis is the right one to randomize over is Bochner's theorem.

### 8.1 Bochner's theorem and the RBF spectral density

**Theorem (Bochner 1932).** A continuous function `k(delta)` on `R^d` is the
covariance of a (weakly) stationary process — equivalently, positive definite
— if and only if it is the Fourier transform of a finite non-negative measure.

Since `k(0) = s2` is finite, that measure can be normalized to a probability
density `p(w)`, the kernel's **spectral density**, giving

```
k(x - x') = s2 * INT p(w) exp(i w . (x - x')) dw = s2 * E_{w ~ p} [ exp(i w . (x - x')) ].
```

The content is the "if and only if": positive definiteness *is* the statement
that a stationary kernel is a superposition of plane waves with non-negative
weights. That superposition is an expectation, and expectations can be
estimated by Monte Carlo — which is the whole idea.

For the RBF the spectral density is available in closed form because the
Gaussian is its own Fourier transform. Take `p(w) = N(0, l^{-2} I_d)`. Its
characteristic function is the standard Gaussian one,

```
E_{w ~ N(0, Sigma)} [ exp(i w . delta) ] = exp( -1/2 delta^T Sigma delta ),
```

so with `Sigma = l^{-2} I`,

```
s2 * E [ exp(i w . delta) ] = s2 * exp( -||delta||^2 / (2 l^2) ) = k_RBF(delta).   (8.1)
```

No constants to chase: **the RBF's spectral density is exactly `N(0, l^{-2}I)`**.
Read it as the frequency-domain meaning of the lengthscale — a *short* `l`
means a *wide* spread of frequencies, i.e. a wiggly prior. The ARD kernel
(Sec. 4) is the same statement per coordinate, `p(w) = N(0, diag(l_d^{-2}))`,
which is why `RFFMap` accepts a vector lengthscale unchanged.

Because `k` is real and `p` is symmetric (`p(w) = p(-w)`), the imaginary part
integrates to zero and

```
k(x - x') = s2 * E_w [ cos( w . (x - x') ) ].                                    (8.2)
```

### 8.2 From an expectation to a feature map

Equation (8.2) is an average of `cos(w . x - w . x')`, and the angle-subtraction
identity splits that into a product of terms in `x` and `x'` separately:

```
cos(a - b) = cos a cos b + sin a sin b.
```

So drawing `w_1, ..., w_m ~ p` i.i.d. and defining

```
z(x) = sqrt(s2 / m) [ cos(w_1.x), ..., cos(w_m.x), sin(w_1.x), ..., sin(w_m.x) ]  in R^D,  D = 2m
```

gives, exactly,

```
z(x) . z(x') = (s2 / m) SUM_j [ cos(w_j.x) cos(w_j.x') + sin(w_j.x) sin(w_j.x') ]
             = (s2 / m) SUM_j cos( w_j . (x - x') ),                              (8.3)
```

an average of `m` i.i.d. copies of the random variable inside (8.2). Hence

```
E[ z(x) . z(x') ] = k(x, x')      — the estimator is unbiased at every pair.
```

**Rahimi & Recht's original map** uses one feature per frequency with a random
phase, `z_RR(x) = sqrt(2 s2 / D) cos(w.x + b)`, `b ~ U(0, 2 pi)`. It is also
unbiased, via `2 cos(A + b) cos(B + b) = cos(A - B) + cos(A + B + 2b)` and
`E_b[cos(A + B + 2b)] = 0`. But that second term is pure noise, and it shows up
in the variance. Write `kt(delta) = k(delta)/s2` and `A = w.(x - x')`. For a
single term of each estimator, using `E[cos^2 A] = (1 + kt(2 delta))/2`:

```
Var_paired / s2^2 = (1 + kt(2 delta))/2 - kt(delta)^2        (m = D/2 terms)
Var_offset / s2^2 = (1 + kt(2 delta))/2 + 1/2 - kt(delta)^2  (D terms)
```

so, dividing by the number of terms, paired beats offset exactly when

```
(1 + kt(2 delta))/2 - kt(delta)^2 < 1/2      <=>      kt(2 delta) / 2 < kt(delta)^2.
```

For the RBF, `kt(2 delta) = exp(-2u)` and `kt(delta)^2 = exp(-u)` with
`u = ||delta||^2 / l^2`, and `exp(-2u)/2 < exp(-u)` for every `u >= 0`. The
paired map is therefore strictly better *everywhere* for this kernel
(Sutherland & Schneider 2015), which `tests/test_rff.py` also measures.

The special case `delta = 0` is worth stating on its own. The paired variance
vanishes identically there (`kt(0) = 1` gives `(1+1)/2 - 1 = 0`), and indeed

```
z(x) . z(x) = (s2 / m) SUM_j [ cos^2(w_j.x) + sin^2(w_j.x) ] = s2   for every x,
```

by the Pythagorean identity, *per frequency* — not in expectation. The prior
variance is reproduced exactly, so the error bars are not polluted by noise in
`k(x, x)`. The offset map has `Var/s2^2 = 1/(2D)` there, and its `k(x, x)`
fluctuates around `s2`.

### 8.3 The rate, and what it costs

(8.3) is a plain Monte Carlo average, so its standard deviation falls as
`m^{-1/2} ~ D^{-1/2}` at every pair of inputs, with **no dependence on the
input dimension** `d` (the dimension enters only through how hard the target
function is). Rahimi & Recht additionally prove a *uniform* bound: the sup
over a compact set of diameter `R` is `O_p( sqrt(d log(R/eps)) / sqrt(D) )`,
i.e. the same rate up to a log factor.

The exchange rate is unforgiving and worth internalizing: **4x the features
buys 2x the accuracy**. `experiments/rff.py` measures 13.5x accuracy for 256x
features (the rate predicts 16x). RFF is a tool for making `n` large, not for
making error small.

### 8.4 Weight space and function space are the same posterior

Once the frequencies are drawn they are *fixed*, and the model

```
f(x) = z(x) . w,   w ~ N(0, I_D),   y = f(x) + eps,   eps ~ N(0, sigma^2)
```

is a Gaussian process with the **finite-rank kernel** `k_D(x, x') = z(x).z(x')`
— exactly, not approximately: `Cov[f(x), f(x')] = z(x)^T E[w w^T] z(x') =
z(x).z(x')`. The approximation lives entirely in `k_D ~ k`; everything after
it is exact inference. Standard Bayesian linear regression (R&W Sec. 2.1) with
`A = Z^T Z + sigma^2 I_D` gives

```
w | y ~ N( A^{-1} Z^T y,  sigma^2 A^{-1} ),
mean(x*) = z(x*) . A^{-1} Z^T y,     var(x*) = sigma^2 z(x*)^T A^{-1} z(x*).      (8.4)
```

The equivalence of (8.4) with the function-space formulas of Sec. 1 applied to
`k_D` is the matrix inversion lemma; `tests/test_rff.py` checks it numerically
to `1e-8` by running `RFFRegressor` and `GPRegressor(RFFMap(...))` on the same
data. That test is what licenses the claim that the *only* error is
Monte Carlo.

The costs are the point of the whole exercise:

| | exact GP | RFF |
|---|---|---|
| build | `n^2` kernel entries | `n D` features (`n D d` flops) |
| solve | `n^3 / 3` Cholesky of `K_y` | `n D^2` Gram + `D^3 / 3` Cholesky of `A` |
| memory | `n^2` | `n D` |
| predict (per point) | `n^2` | `D^2` |

Linear in `n` instead of cubic — so the two curves have different slopes on a
log-log plot and cross (measured at just past `n = 500` for `D = 512`, with
252x at `n = 8000`).

### 8.5 Where it breaks: variance starvation

A rank-`D` prior has `D` degrees of freedom in total. As `n` grows past `D`
the data determines essentially all of them, `A = Z^T Z + sigma^2 I` becomes
dominated by `Z^T Z`, and the posterior variance (8.4) shrinks *everywhere* —
including at inputs far from any data, where the exact GP correctly returns
something near the prior `s2`. The model becomes confidently wrong precisely
in the region where uncertainty was the reason to use a GP at all (Wang et al.
2018).

There is no fix within RFF other than raising `D`: the honest reading is that
RFF approximates the posterior *mean* cheaply, and its error bars are
trustworthy only while `D` is large relative to the number of independent
directions the data pins down. `experiments/rff.py` measures the effect at
`n = 3000` with a gap in the inputs: the exact posterior sd at the gap centre
is 0.97, `D = 2048` gives 0.94, and `D = 64` gives 0.088.

### 8.6 Note on hyperparameters

`RFFMap` exposes no gradients, so ML-II (Sec. 2) cannot be run through it as
written: the frequencies `w_j` are drawn *from* a density that depends on `l`,
so changing `l` changes the model. The practical recipe used here is to fit
the exact GP's hyperparameters first (on a subset if `n` is large) and then
build the map at those values. It is not a fundamental obstacle — the
reparameterization `w_j = eps_j / l` with `eps_j ~ N(0, I)` fixed makes `z(x)`
a differentiable function of `l`, and the evidence of the finite-rank model is
then differentiable in the usual way — but that path is not implemented, and
optimizing the evidence of `k_D` is not the same as optimizing the evidence of
`k`.

---

## 9. The variational sparse GP (Titsias): inducing points and the free-energy bound

Section 8 bought `O(n)` scaling by replacing the kernel with a randomized
finite-rank surrogate, and paid for it in Sec. 8.5: a rank-`D` prior runs out
of degrees of freedom and the error bars collapse. This section takes the other
route to the same `O(n M^2)` cost. Instead of approximating the *kernel*, keep
the model exactly as it is and approximate the *posterior*, with a variational
bound whose free parameters are `M` input locations `Z` (Titsias 2009,
implemented in `gp/sparse.py`). The two failure modes are different, and the
README's sparse section measures which one bites in a data gap.

### 9.1 Augmentation is exact; the approximation comes later

Introduce **inducing variables** `u = f(Z)` at `M` **inducing inputs**
`Z in R^{M x d}`. There is nothing approximate about this: `u` is the same GP
evaluated at `M` more inputs, so `p(f, u)` is the model's own joint and

```
INT p(y | f) p(f, u) df du = INT p(y | f) p(f) df = p(y),                        (9.1)
```

the marginal likelihood is untouched. Write `Kuu = k(Z, Z)`, `Kfu = k(X, Z)`,
`Kff = k(X, X)`. Gaussian conditioning (Sec. 1) on the joint gives

```
p(u)    = N(u | 0, Kuu)
p(f|u)  = N( f | Kfu Kuu^{-1} u,  Kff - Qff ),   Qff := Kfu Kuu^{-1} Kuf.        (9.2)
```

`Qff` is the **Nystrom approximation** to `Kff`: the part of `f` that `u`
explains. Its complement `Kff - Qff` is a conditional covariance, hence PSD,
and it is zero exactly when `u` determines `f`. Both facts get used below.

### 9.2 The variational family, and why `Z` cannot overfit

Approximate `p(f, u | y)` by

```
q(f, u) = p(f | u) q(u),        q(u) = N(u | m, S) free.                         (9.3)
```

The choice that makes the whole thing work is keeping the *exact* conditional
`p(f | u)` as the first factor rather than a free `q(f | u)`: it is what makes
the `Kff` terms cancel in a moment, and it encodes the modelling assumption
that `u` is a sufficient summary of `f`.

The standard evidence lower bound, with the augmented model (9.1):

```
F = E_q[ log p(y | f) ] - KL( q(f, u) || p(f, u) )  <=  log p(y).                (9.4)
```

Now the point that governs everything else about this method. `Z` appears
*only* in `q`, never in the model: the left side of (9.1) does not depend on
`Z`, so `log p(y)` does not either. Moving `Z` therefore cannot change what is
being bounded — it can only change how tight the bound is. Maximizing `F` over
`Z` is minimizing `KL(q || posterior)`, and adding more inducing points can
only improve the bound toward the same fixed ceiling. This is the structural
difference from a model parameter, whose optimum trades data fit against a
complexity penalty and can overfit; `Z` has nothing to trade. It is also the
difference from FITC (Sec. 9.7), where `Z` *does* enter the model.

### 9.3 Collapsing the bound

Substitute (9.3) into (9.4). The `p(f | u)` in `q` cancels against the one in
`p(f, u) = p(f | u) p(u)`, leaving only the `u` marginals in the KL:

```
F = INT q(u) [ INT p(f|u) log p(y|f) df ] du  -  KL( q(u) || p(u) ).             (9.5)
```

**The inner integral.** With `p(y | f) = N(y | f, sigma^2 I)` and
`f | u ~ N(mu_u, Sigma)`, `mu_u = Kfu Kuu^{-1} u`, `Sigma = Kff - Qff`:

```
E_{p(f|u)} [ log N(y | f, sigma^2 I) ]
    = -n/2 log(2 pi sigma^2) - 1/(2 sigma^2) E || y - f ||^2
    = -n/2 log(2 pi sigma^2) - 1/(2 sigma^2) [ ||y - mu_u||^2 + tr Sigma ]
    = log N(y | mu_u, sigma^2 I)  -  1/(2 sigma^2) tr( Kff - Qff ).              (9.6)
```

The bias-variance split `E||y - f||^2 = ||y - E f||^2 + tr Cov(f)` is the only
step, and the trace term drops out of the `u`-integral because `Sigma` does not
depend on `u`. So

```
F = INT q(u) log N(y | Kfu Kuu^{-1} u, sigma^2 I) du - KL(q(u)||p(u))
    - 1/(2 sigma^2) tr(Kff - Qff).                                               (9.7)
```

**Optimal `q(u)`.** The first two terms of (9.7) are

```
INT q(u) log [ N(y | Kfu Kuu^{-1} u, sigma^2 I) p(u) / q(u) ] du,
```

which is the standard variational bound on `log INT N(y | Kfu Kuu^{-1}u,
sigma^2 I) p(u) du` — a bound saturated by the exact posterior of the *linear*
Gaussian model `y = (Kfu Kuu^{-1}) u + eps`, `u ~ N(0, Kuu)`. That marginal is
available in closed form: the linear map sends `Cov(u) = Kuu` to
`Kfu Kuu^{-1} Kuu Kuu^{-1} Kuf = Qff`. Substituting the maximizer collapses `q`
out entirely and leaves

```
F = log N( y | 0, Qff + sigma^2 I )  -  1/(2 sigma^2) tr( Kff - Qff ).           (9.8)
```

Read (9.8) as two pieces that pull in opposite directions:

- The first is exactly the log evidence of a **different, low-rank model** —
  the deterministic-inducing-conditional (DTC) approximation, `Kff -> Qff`.
  On its own it is not a bound on anything: `Qff` is a *smaller* covariance
  than `Kff`, so DTC can happily report an evidence *above* the truth.
- The second term, `-tr(Kff - Qff) / (2 sigma^2)`, is what turns it into a
  bound. It is a sum of `n` non-negative terms, `k(x_i, x_i) - q(x_i, x_i)`,
  each one the posterior variance of `f(x_i)` given `u` — the amount of `f`
  that the inducing set fails to explain. It is scale-free in a useful sense:
  it is measured in units of `sigma^2`, so the same geometric shortfall matters
  more when the data are precise.

That is the whole mechanism. **The trace term is the penalty on a bad inducing
set**, and it is what stops the optimizer from parking `Z` somewhere cheap.

### 9.4 The exact-recovery limit, which is also the test

Set `Z = X`, so `M = n` and `Kfu = Kuu = Kff`. Then

```
Qff = Kff Kff^{-1} Kff = Kff        =>        tr(Kff - Qff) = 0,
```

and (9.8) reduces to `log N(y | 0, Kff + sigma^2 I)`, the exact log marginal
likelihood of Sec. 2. The bound is tight, as it must be: `u = f(X)` is a
sufficient summary of `f` at the data, so nothing is being approximated. The
same substitution in the predictive equations below returns the exact
posterior. `tests/test_sparse.py::test_z_equals_x_recovers_the_exact_gp` is
that statement — the algebra is pinned to an already-trusted implementation
*before* anything approximate is measured. It checks the bound to `1e-7` and
the predictive variance to `1e-8`; the reason those are not `1e-16` is the
jitter, quantified in Sec. 9.5.

### 9.5 Predictive equations, and the numerics that make them safe

The maximizer of (9.7) is `q(u) = N(u | m, S)` with the linear-Gaussian
posterior for `u`. Writing `Sig := Kuu + sigma^{-2} Kuf Kfu`,

```
S = Kuu Sig^{-1} Kuu,        m = sigma^{-2} Kuu Sig^{-1} Kuf y.                  (9.9)
```

Pushing that through `p(f* | u)` (Sec. 1 again) and marginalizing `u`:

```
mean(X*) = sigma^{-2} K*u Sig^{-1} Kuf y
var(X*)  = K** - K*u Kuu^{-1} Ku*  +  K*u Sig^{-1} Ku*.                         (9.10)
```

The middle term is the drop from conditioning on `u`; the last adds back the
uncertainty *in* `u`. Far from every inducing point the two middle-and-last
terms cancel and the variance returns to the prior `k(x*, x*)` — the property
random Fourier features lose (Sec. 8.5), and the reason to expect sparse GPs to
keep their error bars in a gap. "Far from every inducing point" is load-bearing
in that sentence, and Sec. 9.8 measures what happens when it fails.

**Numerics.** No inverse is formed and nothing bigger than `M x M` is
factorized. With `Luu = chol(Kuu + jitter I)` and

```
A  = Luu^{-1} Kuf                (M, n)      =>   Qff = A^T A
B  = I_M + sigma^{-2} A A^T      (M, M),     LB = chol(B)
c  = sigma^{-2} LB^{-1} A y      (M,)
```

the matrix determinant lemma and Woodbury turn (9.8) into `O(n M^2)`:

```
log|Qff + sigma^2 I| = n log sigma^2 + 2 SUM_i log (LB)_ii
y^T (Qff + sigma^2 I)^{-1} y = sigma^{-2} y^T y - c^T c
tr(Kff - Qff) = SUM_i k(x_i, x_i) - ||A||_F^2                                   (9.11)
```

— note the trace never builds `Kff`, only its diagonal. For prediction,
`Sig = Luu B Luu^T`, so with `As = Luu^{-1} Ku*` and `tmp = LB^{-1} As`:

```
mean = tmp^T c,      var = diag(K**) - colsum(As^2) + colsum(tmp^2).            (9.12)
```

The `jitter` on `Kuu` is not cosmetic here the way it is for `K_y` in Sec. 3.
`Kuu` carries *no* noise term, and optimizing `Z` (Sec. 9.2) pushes inducing
points toward informative regions, which means toward each other; `Kuu`
genuinely approaches singularity during a fit. The consequence is that `Z = X`
recovery is exact only up to the jitter: perturbing `Kuu -> Kuu + jI` replaces
`Qff` by `Kff(Kff + jI)^{-1}Kff`, whose eigenvalues are
`lam - lam^2/(lam + j) = lam j / (lam + j) <= j` below `Kff`'s. So the recovery
error is `O(j)` and the residual trace term is `O(n j)`, not exactly zero.
`test_recovery_error_is_exactly_the_jitter` measures that: sweeping `j` over
four decades moves both by a factor of ten per decade, with no floor — which is
what an implementation with correct algebra and one loaded diagonal looks like,
and what an algebra error would not.

### 9.6 Gradients of the bound

Everything the optimizer needs comes out of one contraction, and the point of
doing it carefully is that the three gradients — kernel hyperparameters, noise,
and the inducing locations `Z` — all fall out of the *same* two matrices. Write

```
W := Qff + sigma^2 I,     alpha := W^{-1} y,     T := tr(Kff - Qff),

F = -n/2 log(2 pi) - 1/2 log|W| - 1/2 y^T W^{-1} y - T / (2 sigma^2).           (9.13)
```

Differentiating with the two standard identities `d log|W| = tr(W^{-1} dW)` and
`d(W^{-1}) = -W^{-1} (dW) W^{-1}`:

```
dF = 1/2 tr[ (alpha alpha^T - W^{-1}) dW ] - 1/(2 sigma^2) tr(dKff - dQff)
     + T / (2 sigma^4) d(sigma^2).                                              (9.14)
```

The last term is the `sigma^2` in the *denominator* of the trace penalty, and it
is easy to lose. Now split by which quantity moved.

**Kernel hyperparameters.** `sigma^2` is fixed, so `dW = dQff`, and (9.14)
becomes `1/2 tr[H dQff] - 1/(2 sigma^2) tr(dKff)` with

```
H := alpha alpha^T - W^{-1} + sigma^{-2} I,                                     (9.15)
```

where the `+sigma^{-2} I` is precisely the `+1/(2 sigma^2) tr(dQff)` half of the
trace penalty folded in. That is the whole trick: **the penalty term is not a
separate gradient, it is a rank-correction to the one matrix `H` that everything
contracts against**, plus a diagonal-only piece from `Kff`.

Differentiating `Qff = Kfu Kuu^{-1} Kuf` with `P := Kuu^{-1} Kuf` (shape `M x n`)
and `d(Kuu^{-1}) = -Kuu^{-1} dKuu Kuu^{-1}`:

```
dQff = (dKfu) P + P^T (dKuf) - P^T (dKuu) P.                                    (9.16)
```

The first two terms are transposes of each other and `H` is symmetric, so they
contract identically. Setting

```
R := P H        (M x n),        S := P H P^T     (M x M),                       (9.17)
```

and writing `<A, B>` for the elementwise sum `sum_ij A_ij B_ij`, the whole
hyperparameter gradient is

```
dF/dtheta = <R, dKuf/dtheta> - 1/2 <S, dKuu/dtheta>
            - 1/(2 sigma^2) SUM_i d k(x_i, x_i)/dtheta.                         (9.18)
```

Three things to notice. `Kff` appears only through its *diagonal*, so the
`O(n M)` memory of Sec. 9.5 survives differentiation — which is why
`gp/sparse.py` has a blocked `kernel_grad_diag` rather than calling
`kernel.grads(X)` and taking a diagonal. `dKuf/dtheta` is a *cross*-covariance
gradient, `Z` against `X`: the exact GP never needs one, which is why
`kernel.grads` had to grow a second argument. And `R` and `S` are computed once
and reused for every parameter, so adding a hyperparameter costs one kernel
gradient, not a refactorization.

**Noise.** Here `dW = I d(sigma^2)` and `Kff`, `Qff` do not move, so from (9.14),
in the log parameterization the repo uses everywhere (`d sigma^2 / d log sigma^2
= sigma^2`):

```
dF/d(log sigma^2) = sigma^2 [ 1/2 ( alpha^T alpha - tr W^{-1} ) + T/(2 sigma^4) ]. (9.19)
```

The bracket's first half is exactly the exact-GP noise gradient of Sec. 2. The
second half is new and it always *raises* the optimal noise: a sparse model that
cannot explain `T` worth of function variation would rather call it noise. That
is a real bias, not a bug, and it is the mechanism Sec. 9.7 turns on its head:
VFE pays for the shortfall in `sigma^2`, FITC hides it in the covariance
instead, and fits a noise level below the truth as a result.

**Inducing locations.** `Z` enters `Kuu` and `Kuf` and *not* `Kff` — the
statement of Sec. 9.2 that `Z` is variational, now visible as a missing term.
Only row `m` of `Kuf` depends on `z_m`, while `Kuu` touches `z_m` twice (row and
column). Writing `d1 k(a, b)` for the derivative in the first slot:

```
d Kuf / d z_ma has one nonzero row:  [d1 k(z_m, x_j)]_a
d Kuu / d z_ma has a row and a column, giving
    <S, dKuu/dz_ma> = 2 SUM_j S_mj [d1 k(z_m, z_j)]_a     (S symmetric, k symmetric)
```

so the factor of `1/2` in (9.18) cancels and

```
dF/dz_ma = SUM_j R_mj [d1 k(z_m, x_j)]_a  -  SUM_j S_mj [d1 k(z_m, z_j)]_a.     (9.20)
```

Two forces, and the sign structure is the content: the first pulls `z_m` toward
data the current inducing set explains badly, the second pushes it away from the
other inducing points. The `j = m` term of the second sum is *not* a special
case — `k(z_m, z_m)` is constant for a stationary kernel, so `d1 k(z_m, z_m) = 0`
and the diagonal drops out on its own. (`tests/test_kernels.py::
test_input_derivative_vanishes_at_coincident_points_for_stationary_kernels`
checks that rather than assuming it.)

**Computing `R` and `S` without an `n x n`.** `H` in (9.15) is `n x n` and must
never be formed. With the Sec. 9.5 factors `A = Luu^{-1} Kuf`, `B = I + sigma^{-2}
A A^T`, and Woodbury, two simplifications do all the work:

```
A W^{-1} = sigma^{-2} B^{-1} A          [ since A A^T / sigma^2 = B - I ]
P        = Luu^{-T} A
alpha    = ( y - A^T L_B^{-T} c ) / sigma^2
```

so `R = Luu^{-T} [ (A alpha) alpha^T + sigma^{-2} (A - B^{-1}A) ]` and `S = R P^T`,
both `O(n M^2)` with nothing larger than `M x n` alive. `tr W^{-1}` for (9.19) is
`(n - sigma^{-2} <A, B^{-1}A>) / sigma^2` by the same identity.

**Where the jitter goes, and why it is not negligible.** Sec. 9.5 loads
`Kuu -> Kuu + j * mean(diag Kuu) I`, and that loading is itself a function of
`theta` (through `s2`), so the honest `dKuu/dtheta` carries
`+ j * mean(diag dKuu) I`. The tempting argument for dropping it — `j` is
`1e-10`, so who cares — is wrong, and it is worth seeing why, because the same
trap appears anywhere a regularizer is differentiated.

The term does not enter the gradient on its own. It enters contracted against
`S = P H P^T`, and `P = Kuu^{-1} Kuf` carries an inverse, so the omitted
contribution scales like `j * ||P||^2` — with the *square* of `Kuu`'s
conditioning, not with `j` alone. It was found the way such things should be
found: a finite-difference check that RBF passed and Periodic failed, at seven
inducing points that a period-2 kernel sees as near-duplicates
(`cond(Kuu) ~ 6e4`). The log-`s2` gradient came out `1.3e-4` relative off — five
orders of magnitude larger than `j`, and flat in `eps`, which is the signature
of a real error rather than of finite-difference truncation (truncation shrinks
and then grows again as `eps` falls; this did neither).

Including the term is one line and makes the reported gradient exact for the
objective actually being optimized. The `Z` block needs no analogous correction,
but only for a reason worth naming: every kernel here that *has* an input
derivative is stationary, so `mean(diag Kuu)` does not depend on `Z` at all. A
non-stationary kernel with a `dK_dX1` would need it.

**The check.** `tests/test_sparse.py` central-differences all three blocks —
hyperparameters, `log sigma^2`, and every coordinate of `Z` at three random
inducing sets each, for RBF, both differentiable Materns, RQ, Periodic, a sum
and a product in 1D, and for RBF and ARD in 2D. The
`Z` block is the one that is easy to get subtly wrong (drop the factor of 2 on
the `Kuu` term and the bound still increases, just toward the wrong place), so
it is checked at random `Z` rather than at a single convenient configuration.

### 9.7 FITC: the same factorization, the shortfall moved

Sec. 9.6 left the trace penalty as *the* thing that makes `F` a bound. FITC
(Snelson & Ghahramani 2006) is what happens if you decline to pay it, and it is
worth deriving here rather than citing, because the two methods share almost all
of their algebra and disagree about exactly one diagonal. `experiments/fitc.py`
measures the consequence; this section is why the measurement comes out the way
it does.

**The model.** Instead of penalizing `Kff - Qff`, put its diagonal back into the
prior:

```
Lambda := diag(Kff - Qff),      q(f) = N(0, Qff + Lambda).                       (9.21)
```

`Qff + Lambda` has the exact marginal variances `k(x_i, x_i)` of the true prior
and Nystrom's rank-`M` structure everywhere off the diagonal. That is the whole
idea, and it is a *different prior*, not a bound on this one — which is why the
objective is an ordinary log evidence:

```
L_FITC = log N( y | 0, Qff + Lambda + sigma^2 I ).                               (9.22)
```

Two consequences follow immediately and both are testable. There is no Jensen
step anywhere in (9.21)–(9.22), so `L_FITC` is not a lower bound on `log p(y)`
and not an upper one either: `Qff <= Kff` pushes it up (the DTC effect of
Sec. 9.3) and `+Lambda` pushes it back down, and neither wins in general.
`tests/test_sparse.py` exhibits both signs on one dataset — 253 nats below the
exact evidence at `M = 16`, 1.1 nats above it at `M = 32`. And `Z` now enter the
*model*, so the Sec. 9.2 argument that they cannot overfit does not apply to
them: optimizing `Z` here is model selection, with everything that implies.

**One factorization for both.** Write the observation diagonal as

```
D := sigma^2 I              (VFE)          D := Lambda + sigma^2 I    (FITC)
W := Qff + D,               A := Luu^{-1} Kuf,     B := I + A D^{-1} A^T,
c := L_B^{-1} A D^{-1} y.                                                        (9.23)
```

Every Sec. 9.5 formula survives the substitution unchanged in form:
`log|W| = SUM_i log D_i + 2 SUM_i log (L_B)_ii`, `y^T W^{-1} y = SUM_i y_i^2/D_i
- c^T c`, and the predictive equations (9.12) are *identical* — `mean = tmp^T c`,
`var = diag(K**) - colsum(As^2) + colsum(tmp^2)` — because the derivation of
(9.10) never assumed `D` was constant. So `gp/sparse.py` implements both methods
in one code path with one branch, in `fit`, on what goes into `D`.

**One gradient for both, up to one vector.** Differentiating (9.22) with the
same two identities as (9.14), and writing `H_0 := alpha alpha^T - W^{-1}` and
`h := diag(H_0)`:

```
dL_FITC = 1/2 tr[ H_0 dW ],    dW = dQff + diag( dKff_diag - diag(dQff) )
        = 1/2 <H_0 - diag(h), dQff>  +  1/2 <h, dKff_diag>.                      (9.24)
```

The `-diag(h)` is the entire content of FITC: `Lambda`'s dependence on `theta`
and `Z` cancels the diagonal of `dQff` exactly, so **FITC's `H` has a zero
diagonal**. The model has stopped caring how well `Qff` reproduces `Kff` at the
data — it gets that right by construction, at every `Z`, for free. Setting

```
v := sigma^{-2} 1   (VFE)          v := -h   (FITC),        H := H_0 + diag(v)
```

both gradients are the *same* expression,

```
dF/dtheta = <R, dKuf/dtheta> - 1/2 <S, dKuu/dtheta> - 1/2 <v, dKff_diag/dtheta>
dF/dz_ma  = SUM_j R_mj [d1 k(z_m, x_j)]_a - SUM_j S_mj [d1 k(z_m, z_j)]_a        (9.25)
```

with `R = P H` and `S = P H P^T` as in (9.17). Check the VFE column against
(9.18): `v = sigma^{-2}` turns the last term into `-(1/(2 sigma^2)) SUM_i
dk(x_i,x_i)`, which is the trace penalty. One vector is the difference between
the two methods' hyperparameter and `Z` gradients. The `Z` block is unchanged in
*form* for a second reason worth stating: `Kff` does not depend on `Z` under
either method, so the `v` term simply is not there — `Lambda` moves with `Z`,
but only through `Qff`, which `R` and `S` already carry.

**The noise block is where they part.** For (9.22), `dW/d sigma^2 = I` and
nothing else moves, because `Lambda` is a function of `(theta, Z)` alone:

```
dL_FITC / d(log sigma^2) = sigma^2 * 1/2 ( alpha^T alpha - tr W^{-1} ).          (9.26)
```

That is the exact-GP noise gradient of Sec. 2 and nothing more. Compare (9.19):
VFE carries `+T/(2 sigma^4)`, strictly positive whenever the inducing set is
imperfect, which always pushes the fitted noise *up*. **This one missing term is
the pathology.** Measured at a shared `(theta, sigma^2)` with an imperfect
inducing set (`tests/test_sparse.py::test_the_two_noise_gradients_point_in_
opposite_directions`): the exact GP wants less noise (gradient `-33.8`), FITC
agrees with it (`-9.3`), and VFE alone wants more (`+382.7`). The two fits
separate from the first ascent step, and they separate in the direction the
algebra says.

The mechanism in words: a sparse model has variation it cannot explain. VFE can
only call it noise, and `sigma^2` is one number shared by every data point, so
calling it noise is expensive and honest. FITC has `Lambda`, which is `n` free
non-negative numbers — a per-point noise level it never has to justify — so it
charges the misfit there and leaves `sigma^2` to fit whatever is left. The
result is a noise variance biased low and, where the data actually are,
predictive intervals too narrow. `experiments/fitc.py` measures both, and also
the half of the failure that is easy to miss: *between* the data clusters
FITC's variance is too **wide**, so a predictive-sd average over a uniform grid
reports it as conservative (1.215x the exact GP's) at the same `M` where
held-out data say it is overconfident (0.956x).

**What FITC is not.** It is not a worse approximation to the same posterior — it
is a good fit to a different model, one whose prior has `n` extra variance
parameters. Nothing above says it predicts badly in general; what it says is
that its `sigma^2` is not the data's noise level and its error bars are not the
exact GP's, so the two should not be read as if they were. `gp/sparse.py`
implements it to be measured, not recommended.

### 9.8 When the band does *not* relax: DTC's overconfidence

Sec. 9.5 read (9.10) as a safety property — far from every inducing point the
two corrections cancel and the variance returns to the prior. The converse is
the part that matters in practice, and it is not a caveat but a measured
failure (`experiments/rff_vs_sparse.py`, measurement 4).

Split (9.10) into the two things it is made of:

```
var(x*) = [ k** - Q** ]  +  [ K*u Sig^{-1} Ku* ],      Q** = K*u Kuu^{-1} Ku*.  (9.27)
```

The first bracket is a *margin*: the prior variance at `x*` that the inducing
set cannot explain, which is `>= 0` and is large exactly when `x*` is far from
every `z`. The second is the posterior variance of the inducing values,
read out at `x*`. The margin is what makes the band relax; the second bracket
is what the model actually claims to know.

Now put an inducing point inside an empty region. Then `Q** -> k**`, the margin
goes to zero, and the whole error bar is inherited from the second bracket —
and that bracket is *not* the exact GP's posterior over `f(z)`. Under the
variational family of Sec. 9.2, `p(f | u)` is kept exact, but the resulting
`q(u)` in (9.9) has precision `Kuu^{-1} + sigma^{-2} Kuu^{-1} Kuf Kfu Kuu^{-1}`:
every one of the `n` observations contributes to pinning down `M` numbers,
weighted by `k(z, x_i)^2 / sigma^2`, whether or not any single `x_i` is near
`z`. With `n` large and `sigma^2` small, a `z` that no datum is close to can
still be over-determined by the crowd of data that is merely *nearby*. The
exact GP conditions `f(x*)` on the data directly and makes no such claim.

Measured, at the gap centre of the Sec. 11 setting (`n = 3000`, no data in
`|x| < 1.2`, exact posterior sd `0.9655`, `M = 12`):

| `Z` rule | nearest `z` | `k** - Q**` | `K*u Sig^-1 Ku*` | sd |
|---|---|---|---|---|
| data quantiles | `-1.424` | 0.9991 | 0.0000 | 0.9996 |
| blind uniform grid | `+0.364` | 0.0931 | 0.0446 | 0.3710 |

Same data, same `M`, same kernel; the second is overconfident by 2.6x because
it spent its margin. So "inducing points keep the error bar" is a statement
about `k** - Q**`, not about inducing points — and a placement rule that
follows the *data* (quantiles, a random subset of `X`) preserves it for free,
while one that follows the *domain* does not. This is also the sense in which
(9.10) is the DTC predictive rather than the exact posterior: the variational
argument of Sec. 9.2 bounds the marginal likelihood, and says nothing about the
predictive distribution being conservative pointwise. It is not.
`test_an_isolated_inducing_point_in_a_gap_is_overconfident` pins it.

### 9.9 What is deliberately not here yet

- **SVGP** (Hensman et al. 2013) keeps `q(u)` uncollapsed so the bound
  decomposes over data points and can be minibatched, and admits non-Gaussian
  likelihoods. The collapsed bound (9.8) cannot: it needs all of `y` at once.
- Non-stationary and cusped kernels. `Gibbs` has no input-space derivative
  derived here and `Matern nu=0.5` has none to derive, so (9.20) does not apply
  to either; both raise rather than return something plausible.

---

## 10. The Laplace approximation: a likelihood that is not Gaussian

Everything above this section rests on one structural fact: a Gaussian prior
and a Gaussian likelihood give a Gaussian posterior, so Sec. 1's conditioning
formula does the whole job. Replace $p(y_i \mid f_i)$ with anything else —
Bernoulli labels, Poisson counts — and the posterior

$$p(f \mid y) \;=\; \frac{p(y \mid f)\, \mathcal{N}(f \mid 0, K)}{p(y)},
\qquad p(y) = \int p(y \mid f)\, \mathcal{N}(f \mid 0, K)\, df$$

has no closed form and the evidence is an $n$-dimensional integral. The Laplace
approximation is the cheapest honest answer: fit the Gaussian that agrees with
the posterior at its mode, to second order.

### 10.1 The objective and its mode

Write the un-normalized log posterior as

$$\Psi(f) \;=\; \log p(y \mid f) \;-\; \tfrac12 f^\top K^{-1} f
\;-\; \tfrac12 \log|K| - \tfrac{n}{2}\log 2\pi ,$$

and drop the last two terms, which do not depend on $f$. Then

$$\nabla \Psi = \nabla \log p(y \mid f) - K^{-1} f,
\qquad
\nabla\nabla \Psi = -W - K^{-1},
\qquad
W \equiv -\nabla\nabla \log p(y \mid f),$$

and $W$ is **diagonal** because the likelihood factorizes over the observations
— that single fact is what makes everything below $O(n^3)$ rather than worse.
Setting the gradient to zero gives the mode equation

$$\boxed{\;\hat f \;=\; K \,\nabla \log p(y \mid \hat f)\;}\tag{10.1}$$

which is worth reading before solving: the posterior mode is the kernel matrix
applied to the likelihood's own residual, exactly as the GP regression mean is
$K(K+\sigma^2 I)^{-1}y$. It is also a free correctness check on any solver —
`LaplaceFit.stationarity` evaluates both sides by different routes and reports
the gap.

If $\log p(y\mid f)$ is concave in $f$ then $\Psi$ is strictly concave (the
quadratic term is), so the mode is **unique** and Newton's method reaches it
from anywhere. Bernoulli-logit ($W = \sigma(f)(1-\sigma(f)) \le 1/4$), Poisson
with a log link ($W = e^f$) and Gaussian ($W = \sigma^{-2}$) are all concave.

### 10.2 The Newton step, and why $K$ is never factorized

The Newton update is

$$f^{\text{new}} \;=\; f + (K^{-1} + W)^{-1}\!\left(\nabla \log p(y\mid f) - K^{-1} f\right).$$

Written this way it needs $K^{-1}$, which is exactly what one must not form: a
smooth kernel makes $K$ numerically singular long before $n$ is large, and this
repo's own Sec. 14 measured what carelessness there costs. Two rearrangements
fix it. First multiply out, using $b \equiv W f + \nabla \log p(y \mid f)$:

$$f^{\text{new}} = (K^{-1}+W)^{-1}\big[(K^{-1}+W)f + \nabla\log p - K^{-1}f\big]
= (K^{-1}+W)^{-1} b .$$

Now apply the Woodbury identity to $(K^{-1}+W)^{-1}$ with
$B \equiv I + W^{1/2} K W^{1/2}$:

$$(K^{-1} + W)^{-1} \;=\; K - K W^{1/2} B^{-1} W^{1/2} K. \tag{10.2}$$

(Check it by multiplying on the left by $K^{-1}+W$ and using
$W^{1/2}KW^{1/2} = B - I$; every term cancels.) So with $a \equiv K^{-1}
f^{\text{new}}$,

$$\boxed{\;a = b - W^{1/2} B^{-1} W^{1/2} K b, \qquad f^{\text{new}} = K a\;}
\tag{10.3}$$

and **only $B$ is factorized**. That matters because $B$ is symmetric positive
definite with eigenvalues in $[1,\, 1 + n\max_i W_{ii} \max_{ij}|K_{ij}|]$: its
conditioning depends on the *sizes* of $K$ and $W$, never on $K$'s smallest
eigenvalue. $W^{1/2}$ is real precisely because the likelihood is log-concave,
which is where that assumption is actually spent.

The Newton direction is an ascent direction ($\Psi$ concave), but the full step
can overshoot when the curvature moves quickly — Poisson's $W = e^f$ changes by
$e$ for every unit of $f$. Damping is done in $a$ coordinates rather than $f$:
since $f = Ka$ is linear, the convex combination $a_t = a + t(a^{\text{new}}-a)$
maps to $f_t = f + t(f^{\text{new}}-f)$, so $\Psi(f_t) = \sum_i \log p(y_i \mid
f_{t,i}) - \tfrac12 a_t^\top f_t$ costs no extra solve.

### 10.3 The approximate evidence

Expanding $\Psi$ to second order about $\hat f$ and integrating the Gaussian,

$$p(y) \;\approx\; e^{\Psi(\hat f)} \int \exp\!\left(-\tfrac12 (f-\hat f)^\top
(K^{-1}+W)(f-\hat f)\right) df
\;=\; e^{\Psi(\hat f)} (2\pi)^{n/2} |K^{-1}+W|^{-1/2},$$

where the linear term vanishes because $\hat f$ is the mode. Restoring the
$-\tfrac12\log|K| - \tfrac n2 \log 2\pi$ that was dropped, the $(2\pi)^{n/2}$
cancels and the determinants combine:

$$|K^{-1}+W|\,|K| = |I + KW| = |I + W^{1/2}KW^{1/2}| = |B|,$$

(the middle equality is $|I+AB| = |I+BA|$), so

$$\boxed{\;\log \hat Z \;=\; \log p(y\mid \hat f) - \tfrac12 \hat f^\top K^{-1}\hat f
\;-\; \sum_i \log L_{ii}, \qquad L L^\top = B. \;}\tag{10.4}$$

The log-determinant is free once $B$ has been factorized for the Newton step.

**This is not a bound.** The variational free energy of Sec. 9 is below $\log
p(y)$ by a KL divergence, always; (10.4) is a saddle-point expansion, and its
error is whatever the posterior's departure from Gaussianity contributes, with
no sign guaranteed by the derivation. Empirically, on the logit likelihood, it
comes out **one-signed and negative** — the true posterior has heavier tails
than the Gaussian fitted at its mode, so the fitted Gaussian's normalizer is
too small. Sec. 16 measures that: $-0.007$ nats at $s^2 = 0.25$ growing to
$-0.173$ at $s^2 = 64$, against an importance-sampling estimate of the exact
evidence, and the same sign at $n=1$ against exact quadrature.

### 10.4 Prediction

The latent at a new point is Gaussian under $q$, with

$$\mathbb{E}_q[f_*] = k_*^\top K^{-1}\hat f = k_*^\top \nabla\log p(y\mid \hat f)
\tag{10.5}$$

by the mode equation (10.1) — the same shape as GP regression's $k_*^\top
\alpha$, with the likelihood residual in place of $(K+\sigma^2I)^{-1}y$ — and,
substituting (10.2),

$$\mathbb{V}_q[f_*] = k_{**} - k_*^\top (K + W^{-1})^{-1} k_*
= k_{**} - v^\top v, \qquad v = L^{-1} W^{1/2} k_* . \tag{10.6}$$

Read (10.6) as the usual variance reduction with the likelihood's *local*
curvature standing in for the noise precision. Where an observation is
confidently classified, $W_{ii} \to 0$ and that point removes almost no
variance — which is why a GP classifier stays uncertain in a region of
saturated labels, and is the honest behaviour rather than a defect.

Finally, the quantity actually wanted is usually not $f_*$ but
$\mathbb{E}[y_*]$, which needs the one-dimensional average

$$\pi_* = \int g(f_*)\, \mathcal{N}(f_* \mid \mu_*, \sigma_*^2)\, df_* ,$$

with $g = \sigma$ for the logit link. Gauss-Hermite quadrature does this to
machine precision at a few tens of nodes. Skipping it and reporting
$g(\mu_*)$ is a real error and a one-signed one: $\sigma$ is concave above its
inflection and convex below, so Jensen puts $\pi_*$ strictly closer to $1/2$
than $\sigma(\mu_*)$ at every point. Sec. 16 measures a gap of 0.078 in
probability where the approximation's own error against the exact answer is
0.034 — the shortcut costs more than the approximation does.

### 10.5 Hyperparameter gradients: the implicit term, and its sign

$\hat f$ is itself a function of $\theta$, so the derivative of (10.4) is not
the derivative of its explicit $\theta$-dependence:

$$\frac{d \log \hat Z}{d\theta_j}
= \underbrace{\frac{\partial \log \hat Z}{\partial \theta_j}}_{\text{explicit}}
+ \sum_i \underbrace{\frac{\partial \log \hat Z}{\partial \hat f_i}
\frac{\partial \hat f_i}{\partial \theta_j}}_{\text{implicit}} .$$

The implicit term does not vanish, and the reason is worth being exact about:
$\hat f$ is a stationary point of $\Psi$, not of $\log\hat Z$, and the two
differ by $-\tfrac12\log|B|$, whose $W$ is evaluated *at the mode*. Move the
mode and the determinant moves with it.

**The explicit part.** Hold $\hat f$, hence $W$, fixed. Only
$-\tfrac12 \hat f^\top K^{-1}\hat f$ and $\log|B|$ see $\theta$. With
$a = K^{-1}\hat f$, the first gives $\tfrac12 a^\top \frac{\partial K}{\partial\theta_j} a$.
For the second, split $|B| = |I + KW| = |K|\,|K^{-1}+W|$ and differentiate both
factors:

$$-\tfrac12\operatorname{tr}\!\left(K^{-1}\frac{\partial K}{\partial\theta_j}\right)
+ \tfrac12\operatorname{tr}\!\left(K^{-1}(K^{-1}+W)^{-1}K^{-1}\frac{\partial K}{\partial\theta_j}\right)
= -\tfrac12\operatorname{tr}\!\left((K + W^{-1})^{-1}\frac{\partial K}{\partial\theta_j}\right),$$

the last step being Woodbury, $K^{-1} - K^{-1}(K^{-1}+W)^{-1}K^{-1} = (K+W^{-1})^{-1}$.
So

$$\boxed{\ \frac{\partial \log \hat Z}{\partial \theta_j}
= \tfrac12 a^\top \frac{\partial K}{\partial\theta_j} a
- \tfrac12 \operatorname{tr}\!\left(R\,\frac{\partial K}{\partial\theta_j}\right),
\qquad R \equiv (K + W^{-1})^{-1} = W^{1/2}B^{-1}W^{1/2}.\ }$$

$R$ is the same $B$-Cholesky the fit already has, so $K$ is still never
factorized — the point Sec. 10.2 made about the mode, carried through to the
gradient.

**Sensitivity to the mode.** $\Psi$ is stationary at $\hat f$, so only the
determinant contributes. Using $|B| = |K|\,|K^{-1}+W|$ again, with $K$ free of
$\hat f$:

$$\frac{\partial \log \hat Z}{\partial \hat f_i}
= -\tfrac12 \operatorname{tr}\!\left((K^{-1}+W)^{-1}\frac{\partial W}{\partial \hat f_i}\right)
= -\tfrac12 \left[(K^{-1}+W)^{-1}\right]_{ii} \frac{\partial W_i}{\partial \hat f_i},$$

$W$ being diagonal. And $W = -\nabla\nabla\log p(y|f)$ gives
$\partial W_i/\partial \hat f_i = -\partial^3 \log p(y|\hat f)/\partial f_i^3$,
so the two minus signs cancel:

$$\boxed{\ \frac{\partial \log \hat Z}{\partial \hat f_i}
= +\tfrac12 \left[(K^{-1}+W)^{-1}\right]_{ii}\,
\frac{\partial^3 \log p(y|\hat f)}{\partial f_i^3}.\ }$$

R&W (5.23) prints this with a minus sign while defining $W = -\nabla\nabla\log p$
on the previous page; the two are not consistent, and GPML's `infLaplace` uses
the $+\tfrac12$ form. Rather than pick by authority, `tests/test_laplace.py`
runs the flipped sign against a central difference of $\log\hat Z$ itself and
requires it to *fail* — the wrong sign yields a vector that points broadly the
right way, which is exactly why it survives a casual check.

**Sensitivity of the mode to $\theta$.** Differentiate the mode equation
$\hat f = K\,\nabla\log p(y|\hat f)$ (Sec. 10.1) in $\theta_j$, remembering that
$\nabla\log p$ is evaluated at $\hat f$:

$$\frac{\partial \hat f}{\partial\theta_j}
= \frac{\partial K}{\partial\theta_j}a - KW\frac{\partial \hat f}{\partial\theta_j}
\quad\Longrightarrow\quad
\frac{\partial \hat f}{\partial\theta_j}
= (I + KW)^{-1}\frac{\partial K}{\partial\theta_j}a .$$

And $(I+KW)^{-1} = I - KR$ — check by expanding, using $R = (I+WK)^{-1}W$ — so
the implicit term costs nothing new: the same $R$, and no $W^{-1}$ anywhere.
That last point is not cosmetic. A confidently-classified Bernoulli point has
$W_i$ at $10^{-16}$, and $W^{-1}$ would be the only badly conditioned object in
a derivation that has otherwise been careful to avoid one.

The diagonal $[(K^{-1}+W)^{-1}]_{ii}$ is read off $K - KRK$ (Woodbury once
more). Total cost: one extra $O(n^3)$ for $R$ and $KR$, then $O(n^2)$ per
hyperparameter — against $O(n^3)$ *per grid point*, so the gradient pays for
itself at three grid points and its cost is flat in the parameter count where
the grid's is exponential.

Implemented as `LaplaceGP.log_evidence_grad`, with `maximize_evidence` running
Adam on it. Three checks, because this is the piece most able to be plausibly
wrong: it reproduces `GPRegressor.lml_and_grad` to $10^{-8}$ under a Gaussian
likelihood (where $\partial^3\log p = 0$, so the implicit term is exactly zero
and only the explicit box is being tested); it matches central differences to
$2\times10^{-5}$ relative for Bernoulli and Poisson across five kernel families
(where it is not); and deleting the implicit term, or flipping its sign, fails
that same check.

**Measured (`experiments/laplace.py`, §5).** On 24 points with 15% flipped
labels, an RBF kernel: the $21\times17$ grid spends 357 fits and reports
$\log\hat Z = -14.7613$; Adam from three different starts spends 250 each and
all three land on $(\hat s^2, \hat\ell) = (2.481, 0.762)$ at $-14.7513$. Same
optimum, found between grid points, and no sign of the multimodality Sec. 9
found under a Gaussian likelihood — on *this* surface.

The three-parameter case is the one that argues for the gradient, and it argues
for something more specific than speed. A $13^3$ RationalQuadratic grid spends
$2197$ fits and returns $\alpha = 50$ — **the top of its own grid**, which
`on_edge` flags. Adam spends 250 and beats it ($-14.7515$ against $-14.7708$),
but the three ascents finish at $\alpha = 57.8$, $63.6$ and $771$ while agreeing
on $s^2$ and $\ell$ to three digits and on the evidence to $0.003$ nats. Neither
method resolves $\alpha$, and the reason is structural rather than numerical:
$\text{RQ} \to \text{RBF}$ as $\alpha\to\infty$, so the evidence really is flat
in that direction and there is no interior optimum to find. The gradient's
contribution is that it makes the flatness visible in a few hundred fits instead
of hiding it behind a boundary argmax.

### 10.6 What is missing here, and why it is not small

**Expectation propagation is not implemented.** On binary classification EP is
usually the more accurate approximation, and often by more than the gap Sec. 16
measures — so those numbers are what *Laplace* costs, not what approximate
inference costs.

**Nothing here is sparse.** Every quantity above factorizes an $n\times n$
matrix, so the inducing-point machinery of Sec. 9 and this section do not meet:
there is no sparse GP classifier in this repo, which is what the combination
would be and what most of the applied literature actually uses.

**The oracle runs out before the method does.** The importance sampler that
judges all of this draws from the prior, and its effective sample size falls
below 0.5% by $n = 32$ (Sec. 16), so the error measurements stop where the
sampler stops rather than where the approximation does.

---

## 11. Exercises

Five problems whose answers are already somewhere in this repo — as a line of
code, a measured number, or a test. Solutions are collapsed; each one ends with
the check that keeps it honest.

### Exercise 1 — Leave-one-out without refitting

`GPRegressor.loo` returns, for every training point, the predictive
distribution of the GP refit on the *other* $n-1$ points, at the cost of one
factorization it had already done. Write $\tilde K = K + \operatorname{diag}(\sigma^2)$
for the noisy covariance and $\alpha = \tilde K^{-1} y$.

**(a)** Order the points so that $i$ comes first and partition $\tilde K$ into
blocks. Using the block-inverse formula, show that

$$[\tilde K^{-1}]_{ii} = \big(\tilde K_{ii} - \tilde K_{i,-i}\tilde K_{-i,-i}^{-1}\tilde K_{-i,i}\big)^{-1},$$

and identify the bracket as something you have already derived elsewhere in
this document.

**(b)** Deduce $\sigma_i^2 = 1/[\tilde K^{-1}]_{ii}$ and
$\mu_i = y_i - \alpha_i/[\tilde K^{-1}]_{ii}$ (R&W eqs. 5.10–5.12).

**(c)** Is $\sigma_i^2$ a *latent* or an *observation* predictive variance?
And why is $\sum_i \log p(y_i \mid X, y_{-i})$ a defensible model-selection
score when the log marginal likelihood is already available?

<details>
<summary>Solution</summary>

**(a)** With the ordering $(i, -i)$ the block-inverse formula gives the $(1,1)$
entry of $\tilde K^{-1}$ as the inverse of the **Schur complement** of
$\tilde K_{-i,-i}$ in $\tilde K$:

$$[\tilde K^{-1}]_{ii} = \big(\tilde K_{ii} - \tilde K_{i,-i}\tilde K_{-i,-i}^{-1}\tilde K_{-i,i}\big)^{-1}.$$

The bracket is exactly the conditional variance of Sec. 1: it *is*
$\operatorname{Var}[y_i \mid y_{-i}]$, the posterior variance at $x_i$ of a GP
conditioned on the other points, noise included. So the LOO variance is not
merely computable from $\tilde K^{-1}$ — it is a diagonal entry of
$\tilde K^{-1}$, upside down. Nothing about leaving a point out is new work;
Gaussian conditioning already did it.

**(b)** Inverting (a) gives $\sigma_i^2 = 1/[\tilde K^{-1}]_{ii}$ directly. For
the mean, read off row $i$ of $\tilde K^{-1}$, which the same block formula
gives as

$$[\tilde K^{-1}]_{i,\cdot} = [\tilde K^{-1}]_{ii}\,\big[\,1,\; -\tilde K_{i,-i}\tilde K_{-i,-i}^{-1}\,\big]$$

in the $(i,-i)$ ordering. Hitting $y$ with it,

$$\alpha_i = [\tilde K^{-1}]_{ii}\big(y_i - \tilde K_{i,-i}\tilde K_{-i,-i}^{-1} y_{-i}\big) = [\tilde K^{-1}]_{ii}\,(y_i - \mu_i),$$

since the second bracket is precisely the GP posterior mean at $x_i$ from the
other $n-1$ points. Rearranged, $\mu_i = y_i - \alpha_i/[\tilde K^{-1}]_{ii}$,
and $\alpha_i/[\tilde K^{-1}]_{ii} = \alpha_i \sigma_i^2$ is the LOO residual.
Cost: one $O(n^3)$ Cholesky (already paid by `fit`) plus $O(n^3)$ for
$\tilde K^{-1}$, versus $n$ refits at $O(n^4)$.

**(c)** *Observation*-level. $\tilde K$ carries the noise on its diagonal, so
the bracket in (a) is the variance of $y_i$, not of $f(x_i)$; it matches
`predict(..., include_noise=True)` at the held-out point. Hence the residual
$y_i - \mu_i$ is being compared against a band that is supposed to contain a
noisy observation, which is the honest comparison for CV.

As a score, the LOO log predictive density never conditions a point on itself,
whereas the marginal likelihood is a statement about the model's fit to the
whole dataset *including* $y_i$. The two disagree in a useful way: the evidence
can be maximized by a model that is confident and wrong out of sample (the CO2
extrapolation in §5 of the README is exactly that failure), while
$\sum_i \log p(y_i \mid y_{-i})$ penalizes over-confidence point by point.
R&W Sec. 5.4.2 makes the same argument, adding that LOO-CV is more robust when
the model is misspecified — which, for a hand-chosen kernel, it always is.

*Check it:* `tests/test_gp.py::test_closed_form_loo_matches_brute_force_refits`
compares both returned quantities against $n$ actual refits on the other
$n-1$ points, and `::test_closed_form_loo_matches_brute_force_heteroscedastic`
does it again with per-point noise — both to $10^{-8}$.

</details>

### Exercise 2 — The other optimum ML-II can fall into

`experiments/multistart.py` shows the evidence of an RBF + noise model on 12
sparse points having two optima: a short-lengthscale "it's signal" one at
$-5.40$ nats and a long-lengthscale "it's all noise" one at $-13.74$. This
exercise identifies the second one exactly.

**(a)** Let $\ell \to \infty$ with $s^2$ and $\sigma^2$ fixed. Show
$\tilde K \to s^2 \mathbf{1}\mathbf{1}^\top + \sigma^2 I$, and use
Sherman–Morrison and the matrix determinant lemma to write the log marginal
likelihood in closed form — no linear algebra left.

**(b)** Maximize that expression over $s^2$ and $\sigma^2$ for data with
$\bar y \approx 0$. What does $s^2$ do, and what does $\sigma^2$ become?

**(c)** Evaluate it for the dataset in `_multimodal_1d` ($n = 12$,
$y^\top y/n = 0.577$) and compare with the $-13.74$ the README reports. What
does the comparison say about *why* multi-start, rather than a better local
optimizer, is the right fix?

<details>
<summary>Solution</summary>

**(a)** $k_{\mathrm{RBF}}(x,x') = s^2 \exp(-\lVert x-x'\rVert^2/2\ell^2) \to s^2$
for every pair as $\ell \to \infty$, so $\tilde K \to s^2 J + \sigma^2 I$ with
$J = \mathbf{1}\mathbf{1}^\top$. Sherman–Morrison and the determinant lemma
give

$$\tilde K^{-1} = \frac{1}{\sigma^2}\Big(I - \frac{s^2\,\mathbf{1}\mathbf{1}^\top}{\sigma^2 + n s^2}\Big), \qquad |\tilde K| = (\sigma^2)^{n-1}(\sigma^2 + n s^2),$$

hence

$$\log p(y) = -\frac{1}{2\sigma^2}\Big(y^\top y - \frac{s^2 (\textstyle\sum_i y_i)^2}{\sigma^2 + n s^2}\Big) - \frac{n-1}{2}\log \sigma^2 - \frac{1}{2}\log(\sigma^2 + n s^2) - \frac{n}{2}\log 2\pi.$$

Only two scalars of the data survive: $y^\top y$ and $\sum_i y_i$. An infinitely
long lengthscale is a model with one degree of freedom — a constant — plus
noise.

**(b)** The only place $s^2$ helps is the term $s^2(\sum y_i)^2/(\sigma^2+ns^2)$,
which shrinks the quadratic form by explaining the sample *mean*; and the only
place it hurts is $\tfrac12\log(\sigma^2 + n s^2)$. With $\bar y \approx 0$ the
first is worth nothing, so the penalty wins and $s^2 \to 0$: the constant
component is switched off entirely. What is left is the i.i.d. Gaussian model
$y \sim N(0, \sigma^2 I)$, maximized at $\hat\sigma^2 = y^\top y/n$, giving

$$\mathrm{LML}^* = -\frac{n}{2}\Big[1 + \log\big(2\pi\, y^\top y/n\big)\Big].$$

**(c)** With $n = 12$ and $y^\top y/n = 0.577$ this is $-13.73$. The README's
"noise-mode single start" row is $-13.74$, at a fitted $\ell = 98$ on a domain
of width 8 — i.e. numerically infinite. So the second optimum is not a rival
*explanation* with an interesting lengthscale; it is the flat plateau at the
end of the lengthscale axis, where the model has given up and called the data
noise. That is why the fix is multi-start and not a better local optimizer: the
plateau is genuinely, correctly flat, gradient information there points nowhere
useful, and the only way out is to start somewhere else. It is also why the
evidence gap ($8.3$ nats) is large — these are not two nearby fits.

*Check it:* `tests/test_gp.py::test_evidence_at_a_huge_lengthscale_matches_the_rank_one_closed_form`
pins (a) against `GPRegressor` at $\ell = 10^6$ to $10^{-8}$, and
`::test_the_all_noise_basin_is_the_infinite_lengthscale_plateau` pins (b) and
(c), including that the long-lengthscale fit lands within $0.05$ nats of the
plateau value.

</details>

### Exercise 3 — Warping preserves positive definiteness (and where that argument runs out)

**(a)** Let $k$ be a positive semi-definite kernel on $\mathcal{U}$ and let
$u : \mathcal{X} \to \mathcal{U}$ be *any* map. Show that
$k'(x,x') = k(u(x), u(x'))$ is positive semi-definite on $\mathcal{X}$.

**(b)** Use it to conclude that the periodic kernel of Sec. 5.2,
$k_{\mathrm{per}}(x,x') = s^2\exp(-2\sin^2(\pi|x-x'|/p)/\ell^2)$, is a valid
kernel, and note what (a) buys you that a direct Bochner argument would have
to work for.

**(c)** The Gibbs kernel of `gp/kernels.py` gives every input its own
lengthscale $\ell(x)$. Explain why (a) does **not** establish that it is PSD,
and say what does — in particular, what job the prefactor
$\sqrt{2\ell(x)\ell(x')/(\ell(x)^2+\ell(x')^2)}$ is doing.

<details>
<summary>Solution</summary>

**(a)** Take any points $x_1,\dots,x_n$ and any $c \in \mathbb{R}^n$. Then

$$\sum_{i,j} c_i c_j\, k'(x_i,x_j) = \sum_{i,j} c_i c_j\, k(u(x_i), u(x_j)) \ge 0,$$

because the right-hand side is the PSD quadratic form of $k$ evaluated at the
points $u(x_1),\dots,u(x_n) \in \mathcal{U}$. Nothing is required of $u$ — not
continuity, not injectivity, not even measurability. Repeated images are fine:
if $u(x_1) = u(x_2)$ the Gram matrix is singular, which is permitted (*semi*-
definite), and it says the prior treats those two inputs as the same point.

**(b)** Take $u(x) = (\cos(2\pi x/p), \sin(2\pi x/p))$, the wrapping of the
line onto a circle, and $k$ the RBF on $\mathbb{R}^2$. Then
$\lVert u(x)-u(x')\rVert^2 = 4\sin^2(\pi(x-x')/p)$, so
$k(u(x),u(x')) = s^2\exp(-2\sin^2(\pi(x-x')/p)/\ell^2)$ — the periodic kernel
exactly, PSD for free. The alternative route, showing directly that
$\exp(-2\sin^2(\pi\tau/p)/\ell^2)$ has a non-negative Fourier transform in
$\tau$ (Bochner, Sec. 8.1), is true but real work; MacKay's construction gets
it by *composition*, which is the general lesson: build new kernels from old
ones with operations that preserve the quadratic form (sums, products,
warpings) rather than verifying each from scratch.

**(c)** The Gibbs kernel is not $k(u(x),u(x'))$ for any fixed base kernel $k$
and map $u$. Its *shape* changes from point to point — a wide bump near inputs
where $\ell$ is large, a narrow one where $\ell$ is small — and (a) only ever
relabels inputs, it never reshapes the kernel. Composition arguments cannot
produce a nonstationary kernel out of a stationary one.

What does work is Gibbs's (1997) construction, which builds the kernel as an
honest inner product: place a Gaussian basis function of width $\ell(x)$ at
every input and integrate their product over the whole line. That integral
*is* $\int \phi_x(u)\phi_{x'}(u)\,du$, manifestly a PSD form, and evaluating it
in closed form produces both factors — the exponential
$\exp(-(x-x')^2/(\ell(x)^2+\ell(x')^2))$ **and** the prefactor
$\sqrt{2\ell(x)\ell(x')/(\ell(x)^2+\ell(x')^2)}$. So the prefactor is not
cosmetic normalization: it is part of what the overlap integral evaluates to,
and dropping it breaks positive-definiteness. It is also what makes the
$\ell \equiv$ const case collapse *exactly* to the RBF (the prefactor becomes
1), which is the cheapest available test of having got it right.

*Check it:* `tests/test_kernels.py::test_kernel_matrices_are_positive_semidefinite`
asserts a non-negative spectrum for every kernel in the library, and
`::test_gibbs_recovers_rbf_exactly_when_the_lengthscale_is_constant` pins the
$b = 0$ collapse — which a missing or mistyped prefactor fails.

</details>

### Exercise 4 — Why random features starve the error bars

Sec. 8.5 reports that with $D$ fixed and $n$ large, the RFF posterior variance
collapses *everywhere*, including far from the data where the exact GP returns
roughly the prior $s^2$. Prove it.

**(a)** From the weight-space posterior (8.4), the predictive variance is
$\operatorname{Var}[f(x_*)] = \sigma^2 z(x_*)^\top A^{-1} z(x_*)$ with
$A = Z^\top Z + \sigma^2 I_D$. Bound it above using $\lambda_{\min}(A)$ and the
exact identity for $\lVert z(x)\rVert^2$ established in Sec. 8.2.

**(b)** Argue that for data covering the input region,
$\lambda_{\min}(Z^\top Z)$ grows linearly in $n$, and conclude the rate at
which the RFF posterior variance goes to zero — at **every** $x_*$.

**(c)** Where does the exact GP's variance go instead, and why does the
argument in (b) not apply to it?

<details>
<summary>Solution</summary>

**(a)** $A$ is symmetric positive definite, so
$z^\top A^{-1} z \le \lVert z\rVert^2/\lambda_{\min}(A)$. By the Pythagorean
identity of Sec. 8.2, $\lVert z(x)\rVert^2 = s^2$ *exactly*, for every $x$ —
not in expectation. Hence

$$\operatorname{Var}[f(x_*)] \;\le\; \frac{\sigma^2 s^2}{\lambda_{\min}(Z^\top Z) + \sigma^2} \qquad\text{for every } x_*.$$

The bound has no $x_*$ in it. Whatever happens to $\lambda_{\min}$ happens to
the error bars everywhere at once, which is already the whole phenomenon.

**(b)** $Z^\top Z = \sum_{i=1}^n z(x_i) z(x_i)^\top$ is a sum of $n$ rank-one
PSD terms. If the inputs are drawn from a distribution whose feature
second-moment matrix $M = \mathbb{E}[z(x)z(x)^\top]$ is nonsingular, the law of
large numbers gives $Z^\top Z \approx n M$, so
$\lambda_{\min}(Z^\top Z) \approx n\,\lambda_{\min}(M) = \Theta(n)$. Then

$$\operatorname{Var}[f(x_*)] = O\!\big(\sigma^2 s^2 / n\big) \to 0$$

uniformly in $x_*$. The mechanism is a counting argument, not a geometric one:
the prior has exactly $D$ degrees of freedom in total, $n \gg D$ observations
pin down all of them, and once every direction of $w$ is determined there is
nothing left anywhere for the model to be uncertain about. Note $M$ is
nonsingular precisely when the $D$ features are linearly independent as
functions on the input distribution — so this is not a pathological case, it is
the generic one.

**(c)** The exact GP's posterior variance at $x_*$ is
$k(x_*,x_*) - k_*^\top \tilde K^{-1} k_*$, and far from the data $k_* \to 0$
(the RBF decays), so the variance returns to the prior $s^2$ no matter how
large $n$ is. The counting argument fails because the exact GP's prior has
*infinitely* many degrees of freedom: $n$ observations can never exhaust them,
and the ones that remain live exactly where the data are not. The finite-rank
approximation replaces "infinitely many directions" with "$D$ directions", and
the error bars are the first thing to notice.

The practical reading is the one Sec. 8.5 states: RFF approximates the posterior
*mean* cheaply, and its uncertainty is trustworthy only while $D$ is large
relative to the number of directions the data determine. `experiments/rff.py`
measures the failure at $n = 3000$ with a gap in the inputs — exact posterior
sd at the gap centre $0.97$, $D = 2048$ gives $0.94$, $D = 64$ gives $0.088$.

</details>

### Exercise 5 — The constant in the heteroscedastic fit

`experiments/heteroscedastic.py` fits a second GP to $z_i = \log r_i^2$, the
log-squared leave-one-out residuals, and then adds $1.2704$ to its prediction
before exponentiating.

**(a)** For $r \sim N(0,s^2)$, compute $\mathbb{E}[\log r^2]$. (Hint: the
fractional moments of $\chi^2_1$ are available in closed form; differentiate
them at $0$.)

**(b)** Why must the correction be applied, and what exactly goes wrong
without it — quantitatively?

**(c)** The residuals are the *leave-one-out* residuals, not the in-sample
ones. Why?

<details>
<summary>Solution</summary>

**(a)** Write $r = s\,z$ with $z \sim N(0,1)$, so $r^2 = s^2 z^2$ and
$\log r^2 = \log s^2 + \log z^2$ with $z^2 \sim \chi^2_1$. For $\chi^2_k$,

$$\mathbb{E}\big[(\chi^2_k)^t\big] = 2^t\,\frac{\Gamma(k/2 + t)}{\Gamma(k/2)},$$

and since $\tfrac{d}{dt}\mathbb{E}[X^t]\big|_{t=0} = \mathbb{E}[\log X]$
(differentiate $X^t = e^{t\log X}$ under the expectation), taking $k=1$ gives

$$\mathbb{E}[\log \chi^2_1] = \log 2 + \psi(1/2) = \log 2 + (-\gamma - 2\log 2) = -\gamma - \log 2 = -1.2704,$$

using $\psi(1/2) = -\gamma - 2\log 2$. So
$\mathbb{E}[\log r^2] = \log s^2 - 1.2704$: the log of a squared draw is a
*biased* estimator of the log variance, and biased by a universal constant that
does not depend on $s$ at all.

**(b)** Because a GP fitted to the $z_i$ estimates $\mathbb{E}[\log r^2]$, which
by (a) is $\log s^2(x) - 1.2704$, not $\log s^2(x)$. Exponentiating without
correcting returns $s^2(x)\,e^{-1.2704} = 0.281\,s^2(x)$ — every noise variance
too small by a factor $3.56$, so every error bar too narrow by $\sqrt{3.56} =
1.89$. That failure is invisible to any shape or smoothness check: the noise
*profile* would be exactly right, the whole curve just sits too low, and the
only symptom is under-coverage — which is the one thing the heteroscedastic fit
exists to fix. Constant bias, silent failure, one line of correction.

The reason it is a constant at all is worth keeping: taking logs turns the
multiplicative $\chi^2_1$ noise into additive noise of *fixed* distribution,
which is exactly the condition under which a homoscedastic GP is the right
model for the second stage. The bias is the price of that convenience, and it
is a known number rather than something to estimate.

**(c)** An interpolating GP drives its in-sample residuals toward zero — with
small noise it can pass arbitrarily close to every training point — so
in-sample $r_i^2$ measures how flexible the first-stage fit was, not how noisy
the data are, and the second stage would inherit a badly under-estimated noise
floor. The LOO residuals of Exercise 1 are predictions of held-out points, so
they contain the noise honestly, and they are free: the same $\alpha_i$ and
$[\tilde K^{-1}]_{ii}$ the closed form already produced.

*Check it:*
`tests/test_gp.py::test_log_chi_squared_bias_constant_is_the_number_it_claims_to_be`
pins the constant against $\gamma + \log 2$ and against a 4-million-draw Monte
Carlo estimate of $\mathbb{E}[\log z^2]$;
`::test_two_stage_heteroscedastic_improves_calibration` is the end-to-end
consequence.

</details>

---

## References

- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
  Learning*, MIT Press, 2006. (Conditioning, ML-II and its gradient, Cholesky
  algorithm, Matern family — Ch. 2, 4, 5, Apx. A.)
- D. J. C. MacKay, *Introduction to Gaussian Processes*, 1998. (Periodic
  kernel via the circle warping.)
- Y. Cho and L. K. Saul, "Kernel Methods for Deep Learning," *NeurIPS* 2009.
  (Arc-cosine kernels `kappa0`, `kappa1` and their closed forms.)
- A. Jacot, F. Gabriel, and C. Hongler, "Neural Tangent Kernel: Convergence
  and Generalization in Neural Networks," *NeurIPS* 2018. (NTK, the
  parameterization, linearized training.)
- J. Lee, Y. Bahri, R. Novak, S. Schoenholz, J. Pennington, and
  J. Sohl-Dickstein, "Deep Neural Networks as Gaussian Processes," *ICLR*
  2018. (Finite-width corrections and the covariance-exact / Gaussianity-at-
  1/m distinction.)
- J. Lee, L. Xiao, S. Schoenholz, Y. Bahri, R. Novak, J. Sohl-Dickstein, and
  J. Pennington, "Wide Neural Networks of Any Depth Evolve as Linear Models
  Under Gradient Descent," *NeurIPS* 2019. (Linearized-GD dynamics.)
- D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization,"
  *ICLR* 2015. (The optimizer used for ML-II ascent.)
- S. Bochner, *Vorlesungen über Fouriersche Integrale*, 1932. (The
  characterization of positive-definite stationary kernels, Sec. 8.1.)
- A. Rahimi and B. Recht, "Random Features for Large-Scale Kernel Machines,"
  *NeurIPS* 2007. (Random Fourier features, the offset map, and the uniform
  error bound.)
- D. J. Sutherland and J. Schneider, "On the Error of Random Fourier
  Features," *UAI* 2015. (Variance of the two maps; the paired map's
  advantage, Sec. 8.2.)
- Z. Wang, C. Gehring, P. Kohli, and S. Jegelka, "Batched Large-Scale Bayesian
  Optimization in High-Dimensional Spaces," *AISTATS* 2018. (Variance
  starvation, Sec. 8.5.)
- M. K. Titsias, "Variational Learning of Inducing Variables in Sparse
  Gaussian Processes," *AISTATS* 2009. (The free-energy bound (9.8), the
  collapsed `q(u)`, and the argument that `Z` are variational parameters.)
- J. Quiñonero-Candela and C. E. Rasmussen, "A Unifying View of Sparse
  Approximate Gaussian Process Regression," *JMLR* 2005. (SoR/DTC/FITC as
  different effective priors; `Qff` and the Nystrom view, Sec. 9.3.)
- E. Snelson and Z. Ghahramani, "Sparse Gaussian Processes using Pseudo-inputs,"
  *NeurIPS* 2006. (FITC, Sec. 9.7.)
- M. Bauer, M. van der Wilk, and C. E. Rasmussen, "Understanding Probabilistic
  Sparse Gaussian Process Approximations," *NeurIPS* 2016. (What VFE and FITC
  each do to the fitted noise and the predictive variance; the claim
  `experiments/fitc.py` tests.)
- J. Hensman, N. Fusi, and N. D. Lawrence, "Gaussian Processes for Big Data,"
  *UAI* 2013. (SVGP: the uncollapsed bound that minibatches, Sec. 9.9.)
