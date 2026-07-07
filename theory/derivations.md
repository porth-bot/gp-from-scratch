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
