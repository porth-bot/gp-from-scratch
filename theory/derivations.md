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

## 9. Exercises

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
