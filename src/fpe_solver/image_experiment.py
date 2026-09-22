"""High-dimensional self-consistency stress test on noisy CIFAR-10 images.

This experiment has an explicit smooth empirical Gaussian-mixture initial
density. It does not train on a target OU score, use a pretrained generator,
or establish generation of unseen semantic content. It is separate from the
periodic 1--2D solver and has no high-dimensional numerical certificate.

Run ``python -m fpe_solver.image_experiment --help`` for the independent CLI.
"""
import argparse
import hashlib
import json
import math
import os
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.special import logsumexp as numpy_logsumexp

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
IMAGE_DIM = 3 * 32 * 32


def bernstein(t, T, degree):
    tau = t / T
    powers = jnp.arange(degree + 1)
    return (jnp.asarray([math.comb(degree, k) for k in range(degree + 1)], dtype=jnp.float64)
            * tau**powers * (1-tau)**(degree-powers))


@dataclass
class GaussianMixture:
    centers: object
    sigma: float = .2

    def __post_init__(self):
        centers = np.asarray(self.centers, dtype=np.float64)
        if centers.ndim != 2 or min(centers.shape) < 1 or not np.isfinite(centers).all():
            raise ValueError("Mixture centers must be a nonempty finite matrix")
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        self.mean = jnp.asarray(centers.mean(axis=0))
        self.global_variance = float(np.mean((centers-centers.mean(axis=0))**2) + self.sigma**2)
        self.centers = jnp.asarray(centers)
        self.dim = centers.shape[1]

    def sample(self, key, count, t=0.):
        labels_key, noise_key = jax.random.split(key)
        labels = jax.random.randint(labels_key, (count,), 0, len(self.centers))
        decay = jnp.exp(-t)
        variance = -jnp.expm1(-2*t) + self.sigma**2*jnp.exp(-2*t)
        return decay*self.centers[labels] + jnp.sqrt(variance)*jax.random.normal(
            noise_key, (count, self.dim), dtype=jnp.float64)

    def log_density_score(self, x, t=0.):
        """Exact finite-mixture density/score; t>0 is evaluation only."""
        centers = jnp.exp(-t)*self.centers
        variance = -jnp.expm1(-2*t) + self.sigma**2*jnp.exp(-2*t)
        distance = (jnp.sum(x*x, axis=-1, keepdims=True)
                    + jnp.sum(centers*centers, axis=-1) - 2*x@centers.T)
        component_logits = -.5*distance/variance
        normalizer = -.5*self.dim*jnp.log(2*math.pi*variance) - math.log(len(self.centers))
        log_density = jax.scipy.special.logsumexp(component_logits, axis=-1) + normalizer
        conditional_center = jax.nn.softmax(component_logits, axis=-1)@centers
        return log_density, (conditional_center-x)/variance


def init_parameters(key, dim, rank=32, degree=3):
    if dim < 1 or not 1 <= rank <= 128 or not 0 <= degree <= 15:
        raise ValueError("Require dim>0, rank in [1,128], degree in [0,15]")
    return {
        "A": jax.random.normal(key, (rank, dim), dtype=jnp.float64) / math.sqrt(dim),
        "B": jnp.zeros((degree+1, dim, rank), dtype=jnp.float64),
        "b": jnp.zeros((degree+1, rank), dtype=jnp.float64),
        "c": jnp.zeros((degree+1, dim), dtype=jnp.float64),
    }


def affine_base(t, mean, variance):
    """Probability-flow coefficients for a Gaussian evolving under OU."""
    q = -jnp.expm1(-2*t) + variance*jnp.exp(-2*t)
    return 1/q-1, -jnp.exp(-t)*mean/q


def affine_coordinates(t, mean, variance):
    q = -jnp.expm1(-2*t) + variance*jnp.exp(-2*t)
    return jnp.exp(-t)*mean, jnp.sqrt(q/variance)


def field_components(params, t, x, mean, base_variance, T):
    weights = bernstein(t, T, params["B"].shape[0]-1)
    B = jnp.einsum("p,pdr->dr", weights, params["B"])
    b = weights@params["b"]
    c = weights@params["c"]
    hidden = jnp.tanh(x@params["A"].T + b)
    a, shift = affine_base(t, mean, base_variance)
    velocity = a*x + shift + hidden@B.T + c
    return velocity, hidden, B, a


def velocity(params, t, x, mean, base_variance, T):
    return field_components(params, t, x, mean, base_variance, T)[0]


def lowrank_terms(params, t, x, score, mean, base_variance, T):
    """Exact v, div(v), (Dv)^T score, grad(div(v)) without a d-by-d matrix."""
    v, hidden, B, a = field_components(params, t, x, mean, base_variance, T)
    A = params["A"]
    derivative = 1-hidden**2
    diagonal_AB = jnp.einsum("rd,dr->r", A, B)
    divergence = x.shape[-1]*a + derivative@diagonal_AB
    jacobian_transpose_score = a*score + ((score@B)*derivative)@A
    grad_divergence = (-2*hidden*derivative*diagonal_AB)@A
    return v, divergence, jacobian_transpose_score, grad_divergence


def integrate(params, mixture, x0, *, steps=128, T=3., base_variance=None):
    """RK4 in the analytically known affine-flow coordinates.

    If x=m(t)+scale(t)*y, integrate y, z=scale*s and
    eta=logrho+d*log(scale). This removes known scalar stiffness without
    approximating the initial score or supervising against a target OU score.
    """
    if steps < 1 or T <= 0:
        raise ValueError("steps and T must be positive")
    variance = mixture.sigma**2 if base_variance is None else base_variance
    log_density, score = mixture.log_density_score(x0)
    state = (x0-mixture.mean, score, log_density, jnp.zeros(x0.shape[:-1], dtype=jnp.float64))
    h = T/steps

    def rhs(t, value):
        y, z, _, _ = value
        mean_t, scale = affine_coordinates(t, mixture.mean, variance)
        x, s = mean_t+scale*y, z/scale
        v, hidden, B, _ = field_components(params, t, x, mixture.mean, variance, T)
        weights = bernstein(t, T, params["B"].shape[0]-1)
        correction = hidden@B.T + weights@params["c"]
        derivative = 1-hidden**2
        diagonal_AB = jnp.einsum("rd,dr->r", params["A"], B)
        divergence = derivative@diagonal_AB
        grad_divergence = (-2*hidden*derivative*diagonal_AB)@params["A"]
        jtz = ((z@B)*derivative)@params["A"]
        return correction/scale, -jtz-scale*grad_divergence, -divergence, jnp.mean((v+x+s)**2, axis=-1)

    def add(left, right, scale):
        return jax.tree_util.tree_map(lambda a, b: a+scale*b, left, right)

    @jax.checkpoint
    def step(value, index):
        t = index*h
        k1 = rhs(t, value)
        k2 = rhs(t+h/2, add(value, k1, h/2))
        k3 = rhs(t+h/2, add(value, k2, h/2))
        k4 = rhs(t+h, add(value, k3, h))
        updated = jax.tree_util.tree_map(
            lambda y, a, b, c, d: y+h*(a+2*b+2*c+d)/6, value, k1, k2, k3, k4)
        return updated, None

    y, z, eta, residual = jax.lax.scan(step, state, jnp.arange(steps))[0]
    mean_T, scale_T = affine_coordinates(T, mixture.mean, variance)
    return mean_T+scale_T*y, z/scale_T, eta-mixture.dim*jnp.log(scale_T), residual


def transport(params, x, *, mean, base_variance, T=3., steps=128, reverse=True):
    """Position-only flow; default T -> 0 uses an independently supplied terminal draw."""
    h = (-T if reverse else T)/steps
    initial_time = T if reverse else 0.
    initial_mean, initial_scale = affine_coordinates(initial_time, mean, base_variance)
    initial_y = (x-initial_mean)/initial_scale

    def f(t, y):
        mean_t, scale = affine_coordinates(t, mean, base_variance)
        position = mean_t+scale*y
        _, hidden, B, _ = field_components(params, t, position, mean, base_variance, T)
        weights = bernstein(t, T, params["B"].shape[0]-1)
        return (hidden@B.T+weights@params["c"])/scale

    def step(value, index):
        t = initial_time+index*h
        k1 = f(t, value)
        k2 = f(t+h/2, value+h*k1/2)
        k3 = f(t+h/2, value+h*k2/2)
        k4 = f(t+h, value+h*k3)
        return value+h*(k1+2*k2+2*k3+k4)/6, None

    final_y = jax.lax.scan(step, initial_y, jnp.arange(steps))[0]
    final_mean, final_scale = affine_coordinates(0. if reverse else T, mean, base_variance)
    return final_mean+final_scale*final_y


def verify_archive(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != CIFAR_MD5:
        raise ValueError("CIFAR-10 archive checksum does not match the official binary dataset")
    return digest.hexdigest()


def download_archive(path):
    path = Path(path)
    if path.exists():
        verify_archive(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(CIFAR_URL, timeout=60) as source:
                total = 0
                while block := source.read(1024*1024):
                    total += len(block)
                    if total > 200*1024**2:
                        raise ValueError("Archive exceeds expected size cap")
                    output.write(block)
            output.flush()
            verify_archive(temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def load_cifar(path, *, dataset_seed=0, train_count=512, heldout_count=256):
    """Read only official fixed binary members; never extract archive paths."""
    verify_archive(path)
    if not 1 <= train_count <= 50000 or not 1 <= heldout_count <= 10000:
        raise ValueError("Invalid subset sizes")
    rng = np.random.default_rng(dataset_seed)
    train_indices = rng.choice(50000, train_count, replace=False)
    test_indices = rng.choice(10000, heldout_count, replace=False)
    train = np.empty((train_count, IMAGE_DIM), dtype=np.uint8)
    with tarfile.open(path, "r:gz") as archive:
        def read_batch(name):
            info = archive.getmember("cifar-10-batches-bin/"+name)
            if not info.isfile() or info.size != 10000*(IMAGE_DIM+1):
                raise ValueError("Unexpected CIFAR member type or size")
            with archive.extractfile(info) as stream:
                content = stream.read(info.size+1)
            if len(content) != info.size:
                raise ValueError("Truncated CIFAR batch")
            return np.frombuffer(content, dtype=np.uint8).reshape(10000, IMAGE_DIM+1)[:, 1:]
        for batch in range(5):
            positions = np.flatnonzero(train_indices//10000 == batch)
            if len(positions):
                train[positions] = read_batch(f"data_batch_{batch+1}.bin")[train_indices[positions] % 10000]
        heldout = read_batch("test_batch.bin")[test_indices]
    return (train.astype(np.float64)/127.5-1, heldout.astype(np.float64)/127.5-1,
            {"train_indices": train_indices.tolist(), "heldout_indices": test_indices.tolist(),
             "dataset_seed": dataset_seed, "archive_md5": CIFAR_MD5, "source": CIFAR_URL})


def nearest_neighbors(samples, references):
    distances = (np.sum(samples*samples, axis=1, keepdims=True)
                 + np.sum(references*references, axis=1)[None]-2*samples@references.T)
    index = np.argmin(distances, axis=1)
    rmse = np.sqrt(np.maximum(0, distances[np.arange(len(samples)), index])/samples.shape[1])
    return index, rmse


def sample_metrics(samples, train, heldout):
    if not np.isfinite(samples).all():
        return {"finite": False}
    train_index, train_rmse = nearest_neighbors(samples, train)
    test_index, test_rmse = nearest_neighbors(samples, heldout)
    equal_count = min(len(train), len(heldout))
    _, equal_train_rmse = nearest_neighbors(samples, train[:equal_count])
    _, equal_test_rmse = nearest_neighbors(samples, heldout[:equal_count])
    distance = (np.sum(samples*samples, axis=1, keepdims=True)
                + np.sum(samples*samples, axis=1)[None]-2*samples@samples.T)
    pairwise = np.sqrt(np.maximum(0, distance[np.triu_indices(len(samples), 1)])/samples.shape[1])
    return {"finite": True, "count": len(samples), "mean_pixel": float(samples.mean()),
            "std_pixel": float(samples.std()), "saturation_fraction": float(np.mean(np.abs(samples)>1)),
            "mean_image_rmse_to_train_mean": float(np.sqrt(np.mean((samples.mean(0)-train.mean(0))**2))),
            "pairwise_rmse_mean": float(pairwise.mean()),
            "train_nearest_rmse_mean": float(train_rmse.mean()),
            "heldout_nearest_rmse_mean": float(test_rmse.mean()),
            "equal_size_reference_comparison": {"count_each": equal_count,
                "train_nearest_rmse_mean": float(equal_train_rmse.mean()),
                "heldout_nearest_rmse_mean": float(equal_test_rmse.mean()),
                "note": "Equal reference counts reduce nearest-neighbor size bias; not a novelty or memorization proof."},
            "train_nearest_indices": train_index.tolist(), "heldout_nearest_indices": test_index.tolist(),
            "train_nearest_rmse": train_rmse.tolist(), "heldout_nearest_rmse": test_rmse.tolist()}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(_json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def save_params(path, params):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in params.items()})
        temporary = stream.name
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_sheet(path, samples, title, columns=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if samples.shape[1] != IMAGE_DIM:
        raise ValueError("Contact sheets require CIFAR-10 pixel dimensions")
    rows = math.ceil(len(samples)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns*1.2, rows*1.2), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index < len(samples):
            image = (np.clip(samples[index].reshape(3, 32, 32).transpose(1, 2, 0), -1, 1)+1)/2
            axis.imshow(image)
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .96), pad=.1)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train_image_model(mixture, *, seed=0, rank=32, degree=3, batch=16, updates=500,
                      steps=128, T=3., learning_rate=1e-3, base="component", max_seconds=1800.,
                      checkpoint=None):
    if base not in ("component", "global"):
        raise ValueError("Unknown affine-base variance")
    if not 1 <= updates <= 500 or not 1 <= batch <= 64 or steps < 1:
        raise ValueError("Require 1--500 updates, 1--64 batch, positive steps")
    if not all(math.isfinite(v) and v > 0 for v in (T, learning_rate, max_seconds)) or max_seconds > 7200:
        raise ValueError("Invalid horizon, learning rate, or instance budget")
    variance = mixture.sigma**2 if base == "component" else mixture.global_variance
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    params = init_parameters(init_key, mixture.dim, rank, degree)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(learning_rate))
    opt_state = optimizer.init(params)

    @jax.jit
    def update(parameters, state, initial):
        def loss(p):
            return jnp.mean(integrate(p, mixture, initial, steps=steps, T=T, base_variance=variance)[-1])
        value, gradient = jax.value_and_grad(loss)(parameters)
        changes, candidate_state = optimizer.update(gradient, state, parameters)
        candidate = optax.apply_updates(parameters, changes)
        candidate_value = loss(candidate)
        norm = optax.global_norm(gradient)
        finite = jnp.isfinite(value) & jnp.isfinite(candidate_value) & jnp.isfinite(norm)
        for leaf in jax.tree_util.tree_leaves(candidate):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return candidate, candidate_state, value, candidate_value, norm, finite

    start = time.monotonic()
    history, status, first_step_seconds = [], "UPDATES_COMPLETED", None
    for index in range(updates):
        if time.monotonic()-start >= max_seconds:
            status = "BUDGET_EXHAUSTED"
            break
        key, sample_key = jax.random.split(key)
        initial = mixture.sample(sample_key, batch)
        step_start = time.monotonic()
        proposed, proposed_state, old_value, value, norm, finite = update(params, opt_state, initial)
        value.block_until_ready()
        if first_step_seconds is None:
            first_step_seconds = time.monotonic()-step_start
        item = {"update": index+1, "loss_per_coordinate": float(value),
                "loss_raw_dimension_sum": float(value)*mixture.dim,
                "previous_loss_same_batch": float(old_value), "gradient_norm": float(norm),
                "elapsed_seconds": time.monotonic()-start, "accepted_finite_update": bool(finite)}
        history.append(item)
        if not bool(finite):
            status = "NONFINITE_UPDATE_REJECTED"
            if checkpoint:
                checkpoint(params, history, status)
            break
        params, opt_state = proposed, proposed_state
        if checkpoint and ((index+1) % 25 == 0 or index+1 == updates):
            checkpoint(params, history, "TRAINING")
    elapsed = time.monotonic()-start
    if checkpoint:
        checkpoint(params, history, status)
    return params, {"status": status, "history": history, "seconds": elapsed,
                    "first_step_compile_and_execution_seconds": first_step_seconds,
                    "base_variance": variance, "updates_accepted": sum(h["accepted_finite_update"] for h in history)}


def exact_terminal_log_density_numpy(mixture, samples, T):
    """Independent NumPy oracle used only after training for forward density diagnostics."""
    centers = math.exp(-T)*np.asarray(mixture.centers)
    variance = -math.expm1(-2*T) + mixture.sigma**2*math.exp(-2*T)
    distance = (np.sum(samples*samples, axis=1, keepdims=True)
                + np.sum(centers*centers, axis=1)[None]-2*samples@centers.T)
    return (numpy_logsumexp(-distance/(2*variance), axis=1) - math.log(len(centers))
            - .5*mixture.dim*math.log(2*math.pi*variance))


def run_experiment(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    training, heldout, data_info = load_cifar(args.archive, dataset_seed=args.dataset_seed)
    mixture = GaussianMixture(training, args.sigma)
    config = vars(args).copy()
    config.pop("handler", None)
    config.update(dataset=data_info, dimension=mixture.dim, train_count=len(training),
                  heldout_count=len(heldout), float64=bool(jax.config.jax_enable_x64),
                  jax_version=jax.__version__, optax_version=optax.__version__,
                  devices=[str(d) for d in jax.devices()],
                  affine_base_note="component variance corrects all isotropic noise directions; rank still limits center structure",
                  integrator={"method": "RK4 in exact affine-flow coordinates",
                              "coordinates": "x=m(t)+scale(t)*y; z=scale*s; eta=logrho+d*log(scale)",
                              "equations": "y'=u/scale; z'=-(Du)^T*z-scale*grad(div u); eta'=-div u",
                              "known_base_is_integrated_exactly": True,
                              "physical_RK4_false_residual_counterexample": {
                                  "dimension": 8, "single_center": 0, "sigma": .2, "T": 3.,
                                  "sample_seed": 1, "sample_count": 3,
                                  "steps128_per_coordinate": .004311210777049851,
                                  "steps256_per_coordinate": .00024322279118960742,
                                  "meaning": "The exact Gaussian base has zero continuum residual; direct physical-coordinate RK4 stages introduced this positive bias."}})
    write_json(out/"config.json", config)
    checkpoint_info = {}

    def checkpoint(parameters, history, status):
        accepted = sum(item["accepted_finite_update"] for item in history)
        filename = f"parameters-{accepted:05d}.npz"
        checkpoint_info.update(file=filename, sha256=save_params(out/filename, parameters),
                               accepted_updates=accepted)
        write_json(out/"training.json", {"status": status, "history": history, "checkpoint": checkpoint_info})

    params, progress = train_image_model(
        mixture, seed=args.seed, rank=args.rank, batch=args.batch, updates=args.updates,
        steps=args.steps, T=args.T, learning_rate=args.learning_rate, base=args.base,
        max_seconds=args.max_seconds, checkpoint=checkpoint)
    evaluation_start = time.monotonic()
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 49017)
    independent_key, oracle_key, forward_key = jax.random.split(key, 3)
    noise = jax.random.normal(independent_key, (64, mixture.dim), dtype=jnp.float64)
    oracle_terminal = mixture.sample(oracle_key, 64, t=args.T)
    variance = progress["base_variance"]
    inverse = jax.jit(lambda points: transport(params, points, mean=mixture.mean,
                      base_variance=variance, T=args.T, steps=args.steps))
    inverse_fine = jax.jit(lambda points: transport(params, points, mean=mixture.mean,
                           base_variance=variance, T=args.T, steps=2*args.steps))
    generated = np.asarray(inverse(noise))
    refined = np.asarray(inverse_fine(noise))
    oracle_generated = np.asarray(inverse_fine(oracle_terminal))
    forward_initial = mixture.sample(forward_key, 64)
    forward = jax.jit(lambda points: integrate(params, mixture, points, steps=2*args.steps,
                      T=args.T, base_variance=variance))(forward_initial)
    forward_x, _, forward_log_density, forward_residual = map(np.asarray, forward)
    oracle_log_density = exact_terminal_log_density_numpy(mixture, forward_x, args.T)
    metrics = sample_metrics(refined, training, heldout)
    oracle_array = np.asarray(oracle_terminal)
    oracle_logp = exact_terminal_log_density_numpy(mixture, oracle_array, args.T)
    normal_logp = -.5*np.sum(oracle_array**2, axis=1)-.5*mixture.dim*math.log(2*math.pi)
    terminal_log_ratio = oracle_logp-normal_logp
    q_terminal = -math.expm1(-2*args.T)+mixture.sigma**2*math.exp(-2*args.T)
    terminal_prior_bound = .5*(math.exp(-2*args.T)*float(np.mean(np.sum(training**2, axis=1)))
                              + mixture.dim*(q_terminal-1-math.log(q_terminal)))
    report = {
        "experiment": "Noisy empirical CIFAR-10 mixture / OU self-consistency feasibility stress test",
        "status": progress["status"], "training": progress, "checkpoint": checkpoint_info,
        "independent_standard_gaussian_generation": metrics,
        "oracle_terminal_diagnostic_only": sample_metrics(oracle_generated, training, heldout),
        "terminal_prior_mismatch": {
            "kl_true_ou_terminal_to_standard_normal_mc": float(terminal_log_ratio.mean()),
            "mc_standard_error": float(terminal_log_ratio.std(ddof=1)/math.sqrt(len(terminal_log_ratio))),
            "analytic_kl_upper_bound_by_component_convexity": terminal_prior_bound,
            "note": "This measures replacing the exact OU terminal mixture by independent N(0,I). It is not a learned-model error bound."},
        "training_reference": sample_metrics(training[:64], training, heldout),
        "heldout_reference": sample_metrics(heldout[:64], training, heldout),
        "paired_integration_refinement": {
            "steps": args.steps, "fine_steps": 2*args.steps,
            "rmse_raw_pixels": float(np.sqrt(np.mean((refined-generated)**2))),
            "max_abs_raw_pixel_difference": float(np.max(np.abs(refined-generated)))},
        "independent_forward_validation": {
            "samples": 64, "residual_per_coordinate_mean": float(forward_residual.mean()),
            "residual_raw_dimension_sum_mean": float(mixture.dim*forward_residual.mean()),
            "terminal_kl_signed_monte_carlo_estimate": float(np.mean(forward_log_density-oracle_log_density)),
            "kl_estimate_note": "Uses true finite-mixture OU density after training only; finite-sample and RK4 error remain."},
        "evaluation_seconds": time.monotonic()-evaluation_start,
        "limitations": [
            "Initial distribution is 512 training images plus known isotropic Gaussian noise, not an unknown natural-image law.",
            "The 32-rank model is a constrained high-dimensional extension, outside the periodic low-dimensional acceptance scope.",
            "Independent normal-noise samples are separate from oracle-terminal diagnostic samples.",
            "Nearest-neighbor distances, diversity, and contact sheets are not FID and do not prove novelty or semantic quality.",
            "Full train and heldout neighbor sets have unequal sizes; a separate equal-count comparison is also reported.",
            "Pixel clipping is used only for display; raw arrays and metrics retain all out-of-range values.",
            "No high-dimensional error certificate or successful complex-image generation is asserted.",
            "Loss is normalized by image dimension; the raw dimension-summed residual is also reported.",
        ],
    }
    np.savez_compressed(out/"raw_samples.npz", independent_terminal_noise=np.asarray(noise),
                        generated_coarse=generated, generated_refined=refined,
                        oracle_terminal=np.asarray(oracle_terminal), oracle_generated=oracle_generated,
                        training_centers=training, heldout_images=heldout,
                        forward_initial=np.asarray(forward_initial), forward_final=forward_x)
    save_sheet(out/"training_reference.png", training[:64], "Training centers (not generated)")
    if metrics.get("finite"):
        save_sheet(out/"independent_generated.png", refined, "Independent standard Gaussian -> learned inverse flow")
        neighbors = np.stack((refined[:16], training[np.asarray(metrics["train_nearest_indices"][:16])],
                              heldout[np.asarray(metrics["heldout_nearest_indices"][:16])]), axis=1)
        save_sheet(out/"nearest_neighbors.png", neighbors.reshape(-1, IMAGE_DIM),
                   "Rows: generated | nearest training center | nearest heldout image", columns=3)
    if np.isfinite(oracle_generated).all():
        save_sheet(out/"oracle_terminal_diagnostic.png", oracle_generated, "ORACLE terminal mixture (diagnostic only)")
    write_json(out/"report.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    values = [h["loss_per_coordinate"] for h in progress["history"]]
    ax.semilogy(np.arange(1, len(values)+1), np.maximum(values, 1e-15))
    ax.set(xlabel="Update", ylabel="Self-consistency loss per coordinate", title="Training loss (not image quality)")
    fig.tight_layout()
    fig.savefig(out/"training_curve.png", dpi=140)
    plt.close(fig)
    summary = ("# CIFAR-10 self-consistency feasibility experiment\n\n"
               f"Status: **{progress['status']}**. Accepted updates: {progress['updates_accepted']}.\n\n"
               "This is a high-dimensional engineering stress test on an explicit noisy empirical mixture. "
               "It is outside the validated periodic 1–2D problem class. No successful natural-image "
               "generation or high-dimensional error certificate is claimed.\n\n"
               "- `independent_generated.png`: learned inverse flow from independent standard Gaussian noise.\n"
               "- `nearest_neighbors.png`: generated / nearest training / nearest heldout triples.\n"
               "- `oracle_terminal_diagnostic.png`: a separately labeled oracle-terminal diagnostic.\n"
               "- `raw_samples.npz`: unclipped samples, reference images, and paired integration outputs.\n"
               "- `report.json`: residual, nearest-neighbor, diversity, saturation, and timing evidence.\n\n"
               "The integrator evolves x=m(t)+scale(t)*y, z=scale(t)*score, and "
               "eta=logrho+d*log(scale); the known affine OU base is exact. Direct physical-coordinate RK4 "
               "had a false positive residual even for an exact single Gaussian (sigma=.2, T=3, "
               "128 steps: .00431 per coordinate; 256 steps: .000243). The coordinate change removes "
               "this known-base artifact, while nonlinear corrections still require refinement checks.\n\n"
               "A visually recognizable output can still reproduce a training center. The neighbor panels "
               "and heldout comparisons must be inspected before making a generative-quality claim.\n")
    (out/"README.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"out": str(out.resolve()), "status": progress["status"],
                      "independent_gaussian_metrics": _json_safe(metrics)}, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download and verify the official CIFAR-10 binary archive")
    download.add_argument("--output", required=True)
    run = sub.add_parser("run", help="Run a bounded self-consistency image experiment (no pretrained model)")
    run.add_argument("--archive", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--dataset-seed", type=int, default=0)
    run.add_argument("--rank", type=int, default=32)
    run.add_argument("--batch", type=int, default=16)
    run.add_argument("--updates", type=int, default=500)
    run.add_argument("--steps", type=int, default=128)
    run.add_argument("--T", type=float, default=3.)
    run.add_argument("--sigma", type=float, default=.2)
    run.add_argument("--base", choices=("component", "global"), default="component")
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--max-seconds", type=float, default=1800.)
    args = parser.parse_args(argv)
    if args.command == "download":
        download_archive(args.output)
        print(json.dumps({"archive": str(Path(args.output).resolve()), "verified_md5": CIFAR_MD5}))
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
