"""M1-Explorer: online Bayesian linear routing with safe LCB exploration.

Cold-starts from a handful of tiny universal probe shapes (e.g. H in {128, 1024},
P in {32, 128}), models H*P coupling in prefill latency, updates posterior
estimates incrementally as tasks finish, and uses bounded Lower Confidence Bound
(LCB) exploration so routing escapes early optimism bias and converges toward
the per-turn Oracle upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve small dense linear system Ax = b with partial pivoting."""
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("Singular matrix in BayesianLinearModel")
        aug[col], aug[piv] = aug[piv], aug[col]
        diag = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= diag
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _invert_matrix(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    cols = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        cols.append(_solve_linear(a, e))
    return [[cols[j][i] for j in range(n)] for i in range(n)]


class BayesianLinearComponent:
    """Online Bayesian linear regression with exponential forgetting."""

    def __init__(
        self,
        dim: int,
        prior_mean: list[float],
        prior_var: list[float],
        obs_noise_std: float,
        forgetting: float = 0.98,
    ) -> None:
        if len(prior_mean) != dim or len(prior_var) != dim:
            raise ValueError("prior dimension mismatch")
        self.dim = dim
        self.noise_var = max(1e-6, obs_noise_std * obs_noise_std)
        self.forgetting = forgetting
        self.precision = [
            [1.0 / max(1e-9, prior_var[i]) if i == j else 0.0 for j in range(dim)]
            for i in range(dim)
        ]
        self.info_vec = [prior_mean[i] / max(1e-9, prior_var[i]) for i in range(dim)]
        self.prior_precision = [row[:] for row in self.precision]
        self.prior_info = self.info_vec[:]
        self.n_obs = 0
        self._consecutive_outliers = 0
        self._last_outlier_sign = 0
        self._refresh()

    def _refresh(self) -> None:
        self.cov = _invert_matrix(self.precision)
        self.mean = _solve_linear(self.precision, self.info_vec)

    def update(self, x: list[float], y: float, weight: float = 1.0, robust: bool = True) -> bool:
        """Update posterior with dynamic outlier isolation vs sustained-shift acceptance.

        Returns True if the observation was trusted (full/Huber weight), False if isolated.
        """
        if len(x) != self.dim or not math.isfinite(y) or y < 0.0:
            return False
        eff_weight = weight
        trusted = True
        if robust and self.n_obs >= 3:
            mu, sd = self.predict(x)
            err = y - mu
            abs_err = abs(err)
            z = abs_err / max(sd, 150.0)
            rel_err = abs_err / max(mu, 300.0)
            sign = 1 if err >= 0.0 else -1
            if z > 3.0 and rel_err > 0.8:
                if sign == self._last_outlier_sign:
                    self._consecutive_outliers += 1
                else:
                    self._consecutive_outliers = 1
                    self._last_outlier_sign = sign
                # Isolate isolated spikes; trust if consecutive observations confirm a real regime shift
                if self._consecutive_outliers < 2:
                    return False
                eff_weight *= 0.5
            else:
                self._consecutive_outliers = 0
                self._last_outlier_sign = 0
                if z > 1.5:
                    eff_weight *= 1.5 / z

        lam = self.forgetting
        inv_s2 = eff_weight / self.noise_var
        for i in range(self.dim):
            self.info_vec[i] = lam * self.info_vec[i] + (1.0 - lam) * self.prior_info[i] + inv_s2 * x[i] * y
            for j in range(self.dim):
                self.precision[i][j] = (
                    lam * self.precision[i][j]
                    + (1.0 - lam) * self.prior_precision[i][j]
                    + inv_s2 * x[i] * x[j]
                )
        self.n_obs += 1
        self._refresh()
        return trusted

    def predict(self, x: list[float]) -> tuple[float, float]:
        """Return (clamped_mean_ms, predictive_std_ms)."""
        mu = sum(w * xi for w, xi in zip(self.mean, x))
        var = self.noise_var
        for i in range(self.dim):
            row_sum = sum(self.cov[i][j] * x[j] for j in range(self.dim))
            var += x[i] * row_sum
        return max(0.0, mu), math.sqrt(max(0.0, var))


@dataclass
class RouteDecision:
    route: str
    reason: str
    explored: bool
    cc_mean_ms: float
    cc_std_ms: float
    hc_mean_ms: float
    hc_std_ms: float
    cc_lcb_ms: float
    hc_lcb_ms: float


@dataclass
class M1Explorer:
    """Online adaptive router with H*P coupling and bounded LCB exploration."""

    routing_p_min: int = 32
    beta: float = 1.25
    max_explore_penalty_ms: float = 4000.0
    min_obs_before_guard: int = 6
    exploration_enabled: bool = True
    online_update_enabled: bool = True
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Features: [1, H] for restore/prepare; [1, P, (H/1000)*P] for prefill; [1, H+P] for handoff
        self.cpu_restore = BayesianLinearComponent(
            dim=2, prior_mean=[40.0, 0.09], prior_var=[2500.0, 0.02], obs_noise_std=80.0
        )
        self.cpu_to_htp = BayesianLinearComponent(
            dim=2, prior_mean=[40.0, 0.12], prior_var=[2500.0, 0.04], obs_noise_std=150.0
        )
        self.cpu_prefill = BayesianLinearComponent(
            dim=3, prior_mean=[100.0, 60.0, 8.0], prior_var=[250000.0, 900.0, 100.0], obs_noise_std=600.0
        )
        self.htp_prefill = BayesianLinearComponent(
            dim=3, prior_mean=[250.0, 30.0, 7.0], prior_var=[250000.0, 900.0, 100.0], obs_noise_std=500.0
        )
        self.htp_to_cpu = BayesianLinearComponent(
            dim=2, prior_mean=[50.0, 0.25], prior_var=[40000.0, 0.16], obs_noise_std=300.0
        )

    @staticmethod
    def _prefill_feat(h: int, p: int) -> list[float]:
        return [1.0, float(p), (float(h) / 1000.0) * float(p)]

    def predict_paths(self, h: int, p: int) -> dict[str, float]:
        r_mu, r_sd = self.cpu_restore.predict([1.0, float(h)])
        cp_mu, cp_sd = self.cpu_prefill.predict(self._prefill_feat(h, p))
        hp_mu, hp_sd = self.htp_prefill.predict(self._prefill_feat(h, p))
        ho_mu, ho_sd = self.htp_to_cpu.predict([1.0, float(h + p)])

        cc_mu = r_mu + cp_mu
        cc_sd = math.hypot(r_sd, cp_sd)
        hc_mu = hp_mu + ho_mu
        hc_sd = math.hypot(hp_sd, ho_sd)
        return {
            "cc_mean_ms": cc_mu,
            "cc_std_ms": cc_sd,
            "hc_mean_ms": hc_mu,
            "hc_std_ms": hc_sd,
        }

    def select_route(self, h: int, p: int, htp_ready: bool = True) -> RouteDecision:
        stats = self.predict_paths(h, p)
        cc_mu, cc_sd = stats["cc_mean_ms"], stats["cc_std_ms"]
        hc_mu, hc_sd = stats["hc_mean_ms"], stats["hc_std_ms"]

        if p < self.routing_p_min:
            return RouteDecision("cpu", "below_p_min", False, cc_mu, cc_sd, hc_mu, hc_sd, cc_mu, hc_mu)
        if not htp_ready:
            return RouteDecision("cpu", "htp_not_ready", False, cc_mu, cc_sd, hc_mu, hc_sd, cc_mu, hc_mu)

        greedy_route = "htp" if hc_mu < cc_mu else "cpu"
        if not self.exploration_enabled:
            return RouteDecision(
                greedy_route, "greedy_mean", False, cc_mu, cc_sd, hc_mu, hc_sd, cc_mu, hc_mu
            )

        # Safe LCB: subtract uncertainty bonus, but cap exploration when mean gap is large
        cc_lcb = cc_mu - self.beta * cc_sd
        hc_lcb = hc_mu - self.beta * hc_sd
        lcb_route = "htp" if hc_lcb < cc_lcb else "cpu"

        explored = lcb_route != greedy_route
        if explored:
            mean_gap = abs(hc_mu - cc_mu)
            enough_data = (
                self.cpu_prefill.n_obs >= self.min_obs_before_guard
                and self.htp_prefill.n_obs >= self.min_obs_before_guard
            )
            if enough_data and mean_gap > self.max_explore_penalty_ms:
                return RouteDecision(
                    greedy_route, "safe_guard_greedy", False, cc_mu, cc_sd, hc_mu, hc_sd, cc_lcb, hc_lcb
                )
            return RouteDecision(
                lcb_route, "lcb_explore", True, cc_mu, cc_sd, hc_mu, hc_sd, cc_lcb, hc_lcb
            )

        return RouteDecision(
            greedy_route, "lcb_agree", False, cc_mu, cc_sd, hc_mu, hc_sd, cc_lcb, hc_lcb
        )

    def observe(
        self,
        *,
        h: int,
        p: int,
        route: str,
        restore_ms: float | None = None,
        prepare_ms: float | None = None,
        prefill_ms: float | None = None,
        handoff_ms: float | None = None,
        force: bool = False,
    ) -> None:
        if not self.online_update_enabled and not force:
            return
        if prepare_ms is not None and prepare_ms > 0.0:
            self.cpu_to_htp.update([1.0, float(h)], prepare_ms)
        if route == "cpu":
            if restore_ms is not None and restore_ms >= 0.0:
                self.cpu_restore.update([1.0, float(h)], restore_ms)
            if prefill_ms is not None and prefill_ms >= 0.0 and p > 0:
                self.cpu_prefill.update(self._prefill_feat(h, p), prefill_ms)
        elif route == "htp":
            if prefill_ms is not None and prefill_ms >= 0.0 and p > 0:
                self.htp_prefill.update(self._prefill_feat(h, p), prefill_ms)
            if handoff_ms is not None and handoff_ms >= 0.0:
                self.htp_to_cpu.update([1.0, float(h + p)], handoff_ms)
        self.history.append({"h": h, "p": p, "route": route, "prefill_ms": prefill_ms, "handoff_ms": handoff_ms})

    def bootstrap_probes(self, probes: Iterable[dict]) -> None:
        """Initialize posterior from tiny universal cold-start probe records."""
        for row in probes:
            h = int(row["H"])
            p = int(row["P"])
            if "cc_restore_ms" in row:
                self.cpu_restore.update([1.0, float(h)], float(row["cc_restore_ms"]))
            if "hc_prepare_ms" in row:
                self.cpu_to_htp.update([1.0, float(h)], float(row["hc_prepare_ms"]))
            if "cpu_prefill_ms" in row:
                self.cpu_prefill.update(self._prefill_feat(h, p), float(row["cpu_prefill_ms"]))
            if "htp_prefill_ms" in row:
                self.htp_prefill.update(self._prefill_feat(h, p), float(row["htp_prefill_ms"]))
            if "htp_to_cpu_ms" in row:
                self.htp_to_cpu.update([1.0, float(h + p)], float(row["htp_to_cpu_ms"]))

    def export_linear_hp_v1(self, reference_h: int = 6000) -> dict:
        """Linearize the H*P-coupled model around reference_h for /a2o/costs."""
        h_k = float(reference_h) / 1000.0
        cp_w = self.cpu_prefill.mean
        hp_w = self.htp_prefill.mean
        model = {
            "kind": "linear_hp_v1",
            "routing_p_min": self.routing_p_min,
            "h_min": 128,
            "h_max": 12288,
            "p_min": 32,
            "p_max": 4096,
            "reference_h": reference_h,
            "formula": "CC=R(H)+C(P;H_ref); HC=N(P;H_ref)+M(H+P); online M1-Explorer linearization",
            "cpu_restore": {
                "intercept_ms": max(0.0, self.cpu_restore.mean[0]),
                "slope_ms_per_token": max(0.0, self.cpu_restore.mean[1]),
            },
            "cpu_to_htp": {
                "intercept_ms": max(0.0, self.cpu_to_htp.mean[0]),
                "slope_ms_per_token": max(0.0, self.cpu_to_htp.mean[1]),
            },
            "cpu_prefill": {
                "intercept_ms": max(0.0, cp_w[0]),
                "slope_ms_per_token": max(0.0, cp_w[1] + cp_w[2] * h_k),
            },
            "htp_prefill": {
                "intercept_ms": max(0.0, hp_w[0]),
                "slope_ms_per_token": max(0.0, hp_w[1] + hp_w[2] * h_k),
            },
            "htp_to_cpu": {
                "intercept_ms": max(0.0, self.htp_to_cpu.mean[0]),
                "slope_ms_per_token": max(0.0, self.htp_to_cpu.mean[1]),
            },
        }
        model["model_id"] = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
        return model
