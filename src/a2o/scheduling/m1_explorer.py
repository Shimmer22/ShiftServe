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
class JointDecision:
    path: str  # "cc" | "hc_full" | "hc_compact"
    route: str  # "cpu" | "htp"
    kv_admit: bool
    keep_tokens: int
    h_active: int
    k_floor: int
    reason: str
    explored: bool
    cc_mean_ms: float
    cc_std_ms: float
    cc_lcb_ms: float
    hc_full_mean_ms: float
    hc_full_std_ms: float
    hc_full_lcb_ms: float
    hc_compact_mean_ms: float | None
    hc_compact_std_ms: float | None
    hc_compact_lcb_ms: float | None
    handoff_saving_ms: float = 0.0
    decode_saving_ms: float = 0.0
    prior_tax_ms: float = 0.0
    evict_ms: float = 0.0


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
        self.decode_per_tok = BayesianLinearComponent(
            dim=2, prior_mean=[40.0, 0.008], prior_var=[2500.0, 0.0004], obs_noise_std=10.0
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

    @staticmethod
    def _eval_linear_term(term: dict, x: float) -> float:
        return max(0.0, float(term.get("intercept_ms", 0.0)) + float(term.get("slope_ms_per_token", 0.0)) * x)

    @staticmethod
    def _pad_align_up(n: int, align: int) -> int:
        if align <= 1:
            return n
        return ((n + align - 1) // align) * align

    def _predict_decode(self, h_active: int, expected_d: int, kv_model: dict | None) -> tuple[float, float]:
        if expected_d <= 0:
            return 0.0, 0.0
        if self.decode_per_tok.n_obs >= 2 or kv_model is None or "decode_per_tok" not in kv_model:
            mu_tok, sd_tok = self.decode_per_tok.predict([1.0, float(h_active)])
            return float(expected_d) * mu_tok, float(expected_d) * sd_tok
        return float(expected_d) * self._eval_linear_term(kv_model["decode_per_tok"], float(h_active)), 0.0

    def select_joint_route(
        self,
        h: int,
        p: int,
        expected_d: int,
        kv_model: dict | None = None,
        htp_ready: bool = True,
        kv_enabled: bool = True,
        h_logical: int | None = None,
        pinned_tokens: int = 0,
        sink_tokens: int = 4,
        recent_tokens: int = 32,
    ) -> JointDecision:
        """Three-way joint decision over CC, HC-Full, and HC-Compact using bounded LCB."""
        total = h + p
        r_mu, r_sd = self.cpu_restore.predict([1.0, float(h)])
        cp_mu, cp_sd = self.cpu_prefill.predict(self._prefill_feat(h, p))
        dec_full_mu, dec_full_sd = self._predict_decode(total, expected_d, kv_model)

        cc_mu = r_mu + cp_mu + dec_full_mu
        cc_sd = math.sqrt(r_sd * r_sd + cp_sd * cp_sd + dec_full_sd * dec_full_sd)
        cc_lcb = cc_mu - self.beta * cc_sd

        hp_mu, hp_sd = self.htp_prefill.predict(self._prefill_feat(h, p))
        ho_full_mu, ho_full_sd = self.htp_to_cpu.predict([1.0, float(total)])
        hc_full_mu = hp_mu + ho_full_mu + dec_full_mu
        hc_full_sd = math.sqrt(hp_sd * hp_sd + ho_full_sd * ho_full_sd + dec_full_sd * dec_full_sd)
        hc_full_lcb = hc_full_mu - self.beta * hc_full_sd

        # Evaluate HC-Compact candidate by explicitly plugging H_active = k into Handoff and Decode.
        logical_h = max(h, h_logical) if (h_logical is not None and h_logical >= 0) else h
        logical_total = logical_h + p
        k_min = int(kv_model.get("quality_k_min", 1536)) if kv_model else 1536
        ratio_min = float(kv_model.get("quality_keep_ratio_min", 0.25)) if kv_model else 0.25
        align = int(kv_model.get("align_pad", 256)) if kv_model else 256
        min_evict = int(kv_model.get("min_evict_savings_tokens", 512)) if kv_model else 512
        min_gain = float(kv_model.get("min_gain_ms", 100.0)) if kv_model else 100.0
        eff_sink = max(sink_tokens, pinned_tokens)
        k_floor = max(k_min, eff_sink + recent_tokens, self._pad_align_up(int(math.ceil(ratio_min * logical_total)), align))

        hc_compact_eligible = False
        best_k = total
        hc_cmp_mu: float | None = None
        hc_cmp_sd: float | None = None
        hc_cmp_lcb: float | None = None
        best_hnd_saving = 0.0
        best_dec_saving = 0.0
        tax_ms = 0.0
        evict_ms = 0.0

        if kv_enabled and kv_model is not None and total > k_floor + min_evict:
            base_cands = list(kv_model.get("candidates", [1536, 2048, 2560, 3072, 4096]))
            cands = sorted({c for c in base_cands + [k_floor + off for off in (0, 512, 1024, 2048)] if c >= k_floor and c + min_evict <= total})
            if cands:
                tax_ms = self._eval_linear_term(kv_model["htp_prior_tax"], float(p))
                evict_ms = self._eval_linear_term(kv_model["evict"], float(total))
                for k in cands:
                    ho_k_mu, ho_k_sd = self.htp_to_cpu.predict([1.0, float(k)])
                    dec_k_mu, dec_k_sd = self._predict_decode(k, expected_d, kv_model)
                    cand_mu = hp_mu + tax_ms + evict_ms + ho_k_mu + dec_k_mu
                    cand_sd = math.sqrt(hp_sd * hp_sd + ho_k_sd * ho_k_sd + dec_k_sd * dec_k_sd)
                    if hc_cmp_mu is None or cand_mu < hc_cmp_mu:
                        hc_cmp_mu = cand_mu
                        hc_cmp_sd = cand_sd
                        hc_cmp_lcb = cand_mu - self.beta * cand_sd
                        best_k = k
                        best_hnd_saving = ho_full_mu - ho_k_mu
                        best_dec_saving = dec_full_mu - dec_k_mu
                if hc_cmp_mu is not None and (hc_full_mu - hc_cmp_mu) > min_gain:
                    hc_compact_eligible = True

        if p < self.routing_p_min:
            return JointDecision(
                "cc", "cpu", False, total, total, k_floor, "below_p_min", False,
                cc_mu, cc_sd, cc_lcb, hc_full_mu, hc_full_sd, hc_full_lcb,
                hc_cmp_mu, hc_cmp_sd, hc_cmp_lcb, best_hnd_saving, best_dec_saving, tax_ms, evict_ms,
            )
        if not htp_ready:
            return JointDecision(
                "cc", "cpu", False, total, total, k_floor, "htp_not_ready", False,
                cc_mu, cc_sd, cc_lcb, hc_full_mu, hc_full_sd, hc_full_lcb,
                hc_cmp_mu, hc_cmp_sd, hc_cmp_lcb, best_hnd_saving, best_dec_saving, tax_ms, evict_ms,
            )

        candidates = [
            ("cc", "cpu", False, total, cc_mu, cc_lcb),
            ("hc_full", "htp", False, total, hc_full_mu, hc_full_lcb),
        ]
        if hc_compact_eligible and hc_cmp_mu is not None and hc_cmp_lcb is not None:
            candidates.append(("hc_compact", "htp", True, best_k, hc_cmp_mu, hc_cmp_lcb))

        greedy = min(candidates, key=lambda c: c[4])
        if not self.exploration_enabled:
            return JointDecision(
                greedy[0], greedy[1], greedy[2], greedy[3], greedy[3], k_floor, "m1_greedy_mean", False,
                cc_mu, cc_sd, cc_lcb, hc_full_mu, hc_full_sd, hc_full_lcb,
                hc_cmp_mu, hc_cmp_sd, hc_cmp_lcb, best_hnd_saving, best_dec_saving, tax_ms, evict_ms,
            )

        lcb_pick = min(candidates, key=lambda c: c[5])
        explored = lcb_pick[0] != greedy[0]
        chosen = lcb_pick
        reason = "m1_lcb_explore" if explored else "m1_lcb_agree"
        if explored:
            mean_gap = lcb_pick[4] - greedy[4]
            enough_data = (
                self.cpu_prefill.n_obs >= self.min_obs_before_guard
                and self.htp_prefill.n_obs >= self.min_obs_before_guard
            )
            if enough_data and mean_gap > self.max_explore_penalty_ms:
                chosen = greedy
                explored = False
                reason = "m1_safe_guard_greedy"

        return JointDecision(
            chosen[0], chosen[1], chosen[2], chosen[3], chosen[3], k_floor, reason, explored,
            cc_mu, cc_sd, cc_lcb, hc_full_mu, hc_full_sd, hc_full_lcb,
            hc_cmp_mu, hc_cmp_sd, hc_cmp_lcb, best_hnd_saving, best_dec_saving, tax_ms, evict_ms,
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
        decode_ms: float | None = None,
        predicted_n: int | None = None,
        h_active: int | None = None,
        collect_prior: bool = False,
        force: bool = False,
    ) -> None:
        if not self.online_update_enabled and not force:
            return
        eff_active = h_active if h_active is not None else (h + p)
        if prepare_ms is not None and prepare_ms > 0.0:
            self.cpu_to_htp.update([1.0, float(h)], prepare_ms)
        if route == "cpu":
            if restore_ms is not None and restore_ms >= 0.0:
                self.cpu_restore.update([1.0, float(h)], restore_ms)
            if prefill_ms is not None and prefill_ms >= 0.0 and p > 0:
                self.cpu_prefill.update(self._prefill_feat(h, p), prefill_ms)
        elif route == "htp":
            if prefill_ms is not None and prefill_ms >= 0.0 and p > 0 and not collect_prior:
                self.htp_prefill.update(self._prefill_feat(h, p), prefill_ms)
            if handoff_ms is not None and handoff_ms >= 0.0:
                self.htp_to_cpu.update([1.0, float(eff_active)], handoff_ms)
        if decode_ms is not None and decode_ms >= 0.0 and predicted_n is not None and predicted_n > 0:
            self.decode_per_tok.update([1.0, float(eff_active)], decode_ms / float(predicted_n))
        self.history.append({
            "h": h, "p": p, "h_active": eff_active, "route": route,
            "prefill_ms": prefill_ms, "handoff_ms": handoff_ms, "decode_ms": decode_ms,
        })

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
