import numpy as np
import pandas as pd

DEFAULT_REGIME_PARAMS = {
    0: {'drift': 0.18,  'vol_mult': 0.7,  'name': 'Bull'},
    1: {'drift': -0.10, 'vol_mult': 1.1,  'name': 'Bear'},
    2: {'drift': -0.25, 'vol_mult': 2.0,  'name': 'Crisis'},
    3: {'drift': 0.02,  'vol_mult': 1.5,  'name': 'Choppy'},
}

DEFAULT_TRANSITION_MATRIX = np.array([
    [0.95, 0.03, 0.01, 0.01],  # from Bull
    [0.05, 0.90, 0.03, 0.02],  # from Bear
    [0.15, 0.25, 0.55, 0.05],  # from Crisis
    [0.20, 0.15, 0.05, 0.60],  # from Choppy
])

# How long each regime lasts once it starts (in trading days).
# Supported specs:
#   {'dist': 'lognormal', 'median': m, 'sigma': s}   mean = m * exp(s**2 / 2)
#   {'dist': 'pareto',    'xmin': x, 'alpha': a}      mean = a * x / (a - 1); keep a > 1.5
#   {'dist': 'uniform',   'low': lo, 'high': hi}      whole days, inclusive
#   {'dist': 'geometric', 'p_stay': p}                identical to the old Markov chain
DEFAULT_REGIME_DURATIONS = {
    0: {'dist': 'lognormal', 'median': 80, 'sigma': 0.9},  # Bull:   mean ~120 days, long right tail
    1: {'dist': 'lognormal', 'median': 50, 'sigma': 0.8},  # Bear:   mean ~69 days
    2: {'dist': 'uniform',   'low': 5,     'high': 30},    # Crisis: 5-30 days
    3: {'dist': 'lognormal', 'median': 45, 'sigma': 0.6},  # Choppy: mean ~54 days
}
# Pareto alternative for heavier trend tails, e.g.
#   0: {'dist': 'pareto', 'xmin': 30, 'alpha': 1.8}   # mean ~68 days, ~2% of regimes last > 1 year

def simulate_gbm(mu,sigma,S0,T,seed) -> np.ndarray:
    """
    Simulate a single price path using Geometric Brownian Motion.

    Parameters
    ----------
    mu    : Annual drift (e.g. 0.10 for 10%)
    sigma : Annual volatility (e.g. 0.187 for 18.7%)
    S0    : Starting price
    T     : Number of trading days
    seed  : RNG seed

    Returns
    -------
    np.ndarray of length T (prices).
    """

    rng = np.random.default_rng(seed)
    dt = 1 / 252

    daily_mu = mu * dt
    daily_sigma = sigma * np.sqrt(dt)

    log_returns = daily_mu - 0.5 * sigma ** 2 * dt + daily_sigma * rng.standard_normal(T)
    prices = S0 * np.exp(np.cumsum(np.insert(log_returns, 0, 0)))

    return prices[1:]

def simulate_garch(mu,omega,alpha,beta,S0,T,seed) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate price path with GARCH(1,1) volatility dynamics.

    Parameters
    ----------
    mu    : Annual drift
    omega : Long-run variance floor (e.g. 2e-6)
    alpha : ARCH term — sensitivity to recent shocks (e.g. 0.09)
    beta  : GARCH term — variance persistence (e.g. 0.89)
    S0    : Starting price
    T     : Number of trading days
    seed  : RNG seed

    Returns
    -------
    (prices, annualised_conditional_volatilities), each length T.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252

    long_run_var = omega / (1 - alpha - beta)
    variances = np.zeros(T)
    returns = np.zeros(T)
    variances[0] = long_run_var

    for t in range(T):
        z = rng.standard_normal()
        returns[t] = mu * dt + np.sqrt(variances[t] * dt) * z

        epsilon_sq = (returns[t] - mu * dt) ** 2
        if t + 1 < T:
            variances[t + 1] = omega + alpha * epsilon_sq / dt + beta * variances[t]

    prices = S0 * np.exp(np.cumsum(np.insert(returns, 0, 0)))
    vols = np.sqrt(variances * 252)

    return prices[1:], vols

def draw_fat_tail_shocks(T, nu, seed) -> np.ndarray:
    """
    Draw T noise samples from a unit-variance Student's t distribution.

    Parameters
    ----------
    T    : Number of samples
    nu   : Degrees of freedom.
           nu=4.51 → excess kurtosis ~11.7 (matches SPY)
           nu=6.0  → moderate fat tails
           nu=30   → close to Gaussian
    seed : RNG seed

    Returns
    -------
    np.ndarray of unit-variance t-distributed shocks, length T.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_t(nu, size=T)
    scale = np.sqrt((nu - 2) / nu)  # normalise to unit variance
    return raw * scale

def simulate_ou_process(theta,mu,sigma,X0,T,seed) -> np.ndarray:
    """
    Simulate an Ornstein-Uhlenbeck mean-reverting process.

    Parameters
    ----------
    theta : Mean-reversion speed (higher → faster reversion)
    mu    : Long-run mean
    sigma : Diffusion coefficient
    X0    : Starting value
    T     : Number of steps
    seed  : RNG seed

    Returns
    -------
    np.ndarray of length T+1 (includes X0).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252

    path = np.zeros(T + 1)
    path[0] = X0

    for t in range(T):
        drift = theta * (mu - path[t]) * dt
        diffusion = sigma * np.sqrt(dt) * rng.standard_normal()
        path[t + 1] = path[t] + drift + diffusion

    return path

def simulate_regime_sequence(T,initial_regime=0,transition_matrix= None,seed=42) -> np.ndarray:
    """
    Simulate a regime label sequence using a Markov chain.

    Parameters
    ----------
    T                 : Number of days
    initial_regime    : Starting regime index (0=Bull, 1=Bear, 2=Crisis, 3=Choppy)
    transition_matrix : (4×4) row-stochastic matrix. Uses DEFAULT_TRANSITION_MATRIX if None.
    seed              : RNG seed

    Returns
    -------
    np.ndarray of int regime indices, length T.
    """
    if transition_matrix is None:
        transition_matrix = DEFAULT_TRANSITION_MATRIX

    rng = np.random.default_rng(seed)
    regimes = np.zeros(T, dtype=int)
    regimes[0] = initial_regime

    for t in range(1, T):
        regimes[t] = rng.choice(len(transition_matrix), p=transition_matrix[regimes[t - 1]])

    return regimes

def draw_regime_duration(rng, spec, max_duration=1008):
    """Draw one regime duration in whole days (at least 1, at most max_duration)."""
    dist = spec['dist']
    if dist == 'lognormal':
        d = rng.lognormal(mean=np.log(spec['median']), sigma=spec['sigma'])
    elif dist == 'pareto':
        d = spec['xmin'] * (1.0 + rng.pareto(spec['alpha']))  # numpy's pareto is Lomax; +1 gives classic Pareto
    elif dist == 'uniform':
        d = rng.integers(spec['low'], spec['high'] + 1)
    elif dist == 'geometric':
        d = rng.geometric(1.0 - spec['p_stay'])
    else:
        raise ValueError(f"Unknown duration distribution: {dist}")
    return int(min(max(round(d), 1), max_duration))

def jump_matrix_from_transition(transition_matrix):
    """
    Turn a Markov transition matrix into a jump matrix: zero the diagonal
    (staying is now handled by the duration) and rescale each row to sum to 1.
    """
    J = np.array(transition_matrix, dtype=float).copy()
    np.fill_diagonal(J, 0.0)
    row_sums = J.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Every regime needs a non-zero probability of moving to another regime")
    return J / row_sums

def simulate_semi_markov_regimes(T, initial_regime, transition_matrix, regime_durations,
                                 rng, max_duration=756):
    """
    Regime sequence where each regime lasts for a duration drawn from its own
    distribution, then jumps to a different regime via the jump matrix.
    """
    J = jump_matrix_from_transition(transition_matrix)
    n = len(J)
    regimes = np.empty(T, dtype=int)

    t, r = 0, initial_regime
    while t < T:
        d = draw_regime_duration(rng, regime_durations[r], max_duration)
        regimes[t:t + d] = r          # slicing past T is safe; the last regime is cut off
        t += d
        r = rng.choice(n, p=J[r])
    return regimes

def regime_duration_summary(regime_paths, regime_params):
    """
    Realised regime durations, pooled over one or more paths.
    Pass a single regime array or a list of them (e.g. from many seeds).
    The final regime of each path is excluded because it is cut off at T.
    """
    if isinstance(regime_paths, np.ndarray) and regime_paths.ndim == 1:
        regime_paths = [regime_paths]

    lengths_all, labels_all, counts = [], [], np.zeros(len(regime_params))
    total_days = 0
    for regimes in regime_paths:
        regimes = np.asarray(regimes)
        change = np.flatnonzero(np.diff(regimes)) + 1
        starts = np.r_[0, change]
        lengths = np.diff(np.r_[starts, len(regimes)])
        labels = regimes[starts]
        lengths_all.append(lengths[:-1])
        labels_all.append(labels[:-1])
        counts += np.bincount(regimes, minlength=len(regime_params))
        total_days += len(regimes)

    lengths_all = np.concatenate(lengths_all)
    labels_all = np.concatenate(labels_all)

    rows = []
    for r, p in regime_params.items():
        L = lengths_all[labels_all == r]
        has = len(L) > 0
        rows.append({
            'regime':      p['name'],
            'episodes':    len(L),
            'mean_days':   L.mean() if has else np.nan,
            'median_days': np.median(L) if has else np.nan,
            'p95_days':    np.percentile(L, 95) if has else np.nan,
            'max_days':    L.max() if has else np.nan,
            'time_share':  counts[r] / total_days,
        })
    return pd.DataFrame(rows).set_index('regime')


# =============================================================================
# Market generator
# =============================================================================

class SyntheticMarketGenerator:
    """
    Full stochastic market generator with OHLC bars.

    Combines GBM + GARCH(1,1) + Student-t fat tails +
    mean reversion + regime switching (semi-Markov by default).

    Each day's return is split into an overnight gap (close -> open) and an
    intraday move (open -> close). The intraday path is a Brownian bridge
    pinned to the open and close, and its max/min give the high/low.

    Usage
    -----
    gen = SyntheticMarketGenerator()
    df  = gen.generate(T=252, S0=450.0, seed=42)
    df  = add_atr(df, period=14)
    """

    def __init__(
        self,
        # GBM baseline
        mu: float = 0.12,
        sigma: float = 0.15,
        # GARCH (daily variance units)
        garch_omega: float = 2e-6,
        garch_alpha: float = 0.09,
        garch_beta: float = 0.89,
        # Fat tails (Student's t degrees of freedom)
        t_nu: float = 4.51,
        # Mean reversion speed
        mr_theta: float = 0.08,
        # Regime switching
        use_regimes: bool = True,
        transition_matrix: np.ndarray = None,
        regime_params: dict = None,
        # NEW: regime durations
        duration_model: str = 'semi_markov',   # 'semi_markov' or 'markov' (the old day-by-day chain)
        regime_durations: dict = None,
        max_regime_duration: int = 252*4,        # cap any single regime at ~4 years
        # OHLC construction
        overnight_var_share: float = 0.25,     # share of daily variance in the overnight gap
        n_substeps: int = 78,                  # intraday steps per bar (78 = 5-min bars in a 6.5h session)
    ):
        self.mu = mu
        self.sigma = sigma
        self.garch_omega = garch_omega
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta
        self.t_nu = t_nu
        self.mr_theta = mr_theta
        self.use_regimes = use_regimes
        self.transition_matrix = transition_matrix if transition_matrix is not None \
            else DEFAULT_TRANSITION_MATRIX
        self.regime_params = regime_params if regime_params is not None \
            else DEFAULT_REGIME_PARAMS

        # NEW
        if duration_model not in ('semi_markov', 'markov'):
            raise ValueError("duration_model must be 'semi_markov' or 'markov'")
        self.duration_model = duration_model
        self.regime_durations = regime_durations if regime_durations is not None \
            else DEFAULT_REGIME_DURATIONS
        self.max_regime_duration = int(max_regime_duration)

        if not 0.0 <= overnight_var_share < 1.0:
            raise ValueError("overnight_var_share must be in [0, 1)")
        self.overnight_var_share = overnight_var_share
        self.n_substeps = int(n_substeps)

    def generate(
        self,
        T: int = 252,
        S0: float = 100.0,
        initial_regime: int = 0,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate T days of synthetic OHLC data.

        Returns
        -------
        pd.DataFrame with columns:
            open, high, low, close, price (= close), daily_return, log_return,
            overnight_return, intraday_return, regime, regime_name,
            conditional_vol (GARCH only, annualised),
            effective_vol (GARCH x regime multiplier, annualised), garch_var
        """
        rng = np.random.default_rng(seed)
        dt = 1 / 252
        w = self.overnight_var_share

        # Step 1: regime sequence  # NEW: semi-Markov option
        if not self.use_regimes:
            regimes = np.zeros(T, dtype=int)
            regimes[0] = initial_regime
        elif self.duration_model == 'semi_markov':
            regimes = simulate_semi_markov_regimes(
                T, initial_regime, self.transition_matrix, self.regime_durations,
                rng, self.max_regime_duration,
            )
        else:  # original day-by-day Markov chain
            regimes = np.zeros(T, dtype=int)
            regimes[0] = initial_regime
            for t in range(1, T):
                regimes[t] = rng.choice(4, p=self.transition_matrix[regimes[t - 1]])

        # Step 2: GARCH initialisation (daily variance)
        long_run_var = self.garch_omega / (1 - self.garch_alpha - self.garch_beta)
        garch_var = np.zeros(T)
        garch_var[0] = long_run_var

        # Step 3: fat-tail shocks, separate for the overnight gap and the intraday move
        t_scale = np.sqrt((self.t_nu - 2) / self.t_nu)  # normalise to unit variance
        overnight_shocks = rng.standard_t(self.t_nu, size=T) * t_scale
        intraday_shocks = rng.standard_t(self.t_nu, size=T) * t_scale

        # Step 4: build returns
        log_returns = np.zeros(T)
        overnight_returns = np.zeros(T)
        intraday_returns = np.zeros(T)
        daily_vols = np.zeros(T)
        prev_log_return = 0.0

        for t in range(T):
            regime = regimes[t]
            regime_drift = self.regime_params[regime]['drift'] if self.use_regimes else self.mu
            regime_vol_mult = self.regime_params[regime]['vol_mult'] if self.use_regimes else 1.0

            daily_vol = np.sqrt(garch_var[t]) * regime_vol_mult
            daily_vol = max(daily_vol, 0.001)
            daily_vols[t] = daily_vol

            mr_adj = -self.mr_theta * prev_log_return * dt
            drift_term = regime_drift * dt - 0.5 * (daily_vol ** 2)

            # Split the day's shock into overnight gap + intraday move.
            # Variances add to daily_vol**2, so the close-to-close series keeps the same volatility.
            overnight_returns[t] = np.sqrt(w) * daily_vol * overnight_shocks[t]
            intraday_returns[t] = drift_term + np.sqrt(1 - w) * daily_vol * intraday_shocks[t] + mr_adj
            log_returns[t] = overnight_returns[t] + intraday_returns[t]
            prev_log_return = log_returns[t]

            if t + 1 < T:
                epsilon_sq = ((log_returns[t] - regime_drift * dt) ** 2) / (regime_vol_mult ** 2)
                garch_var[t + 1] = (
                    self.garch_omega
                    + self.garch_alpha * epsilon_sq
                    + self.garch_beta * garch_var[t]
                )
                garch_var[t + 1] = max(garch_var[t + 1], 1e-8)

        if not np.all(np.isfinite(log_returns)):
            raise FloatingPointError(f"Non-finite returns generated (seed={seed})")

        # Step 5: close and open prices
        close = S0 * np.exp(np.cumsum(log_returns))
        prev_close = np.r_[S0, close[:-1]]
        open_ = prev_close * np.exp(overnight_returns)

        # Step 6: high and low from an intraday Brownian bridge
        intraday_vols = np.sqrt(1 - w) * daily_vols
        high, low = self._intraday_high_low(open_, intraday_returns, intraday_vols, rng)

        daily_returns = np.exp(log_returns) - 1

        df = pd.DataFrame({
            'open':             open_,
            'high':             high,
            'low':              low,
            'close':            close,
            'price':            close,  # kept so existing code still works
            'daily_return':     daily_returns,
            'log_return':       log_returns,
            'overnight_return': overnight_returns,
            'intraday_return':  intraday_returns,
            'regime':           regimes,
            'regime_name':      [self.regime_params[r]['name'] for r in regimes],
            'conditional_vol':  np.sqrt(garch_var * 252),
            'effective_vol':    daily_vols * np.sqrt(252),
            'garch_var':        garch_var,
        })
        df.index.name = 'day'

        return df

    def _intraday_high_low(self, open_, intraday_returns, intraday_vols, rng):
        """
        Build each day's intraday log-price path as a Brownian bridge that starts
        at the open and ends exactly at the close, then take its max and min.
        """
        T = len(open_)
        N = self.n_substeps

        # Random walk of N steps, scaled so its total variance equals the day's intraday variance
        steps = rng.standard_normal((T, N)) * (intraday_vols / np.sqrt(N))[:, None]
        walk = np.zeros((T, N + 1))
        walk[:, 1:] = np.cumsum(steps, axis=1)

        # Pin the walk: remove its own endpoint linearly, add the required open->close move
        frac = np.arange(N + 1) / N
        bridge = walk - frac[None, :] * walk[:, [-1]] + frac[None, :] * intraday_returns[:, None]

        # bridge[:, 0] = 0 (open) and bridge[:, -1] = intraday return (close), so both are included
        high = open_ * np.exp(bridge.max(axis=1))
        low = open_ * np.exp(bridge.min(axis=1))
        return high, low


