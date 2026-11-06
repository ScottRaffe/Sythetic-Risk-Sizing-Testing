"""
Backtesting tools for the synthetic-market position-sizing project.

Strategy: moving-average crossover with a trailing stop
-------------------------------------------------------
Entry   Fast MA crosses above the slow MA at the close -> buy at the next open.
        Fast MA crosses below the slow MA at the close -> short at the next open
        (if allow_short). Only a fresh crossover opens a trade: after a stop-out,
        the strategy waits for the next cross.

Stops   'atr'      initial: entry -/+ initial_mult x ATR (ATR at the signal close)
                   trail:   highest high (lowest low) since entry -/+ trail_mult x ATR
        'percent'  initial: entry x (1 -/+ initial_pct)
                   trail:   highest high (lowest low) since entry x (1 -/+ trail_pct)
        'none'     exit on the opposite crossover only.
                   R is measured against a nominal initial_mult x ATR.
        Trailing stops are updated after each close, only ever tighten,
        and apply from the next day.

Fills   Open gaps through the stop          -> exit at the open  ('gap')
        Day's low/high touches the stop     -> exit at the stop  ('stop')
        Opposite crossover                  -> exit at next open ('signal'),
                                               then reverse if allowed

Sizing  Decided once at entry; the size is not changed during the trade.
        {'method': 'notional',   'fraction': 1.0}    position value = fraction x equity
        {'method': 'fixed_risk', 'risk_pct': 0.01}   shares = risk_pct x equity / stop distance
        {'method': 'vol_target', 'atr_pct': 0.0033}  shares = atr_pct x equity / ATR
        Optional keys for every method:
            'compound'        True   size off current equity (False = initial equity)
            'max_leverage'    2.0    cap position value at this multiple of equity
            'integer_shares'  False  round down to whole shares (0 shares skips the trade)
        fixed_risk with an ATR stop == vol_target with atr_pct = risk_pct / initial_mult.

The strategy only sees OHLC. Regime labels are recorded on each trade for analysis only.

Contents
--------
1. Indicators           add_atr, add_ma_signals
2. Position sizing      position_size
3. Backtest engine      backtest_ma_atr
4. Statistics           trade_stats, equity_stats, summarise, ruin_table
5. Running many paths   generate_paths, run_many, compare, compare_parallel
6. Stop calibration     mae_profile, split_paths, stop_sweep
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

TRADING_DAYS = 252
DEFAULT_EQUITY = 100_000.0
DEFAULT_SIZING = {'method': 'notional', 'fraction': 1.0}
DEFAULT_DD_THRESHOLDS = (0.2, 0.3, 0.5)


# =============================================================================
# 1. Indicators
# =============================================================================

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add true range and a Wilder-smoothed ATR column (atr_<period>)."""
    prev_close = df['close'].shift(1)
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)

    df['true_range'] = true_range
    df[f'atr_{period}'] = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


def add_ma_signals(df: pd.DataFrame, fast: int = 10, slow: int = 20, atr_period: int = 14) -> pd.DataFrame:
    """Add moving averages, ATR and crossover flags. Everything is known at each day's close."""
    out = add_atr(df.copy(), atr_period)
    out['atr'] = out[f'atr_{atr_period}']
    out['fast_ma'] = out['close'].rolling(fast).mean()
    out['slow_ma'] = out['close'].rolling(slow).mean()

    valid = out['fast_ma'].notna() & out['slow_ma'].notna() & out['atr'].notna()
    prev_valid = valid.shift(1, fill_value=False)
    above = out['fast_ma'] > out['slow_ma']
    prev_above = above.shift(1, fill_value=False)

    out['cross_up'] = valid & prev_valid & above & ~prev_above
    out['cross_down'] = valid & prev_valid & ~above & prev_above
    return out


# =============================================================================
# 2. Position sizing
# =============================================================================

def position_size(sizing, equity, initial_equity, entry_price, stop_distance, atr):
    """
    Number of shares for a new trade, and whether the leverage cap was hit.

    stop_distance : initial risk per share (defines 1R)
    atr           : ATR at the signal close
    """
    method = sizing.get('method', 'notional')
    base = equity if sizing.get('compound', True) else initial_equity

    if method == 'notional':
        qty = sizing.get('fraction', 1.0) * base / entry_price
    elif method == 'fixed_risk':
        qty = sizing['risk_pct'] * base / stop_distance
    elif method == 'vol_target':
        qty = sizing['atr_pct'] * base / atr
    else:
        raise ValueError(f"Unknown sizing method: {method}")

    max_qty = sizing.get('max_leverage', 1.0) * equity / entry_price
    capped = qty > max_qty
    qty = min(qty, max_qty)
    if sizing.get('integer_shares', False):
        qty = np.floor(qty)
    return qty, capped


# =============================================================================
# 3. Backtest engine
# =============================================================================

@dataclass
class _OpenTrade:
    """State of the trade currently held."""
    direction: int          # +1 long, -1 short
    entry_day: int
    entry_price: float      # fill price, including costs
    initial_risk: float     # stop distance at entry in price units (= 1R)
    qty: float
    stop: float             # current stop level (nan when stop_type='none')
    best_price: float       # most favourable price since entry (drives the trail and MFE)
    worst_price: float      # least favourable price since entry (drives MAE)
    entry_atr: float
    equity_at_entry: float
    capped: bool            # True if the leverage cap limited the size
    last_mark: float        # price the position was last marked at


def _stop_fill(direction, stop, open_, high, low):
    """Return (price, reason) if today's bar hits the stop, else None."""
    if direction == 1:
        if open_ <= stop:
            return open_, 'gap'
        if low <= stop:
            return stop, 'stop'
    else:
        if open_ >= stop:
            return open_, 'gap'
        if high >= stop:
            return stop, 'stop'
    return None


def _trade_record(tr, exit_day, exit_price, reason, regime):
    """Summary row for a closed trade."""
    d = tr.direction
    worst = min(tr.worst_price, exit_price) if d == 1 else max(tr.worst_price, exit_price)
    adverse = d * (tr.entry_price - worst)       # how far price went against the trade (>= 0)
    return {
        'direction':        'long' if d == 1 else 'short',
        'entry_day':        tr.entry_day,
        'exit_day':         exit_day,
        'bars_held':        exit_day - tr.entry_day,
        'entry_price':      tr.entry_price,
        'exit_price':       exit_price,
        'initial_risk_pct': tr.initial_risk / tr.entry_price,
        'return_pct':       d * (exit_price / tr.entry_price - 1),
        'R':                d * (exit_price - tr.entry_price) / tr.initial_risk,
        'mfe_R':            d * (tr.best_price - tr.entry_price) / tr.initial_risk,
        'mae_R':            adverse / tr.initial_risk,
        'mae_pct':          adverse / tr.entry_price,
        'mae_atr':          adverse / tr.entry_atr,
        'entry_atr':        tr.entry_atr,
        'qty':              tr.qty,
        'equity_at_entry':  tr.equity_at_entry,
        'notional_pct':     tr.qty * tr.entry_price / tr.equity_at_entry,
        'risk_pct_actual':  tr.qty * tr.initial_risk / tr.equity_at_entry,
        'capped':           tr.capped,
        'pnl_cash':         d * tr.qty * (exit_price - tr.entry_price),
        'exit_reason':      reason,
        'entry_regime':     regime[tr.entry_day] if regime is not None else None,
    }


def backtest_ma_atr(df, fast=10, slow=20, atr_period=14,
                    stop_type='atr', trail_mult=3.0, initial_mult=None,
                    trail_pct=0.05, initial_pct=None,
                    allow_short=True, exit_on_opposite_cross=True, cost_bps=0.0,
                    sizing=None, initial_equity=DEFAULT_EQUITY):
    """
    Backtest the MA crossover / trailing stop strategy with position sizing on one OHLC path.

    Returns a dict with:
        trades : one row per trade (R-multiples, size, cash P&L, MAE/MFE, exit reason, ...)
        daily  : daily cash P&L, equity, return, position and shares
        data   : the input data with indicators and the stop level in force each day
    """
    if stop_type not in ('atr', 'percent', 'none'):
        raise ValueError("stop_type must be 'atr', 'percent' or 'none'")
    if stop_type == 'none' and not exit_on_opposite_cross:
        raise ValueError("With stop_type='none', exit_on_opposite_cross must be True or trades never close")
    initial_mult = trail_mult if initial_mult is None else initial_mult
    initial_pct = trail_pct if initial_pct is None else initial_pct
    sizing = DEFAULT_SIZING if sizing is None else sizing
    cost = cost_bps / 1e4

    # --- Data -----------------------------------------------------------------
    d = add_ma_signals(df, fast, slow, atr_period)
    o, h, l, c = (d[col].to_numpy() for col in ('open', 'high', 'low', 'close'))
    atr = d['atr'].to_numpy()
    cross_up, cross_down = d['cross_up'].to_numpy(), d['cross_down'].to_numpy()
    regime = d['regime'].to_numpy() if 'regime' in d else None
    n = len(d)

    # --- Outputs --------------------------------------------------------------
    pnl = np.zeros(n)                   # daily cash P&L
    equity_curve = np.zeros(n)          # mark-to-market equity at each close
    position = np.zeros(n, dtype=int)
    shares = np.zeros(n)
    stop_line = np.full(n, np.nan)      # stop level in force during each day
    trades = []

    # --- State ----------------------------------------------------------------
    equity = initial_equity
    trade = None                        # _OpenTrade while in a position
    pending_entry = 0                   # +1 / -1: enter at the next open
    pending_exit = False                # exit at the next open

    def close_trade(t, fill, reason):
        nonlocal equity, trade
        cash = trade.direction * trade.qty * (fill - trade.last_mark)
        pnl[t] += cash
        equity += cash
        trades.append(_trade_record(trade, t, fill, reason, regime))
        trade = None

    for t in range(n):
        # 1. Orders placed at yesterday's close fill at today's open
        if pending_exit and trade is not None:
            close_trade(t, o[t] * (1 - trade.direction * cost), 'signal')
        pending_exit = False

        if pending_entry != 0 and trade is None and equity > 0:
            direction = pending_entry
            entry_price = o[t] * (1 + direction * cost)
            signal_atr = atr[t - 1]                         # ATR known at the signal close
            risk = initial_pct * o[t] if stop_type == 'percent' else initial_mult * signal_atr
            qty, capped = position_size(sizing, equity, initial_equity, entry_price, risk, signal_atr)
            if qty > 0:
                trade = _OpenTrade(
                    direction=direction, entry_day=t, entry_price=entry_price,
                    initial_risk=risk, qty=qty,
                    stop=o[t] - direction * risk if stop_type != 'none' else np.nan,
                    best_price=o[t], worst_price=o[t], entry_atr=signal_atr,
                    equity_at_entry=equity, capped=capped, last_mark=entry_price,
                )
        pending_entry = 0

        # 2. Stop check during the day
        if trade is not None and stop_type != 'none':
            stop_line[t] = trade.stop
            hit = _stop_fill(trade.direction, trade.stop, o[t], h[t], l[t])
            if hit is not None:
                price, reason = hit
                close_trade(t, price * (1 - trade.direction * cost), reason)

        # 3. Mark to market and trail the stop (the new level applies from tomorrow)
        if trade is not None:
            d_ = trade.direction
            cash = d_ * trade.qty * (c[t] - trade.last_mark)
            pnl[t] += cash
            equity += cash
            trade.last_mark = c[t]

            if d_ == 1:
                trade.best_price = max(trade.best_price, h[t])
                trade.worst_price = min(trade.worst_price, l[t])
            else:
                trade.best_price = min(trade.best_price, l[t])
                trade.worst_price = max(trade.worst_price, h[t])

            if stop_type == 'atr':
                new_stop = trade.best_price - d_ * trail_mult * atr[t]
            elif stop_type == 'percent':
                new_stop = trade.best_price * (1 - d_ * trail_pct)
            if stop_type != 'none':
                trade.stop = max(trade.stop, new_stop) if d_ == 1 else min(trade.stop, new_stop)

        pos = trade.direction if trade is not None else 0
        position[t] = pos
        shares[t] = trade.qty if trade is not None else 0.0
        equity_curve[t] = equity

        # 4. Signals at today's close, executed at tomorrow's open
        if t < n - 1:
            if cross_up[t]:
                if pos == -1 and exit_on_opposite_cross:
                    pending_exit = True
                if pos == 0 or pending_exit:
                    pending_entry = 1
            elif cross_down[t]:
                if pos == 1 and exit_on_opposite_cross:
                    pending_exit = True
                if allow_short and (pos == 0 or pending_exit):
                    pending_entry = -1

    # Close anything still open at the final close
    if trade is not None:
        close_trade(n - 1, c[-1] * (1 - trade.direction * cost), 'end_of_data')
        equity_curve[-1] = equity

    d['stop'] = stop_line
    prev_equity = np.r_[initial_equity, equity_curve[:-1]]
    daily = pd.DataFrame({
        'pnl':      pnl,
        'equity':   equity_curve,
        'return':   np.where(prev_equity > 0, pnl / prev_equity, 0.0),
        'position': position,
        'shares':   shares,
    }, index=d.index)
    return {'trades': pd.DataFrame(trades), 'daily': daily, 'data': d}


# =============================================================================
# 4. Statistics
# =============================================================================

def trade_stats(trades: pd.DataFrame) -> pd.Series:
    """Trade statistics in R-multiples, plus how the sizing behaved."""
    if len(trades) == 0:
        return pd.Series({'trades': 0})

    R = trades['R']
    wins, losses = R[R > 0], R[R <= 0]
    total_R = R.sum()
    has_wins, has_losses = len(wins) > 0, len(losses) > 0

    return pd.Series({
        'trades':           len(R),
        'win_rate':         (R > 0).mean(),
        'avg_win_R':        wins.mean() if has_wins else np.nan,
        'avg_loss_R':       losses.mean() if has_losses else np.nan,
        'payoff_ratio':     wins.mean() / abs(losses.mean()) if has_wins and has_losses else np.nan,
        'profit_factor':    wins.sum() / abs(losses.sum()) if has_losses and losses.sum() != 0 else np.nan,
        'expectancy_R':     R.mean(),
        'skew_R':           R.skew(),
        'max_R':            R.max(),
        'min_R':            R.min(),
        'top5_share_of_R':  R.nlargest(5).sum() / total_R if total_R > 0 else np.nan,
        'avg_bars_win':     trades.loc[R > 0, 'bars_held'].mean(),
        'avg_bars_loss':    trades.loc[R <= 0, 'bars_held'].mean(),
        'gap_exit_share':   (trades['exit_reason'] == 'gap').mean(),
        'avg_notional_pct': trades['notional_pct'].mean(),
        'avg_risk_pct':     trades['risk_pct_actual'].mean(),
        'capped_share':     trades['capped'].mean(),
    })


def equity_stats(daily: pd.DataFrame, initial_equity: float = DEFAULT_EQUITY) -> pd.Series:
    """
    Growth and drawdown statistics on mark-to-market equity.

    max_dd      : worst fall from a previous peak (fraction of the peak)
    max_loss    : worst fall below starting equity (fraction of starting equity, 0 if never below)
    """
    eq = daily['equity'].to_numpy()
    years = len(eq) / TRADING_DAYS
    final = eq[-1]
    cagr = (final / initial_equity) ** (1 / years) - 1 if final > 0 else -1.0

    peak = np.maximum.accumulate(np.r_[initial_equity, eq])[1:]
    dd = eq / peak - 1
    max_dd = -dd.min()
    max_loss = max(0.0, 1 - eq.min() / initial_equity)

    # Longest run of consecutive days below a previous peak
    day = np.arange(len(dd))
    last_at_peak = np.maximum.accumulate(np.where(dd < 0, -1, day))
    longest_underwater = int((day - last_at_peak).max())

    r = daily['return']
    return pd.Series({
        'cagr':               cagr,
        'max_dd':             max_dd,
        'max_loss':           max_loss,
        'calmar':             cagr / max_dd if max_dd > 0 else np.nan,
        'ulcer_index':        np.sqrt(np.mean((100 * dd) ** 2)),
        'longest_underwater': longest_underwater,
        'sharpe':             r.mean() / r.std() * np.sqrt(TRADING_DAYS) if r.std() > 0 else np.nan,
        'time_in_market':     (daily['position'] != 0).mean(),
        'final_equity':       final,
    })


def _path_stats(df, result, initial_equity):
    """All per-path statistics for one backtest result."""
    s = pd.concat([trade_stats(result['trades']), equity_stats(result['daily'], initial_equity)])
    lr = df['log_return']
    s['buy_hold_sharpe'] = lr.mean() / lr.std() * np.sqrt(TRADING_DAYS)
    return s


def summarise(per_path: pd.DataFrame, dd_thresholds=DEFAULT_DD_THRESHOLDS) -> pd.Series:
    """Distribution of outcomes across paths: medians, tail drawdowns and drawdown probabilities."""
    s = {
        'cagr_median':         per_path['cagr'].median(),
        'cagr_p5':             per_path['cagr'].quantile(0.05),
        'sharpe_median':       per_path['sharpe'].median(),
        'max_dd_median':       per_path['max_dd'].median(),
        'max_dd_p95':          per_path['max_dd'].quantile(0.95),
        'calmar_median':       per_path['calmar'].median(),
        'ulcer_median':        per_path['ulcer_index'].median(),
        'underwater_days_med': per_path['longest_underwater'].median(),
        'avg_risk_pct':        per_path['avg_risk_pct'].median(),
        'avg_notional_pct':    per_path['avg_notional_pct'].median(),
        'capped_share':        per_path['capped_share'].median(),
    }
    for x in dd_thresholds:
        s[f'p_dd_over_{x * 100:g}'] = (per_path['max_dd'] >= x).mean()
    return pd.Series(s)


def ruin_table(per_path: pd.DataFrame, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
               measure='drawdown') -> pd.DataFrame:
    """
    Probability of ruin for each config: the share of paths that ever hit each threshold.

    per_path  : the second output of compare(..., return_per_path=True)
                or compare_parallel(..., return_per_path=True)
    measure   : 'drawdown'  fall from a previous equity peak (uses max_dd)
                'loss'      fall below starting equity       (uses max_loss)

    Returns a table with one row per config and one column per threshold.
    """
    col = {'drawdown': 'max_dd', 'loss': 'max_loss'}[measure]
    label = 'DD' if measure == 'drawdown' else 'Loss'
    configs = pd.unique(per_path['config'])
    grouped = per_path.groupby('config')[col]
    table = pd.DataFrame({
        f'{label} >= {x * 100:g}%': grouped.apply(lambda v, x=x: (v >= x).mean())
        for x in thresholds
    })
    return table.loc[configs]


# =============================================================================
# 5. Running many paths
# =============================================================================

def generate_paths(gen, seeds, T=TRADING_DAYS * 20, S0=100.0):
    """Generate the synthetic paths once, so every configuration is tested on the same paths."""
    return {seed: gen.generate(T=T, S0=S0, seed=seed)
            for seed in tqdm(seeds, desc='Generating paths')}


def run_many(paths, progress=None, **bt_params):
    """
    Run one configuration on every path in a {seed: df} dict.

    Returns
    -------
    per_path : one row of stats per seed
    pooled   : every trade from every path, with a 'seed' column
    """
    initial_equity = bt_params.get('initial_equity', DEFAULT_EQUITY)
    rows, all_trades = [], []
    for seed, df in paths.items():
        res = backtest_ma_atr(df, **bt_params)
        s = _path_stats(df, res, initial_equity)
        s['seed'] = seed
        rows.append(s)
        if len(res['trades']):
            all_trades.append(res['trades'].assign(seed=seed))
        if progress is not None:
            progress.update(1)

    per_path = pd.DataFrame(rows).set_index('seed')
    pooled = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return per_path, pooled


def _summary_table(per_path, configs, dd_thresholds):
    """One summary row per config, in the order the configs were given."""
    return pd.DataFrame({
        name: summarise(per_path[per_path['config'] == name].set_index('seed').sort_index(),
                        dd_thresholds)
        for name in configs
    }).T


def compare(paths, configs, dd_thresholds=DEFAULT_DD_THRESHOLDS, return_per_path=False, **common):
    """
    Run several configurations on the same pre-generated paths and summarise each.

    configs : {name: dict of backtest parameters}, e.g.
              {'risk 1%': {'sizing': {'method': 'fixed_risk', 'risk_pct': 0.01}}}
    common  : parameters shared by every configuration (e.g. cost_bps=5)
    return_per_path : also return every path's stats for every config (for ruin_table)
    """
    frames = []
    with tqdm(total=len(configs) * len(paths), desc='Backtesting') as bar:
        for name, params in configs.items():
            bar.set_postfix_str(name)
            per_path, _ = run_many(paths, progress=bar, **{**common, **params})
            frames.append(per_path.reset_index().assign(config=name))

    per_path = pd.concat(frames, ignore_index=True)
    table = _summary_table(per_path, configs, dd_thresholds)
    return (table, per_path) if return_per_path else table


def _run_seed_chunk(gen, seeds, configs, T, S0, common):
    """Worker for compare_parallel: generate each path from its seed and run every config on it."""
    rows = []
    for seed in seeds:
        df = gen.generate(T=T, S0=S0, seed=seed)
        for name, params in configs.items():
            p = {**common, **params}
            res = backtest_ma_atr(df, **p)
            s = _path_stats(df, res, p.get('initial_equity', DEFAULT_EQUITY))
            s['seed'] = seed
            s['config'] = name
            rows.append(s.to_dict())
    return rows


def compare_parallel(gen, seeds, configs, T=TRADING_DAYS * 20, S0=100.0, n_jobs=None,
                     chunk_size=10, dd_thresholds=DEFAULT_DD_THRESHOLDS,
                     return_per_path=False, **common):
    """
    Same output as compare(), but split across CPU cores.

    Paths are generated inside the workers from their seeds, so there is no need to
    call generate_paths() first and the paths are never all held in memory.
    Results are identical to compare() on the same seeds.

    gen             : a SyntheticMarketGenerator (its class must be defined in a module)
    seeds           : e.g. range(1000, 6000)
    n_jobs          : worker processes (default: all cores but one)
    chunk_size      : seeds per task; larger = less overhead, jumpier progress bar
    return_per_path : also return every path's stats for every config (for ruin_table)
    common          : parameters shared by every config (e.g. cost_bps=5)
    """
    seeds = list(seeds)
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    chunks = [seeds[i:i + chunk_size] for i in range(0, len(seeds), chunk_size)]

    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as pool, \
            tqdm(total=len(seeds) * len(configs), desc=f'Backtesting ({n_jobs} workers)') as bar:
        futures = {pool.submit(_run_seed_chunk, gen, chunk, configs, T, S0, common): len(chunk)
                   for chunk in chunks}
        for f in as_completed(futures):
            rows.extend(f.result())
            bar.update(futures[f] * len(configs))

    per_path = pd.DataFrame(rows)
    table = _summary_table(per_path, configs, dd_thresholds)
    return (table, per_path) if return_per_path else table


# =============================================================================
# 6. Stop calibration
# =============================================================================

MAE_COLUMNS = {'atr': 'mae_atr', 'pct': 'mae_pct', 'R': 'mae_R'}
MAE_DEFAULT_LEVELS = {
    'atr': np.arange(0.5, 8.01, 0.5),
    'pct': np.arange(0.01, 0.1501, 0.01),
    'R':   np.arange(0.25, 3.01, 0.25),
}


def mae_profile(trades, unit='atr', levels=None):
    """
    For each candidate stop distance, what share of trades would it have cut?
    Run the backtest with stop_type='none' first so the stop doesn't truncate MAE.

    winners_cut    : share of eventual winners whose MAE reached this distance
    losers_cut     : share of eventual losers whose MAE reached this distance
    winning_R_lost : share of all winning R that came from those cut winners
                     (the trend-following cost of a tight stop)
    """
    col = MAE_COLUMNS[unit]
    levels = MAE_DEFAULT_LEVELS[unit] if levels is None else levels

    win = trades['R'] > 0
    total_win_R = trades.loc[win, 'R'].sum()
    rows = []
    for x in levels:
        cut = trades[col] >= x
        rows.append({
            'stop_at':        x,
            'winners_cut':    cut[win].mean(),
            'losers_cut':     cut[~win].mean(),
            'winning_R_lost': trades.loc[win & cut, 'R'].sum() / total_win_R if total_win_R > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index('stop_at')


def split_paths(paths, calib_frac=0.5, split_seed=0):
    """Randomly split a {seed: df} dict into calibration and test sets."""
    seeds = np.array(list(paths.keys()))
    np.random.default_rng(split_seed).shuffle(seeds)
    n_calib = int(round(len(seeds) * calib_frac))
    calib = {s: paths[s] for s in seeds[:n_calib]}
    test = {s: paths[s] for s in seeds[n_calib:]}
    return calib, test


def _sweep_stats(paths, **params):
    """Headline statistics for one stop width on one set of paths."""
    per_path, pooled = run_many(paths, **params)
    years = len(next(iter(paths.values()))) / TRADING_DAYS
    pooled_stats = trade_stats(pooled)
    return {
        'sharpe':          per_path['sharpe'].median(),
        'cagr':            per_path['cagr'].median(),
        'max_dd_p95':      per_path['max_dd'].quantile(0.95),
        'win_rate':        pooled_stats['win_rate'],
        'profit_factor':   pooled_stats['profit_factor'],
        'trades_per_year': per_path['trades'].median() / years,
    }


def stop_sweep(paths, param, values, calib_frac=0.5, split_seed=0, metric='sharpe', **base_params):
    """
    Test a range of stop widths on calibration paths and check them on separate test paths.

    param       : the parameter to vary, e.g. 'trail_mult' (ATR) or 'trail_pct' (percent)
    values      : the widths to test
    metric      : the column used to choose a width ('sharpe' is unaffected by risk level)
    base_params : everything else, e.g. stop_type='percent', cost_bps=5.
                  Defaults to fixed-risk sizing at 0.5%, so every width takes the same risk per trade.
                  For ATR stops, initial_mult follows trail_mult unless you pass it.

    Returns a DataFrame indexed by width with calib_* and test_* columns, a smoothed
    calibration metric, and a 'chosen' flag marking the width picked from the plateau.
    """
    base_params.setdefault('sizing', {'method': 'fixed_risk', 'risk_pct': 0.005})
    calib, test = split_paths(paths, calib_frac, split_seed)

    rows = []
    for v in tqdm(values, desc=f'Sweeping {param}'):
        params = {**base_params, param: v}
        c = _sweep_stats(calib, **params)
        t = _sweep_stats(test, **params)
        rows.append({param: v,
                     **{f'calib_{k}': x for k, x in c.items()},
                     **{f'test_{k}': x for k, x in t.items()}})
    out = pd.DataFrame(rows).set_index(param)

    # Pick from a plateau: average each width with its neighbours before choosing.
    # The end widths have only one neighbour, so they can't be chosen.
    smoothed = out[f'calib_{metric}'].rolling(3, center=True).mean()
    out[f'calib_{metric}_smoothed'] = smoothed
    out['chosen'] = out.index == smoothed.idxmax()

    raw_best = out[f'calib_{metric}'].idxmax()
    if raw_best in (out.index[0], out.index[-1]):
        print(f"Note: the best calibration {metric} is at the edge of the range ({raw_best:g}). "
              f"Extend the range in that direction before trusting the chosen width.")
    return out