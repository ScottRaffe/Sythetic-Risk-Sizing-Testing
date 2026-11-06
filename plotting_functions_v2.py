"""
Plotting tools for the synthetic-market position-sizing project.

Contents
--------
1. Style                 palette, axis styling and formatting helpers
2. Market paths          shade_regimes, plot_with_regimes
3. Single backtest       plot_backtest
4. Stop calibration      plot_mae, plot_stop_sweep
5. Comparing configs     plot_frontier, plot_by_risk
6. Probability of ruin   plot_ruin_by_risk, plot_ruin_curves

Sections 5 and 6 read the family and risk level from each config name, which must
end in a risk level such as "7 ATR stop | risk 1.0%". Everything before the number
is the family. Names without a number (e.g. "notional 1x") are drawn as a grey reference.
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

import backtester_functions as bf


# =============================================================================
# 1. Style
# =============================================================================

# Text and surfaces
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'

# Series colours, assigned in this fixed order
FAMILY_COLOURS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
FAMILY_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
BLUE, ORANGE, GREEN = FAMILY_COLOURS[0], FAMILY_COLOURS[1], FAMILY_COLOURS[2]
REF_COLOUR = '#8c8b86'          # notional 1x reference

# Market regimes: 0 Bull, 1 Bear, 2 Crisis, 3 Choppy
REGIME_COLOURS = {0: 'tab:green', 1: 'tab:orange', 2: 'tab:red', 3: 'tab:grey'}

# Tick formatters
PCT = FuncFormatter(lambda v, _: f'{v:g}%')               # value already in percent
PCT_OF_FRACTION = FuncFormatter(lambda v, _: f'{v * 100:g}%')   # value is a fraction
PLAIN = FuncFormatter(lambda v, _: f'{v:g}')


def _style(ax):
    """Light background, recessive grid and axes."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _log_price_axis(ax):
    """Log-scale price axis with plain number labels (no scientific notation)."""
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(PLAIN)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _line(ax, x, y, i, **kwargs):
    """A family's line with its colour and marker."""
    ax.plot(x, y, color=FAMILY_COLOURS[i], lw=2, marker=FAMILY_MARKERS[i], ms=7,
            mec=SURFACE, mew=1.5, zorder=3, **kwargs)


def _grid(n_panels, ncols=3, panel_size=(4.6, 3.6)):
    """Grid of panels sharing both axes; unused panels are hidden."""
    ncols = min(ncols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows + 0.9),
                             sharex=True, sharey=True, squeeze=False, facecolor=SURFACE)
    axes = list(axes.ravel())
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes[:n_panels], ncols


def _label_panels(axes, ncols, xlabel, ylabel):
    """Axis labels and tick labels on the outer panels, including panels with an empty slot below."""
    n = len(axes)
    for i, ax in enumerate(axes):
        if i + ncols >= n:                      # nothing visible below this panel
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel(xlabel)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)


def _finish(fig, title, handles):
    """Figure title, a shared legend across the top, and show."""
    fig.suptitle(title, x=0.01, ha='left', fontsize=13, color=INK)
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plt.show()


# =============================================================================
# 2. Market paths
# =============================================================================

def shade_regimes(ax, regimes, regime_params, colours=REGIME_COLOURS, alpha=0.25):
    """Shade each continuous block of a regime behind whatever is plotted on ax."""
    regimes = np.asarray(regimes)
    change = np.flatnonzero(np.diff(regimes)) + 1       # first day of each new block
    starts = np.r_[0, change]
    widths = np.r_[change, len(regimes)] - starts
    labels = regimes[starts]

    ymin, ymax = ax.get_ylim()
    for r in regime_params:
        in_regime = labels == r
        if in_regime.any():
            ax.broken_barh(list(zip(starts[in_regime], widths[in_regime])), (ymin, ymax - ymin),
                           facecolors=colours[r], alpha=alpha, linewidth=0, zorder=0)
    ax.set_ylim(ymin, ymax)


def plot_with_regimes(df, regime_params, log_price=False):
    """Price (top) and effective volatility (bottom) with the regimes shaded behind."""
    if 'effective_vol' in df:
        effective_vol = df['effective_vol'] * 100
    else:   # older paths: GARCH vol x regime multiplier
        vol_mult = df['regime'].map({r: p['vol_mult'] for r, p in regime_params.items()})
        effective_vol = df['conditional_vol'] * vol_mult * 100

    fig, (ax_price, ax_vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]}, facecolor=SURFACE)
    for ax in (ax_price, ax_vol):
        _style(ax)
        ax.grid(False)

    ax_price.plot(df.index, df['price'], color=INK, lw=0.8)
    if log_price:
        _log_price_axis(ax_price)
    ax_price.set_ylabel('Price')

    ax_vol.plot(df.index, effective_vol, color=INK, lw=0.6)
    ax_vol.yaxis.set_major_formatter(PCT)
    ax_vol.set_ylabel('Effective vol')
    ax_vol.set_xlabel('Day')

    for ax in (ax_price, ax_vol):
        shade_regimes(ax, df['regime'], regime_params)
        ax.set_xlim(df.index[0], df.index[-1])

    handles = [Patch(facecolor=REGIME_COLOURS[r], alpha=0.25, label=p['name'])
               for r, p in regime_params.items()]
    ax_price.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Single backtest
# =============================================================================

def plot_backtest(result, start=None, end=None, log_price=True):
    """Price with MAs, the trailing stop and trade markers (top), and the equity curve (bottom)."""
    d = result['data'].loc[start:end]
    daily = result['daily'].loc[start:end]
    trades = result['trades']
    trades = trades[(trades['entry_day'] >= d.index[0]) & (trades['entry_day'] <= d.index[-1])]
    longs = trades[trades['direction'] == 'long']
    shorts = trades[trades['direction'] == 'short']

    fig, (ax_price, ax_eq) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                          gridspec_kw={'height_ratios': [3, 1]}, facecolor=SURFACE)
    for ax in (ax_price, ax_eq):
        _style(ax)

    ax_price.plot(d.index, d['close'], color=INK, lw=0.8, label='Close')
    ax_price.plot(d.index, d['fast_ma'], color=BLUE, lw=1, label='Fast MA')
    ax_price.plot(d.index, d['slow_ma'], color=FAMILY_COLOURS[4], lw=1, label='Slow MA')
    ax_price.plot(d.index, d['stop'], color=ORANGE, lw=1, ls='--', label='Trailing stop')
    ax_price.scatter(longs['entry_day'], longs['entry_price'], marker='^', color=GREEN, s=60,
                     edgecolor=SURFACE, zorder=3, label='Long entry')
    ax_price.scatter(shorts['entry_day'], shorts['entry_price'], marker='v', color='tab:red', s=60,
                     edgecolor=SURFACE, zorder=3, label='Short entry')
    ax_price.scatter(trades['exit_day'], trades['exit_price'], marker='x', color=INK, s=40,
                     zorder=3, label='Exit')
    if log_price:
        _log_price_axis(ax_price)
    ax_price.set_ylabel('Price')
    ax_price.legend(loc='lower left', bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=7)

    ax_eq.plot(daily.index, daily['equity'], color=INK, lw=0.8)
    ax_eq.set_ylabel('Equity')
    ax_eq.set_xlabel('Day')
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. Stop calibration
# =============================================================================

MAE_COLUMNS = {'atr': 'mae_atr', 'pct': 'mae_pct', 'R': 'mae_R'}
MAE_UNIT_LABELS = {'atr': 'ATRs', 'pct': '% of entry price', 'R': 'R'}


def plot_mae(trades, unit='atr', levels=None):
    """
    Left:  how far eventual winners and losers went against you (MAE distribution).
    Right: what a stop at each distance would have cost (winners cut, winning R lost).
    Run the backtest with stop_type='none' first so the stop doesn't truncate MAE.
    """
    col = MAE_COLUMNS[unit]
    x_fmt = PCT_OF_FRACTION if unit == 'pct' else PLAIN
    win = trades['R'] > 0
    profile = bf.mae_profile(trades, unit, levels)

    fig, (ax_dist, ax_cost) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=SURFACE)
    for ax in (ax_dist, ax_cost):
        _style(ax)
        ax.xaxis.set_major_formatter(x_fmt)
        ax.yaxis.set_major_formatter(PCT_OF_FRACTION)

    # Left: cumulative distribution of MAE
    for mask, colour, name in ((win, BLUE, 'Winners'), (~win, ORANGE, 'Losers')):
        x = np.sort(trades.loc[mask, col].to_numpy())
        ax_dist.plot(x, np.arange(1, len(x) + 1) / len(x), color=colour, lw=2, label=f'{name} ({len(x)})')
    ax_dist.set_xlim(0, profile.index.max())
    ax_dist.set_xlabel(f'Max adverse excursion ({MAE_UNIT_LABELS[unit]})')
    ax_dist.set_ylabel('Share of trades with MAE at or below this')
    ax_dist.set_title('How far trades went against you', loc='left', fontsize=11)
    ax_dist.legend(frameon=False, fontsize=9, loc='lower right')

    # Right: cost of a stop at each distance
    _line(ax_cost, profile.index, profile['winners_cut'], 0, label='Winners that would be stopped out')
    _line(ax_cost, profile.index, profile['winning_R_lost'], 1, label='Share of winning R lost')
    ax_cost.set_xlabel(f'Stop distance ({MAE_UNIT_LABELS[unit]})')
    ax_cost.set_title('Cost of a stop at each distance', loc='left', fontsize=11)
    ax_cost.legend(frameon=False, fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_stop_sweep(sweep, metric='sharpe', title=None):
    """Calibration vs test results across stop widths (from bf.stop_sweep), with the chosen width marked."""
    param = sweep.index.name
    x = sweep.index.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _style(ax)
    _line(ax, x, sweep[f'calib_{metric}'], 0, label='Calibration paths')
    _line(ax, x, sweep[f'test_{metric}'], 1, label='Test paths')
    ax.plot(x, sweep[f'calib_{metric}_smoothed'], color=BLUE, lw=1, ls=':', label='Calibration, smoothed')

    chosen = sweep.index[sweep['chosen']][0]
    ax.axvline(chosen, color=INK_2, lw=1, ls='--')
    ax.annotate(f'chosen: {chosen:g}', (chosen, ax.get_ylim()[1]), textcoords='offset points',
                xytext=(6, -14), fontsize=9, color=INK_2)

    if param.endswith('pct'):
        ax.xaxis.set_major_formatter(PCT_OF_FRACTION)
    metric_label = metric.replace('_', ' ')
    ax.set_xlabel(param)
    ax.set_ylabel(metric_label)
    ax.set_title(title or f'Median {metric_label} by stop width', loc='left', fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. Comparing configs (table from compare / compare_parallel)
# =============================================================================

def _parse(table):
    """Add 'family' and 'risk' columns parsed from each config name (the table's index)."""
    rows = []
    for name in table.index:
        m = re.match(r'^(.*?)\s*([\d.]+)%\s*$', name)
        if m:
            rows.append({'name': name, 'family': m.group(1).strip(), 'risk': float(m.group(2)) / 100})
        else:
            rows.append({'name': name, 'family': None, 'risk': np.nan})
    meta = pd.DataFrame(rows).set_index('name')
    families = list(dict.fromkeys(meta['family'].dropna()))     # order of first appearance
    return table.join(meta), families


def _legend_handles(families, has_ref, markers=True):
    """Legend entries for each family, plus the notional reference if present."""
    handles = [Line2D([0], [0], color=FAMILY_COLOURS[i], lw=2,
                      marker=FAMILY_MARKERS[i] if markers else None, ms=7, mec=SURFACE, mew=1.5, label=f)
               for i, f in enumerate(families)]
    if has_ref:
        handles.append(Line2D([0], [0], color=REF_COLOUR, ls='--', lw=1.5, label='Notional 1x (reference)'))
    return handles


def _format_risk_axis(ax, risks):
    """Evenly spaced risk levels on the x-axis, so close values like 2% and 2.5% don't overlap."""
    ax.set_xticks(range(len(risks)))
    ax.set_xticklabels([f'{r * 100:g}%' for r in risks])
    ax.set_xlim(-0.4, len(risks) - 0.6)


def plot_frontier(table):
    """Median CAGR against tail drawdown, one line per family running from low to high risk."""
    df, families = _parse(table)
    ref = df[df['family'].isna()]

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK_2, lw=0.8)

    for i, fam in enumerate(families):
        sub = df[df['family'] == fam].sort_values('risk')
        x, y = sub['max_dd_p95'] * 100, sub['cagr_median'] * 100
        _line(ax, x, y, i)
        # Label only the highest risk level (the low-risk ends cluster near the origin)
        ax.annotate(f"{sub['risk'].iloc[-1] * 100:g}%", (x.iloc[-1], y.iloc[-1]),
                    textcoords='offset points', xytext=(6, 4), fontsize=8, color=INK_2)

    for name, row in ref.iterrows():
        point = (row['max_dd_p95'] * 100, row['cagr_median'] * 100)
        ax.scatter(*point, color=REF_COLOUR, s=60, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(name, point, textcoords='offset points', xytext=(6, 4), fontsize=8, color=INK_2)

    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel('Tail drawdown: 95th percentile of max drawdown across paths')
    ax.set_ylabel('Median CAGR')
    ax.set_title('Return vs tail drawdown (each line runs from lowest to highest risk %)',
                 loc='left', fontsize=12)
    ax.legend(handles=_legend_handles(families, False), loc='upper left', frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_by_risk(table, dd_threshold=30):
    """Median CAGR, median Sharpe, tail drawdown and drawdown probability against risk per trade."""
    df, families = _parse(table)
    ref = df[df['family'].isna()]
    risks = np.sort(df['risk'].dropna().unique())
    xpos = {r: i for i, r in enumerate(risks)}

    metrics = [   # (column, title, shown as %)
        ('cagr_median', 'Median CAGR', True),
        ('sharpe_median', 'Median Sharpe', False),
        ('max_dd_p95', 'Tail drawdown (95th pct of max DD)', True),
        (f'p_dd_over_{dd_threshold:g}', f'Share of paths with max DD ≥ {dd_threshold:g}%', True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, facecolor=SURFACE)
    for ax, (col, title, is_pct) in zip(axes.flat, metrics):
        _style(ax)
        scale = 100 if is_pct else 1
        for i, fam in enumerate(families):
            sub = df[df['family'] == fam].sort_values('risk')
            _line(ax, sub['risk'].map(xpos), sub[col] * scale, i)
        for _, row in ref.iterrows():
            ax.axhline(row[col] * scale, color=REF_COLOUR, ls='--', lw=1.5, zorder=2)

        _format_risk_axis(ax, risks)
        if is_pct:
            ax.yaxis.set_major_formatter(PCT)
        ax.set_title(title, loc='left', fontsize=11)

    for ax in axes[1]:
        ax.set_xlabel('Risk per trade')
    _finish(fig, 'Results by risk per trade', _legend_handles(families, len(ref) > 0))


# =============================================================================
# 6. Probability of ruin (per_path from compare / compare_parallel with return_per_path=True)
# =============================================================================

RUIN_COLUMNS = {'drawdown': 'max_dd', 'loss': 'max_loss'}
RUIN_LABELS = {'drawdown': 'drawdown from peak', 'loss': 'loss below starting equity'}


def plot_ruin_by_risk(per_path, thresholds=(0.1, 0.2, 0.3, 0.5), measure='drawdown', ncols=3):
    """
    One panel per ruin threshold: the share of paths that ever reached it, against
    risk per trade, one line per family. Notional 1x is the dashed grey reference.

    measure : 'drawdown' (fall from a previous peak) or 'loss' (fall below starting equity)
    """
    col = RUIN_COLUMNS[measure]
    grouped = per_path.groupby('config')[col]
    probs = pd.DataFrame({x: grouped.apply(lambda v, x=x: (v >= x).mean()) for x in thresholds})
    probs = probs.loc[pd.unique(per_path['config'])]    # keep config order so colours match other plots
    df, families = _parse(probs)
    ref = df[df['family'].isna()]
    risks = np.sort(df['risk'].dropna().unique())
    xpos = {r: i for i, r in enumerate(risks)}

    fig, axes, ncols = _grid(len(thresholds), ncols)
    for ax, x in zip(axes, thresholds):
        _style(ax)
        for _, row in ref.iterrows():
            ax.axhline(row[x] * 100, color=REF_COLOUR, ls='--', lw=1.5, zorder=2)
        for i, fam in enumerate(families):
            sub = df[df['family'] == fam].sort_values('risk')
            _line(ax, sub['risk'].map(xpos), sub[x] * 100, i)

        _format_risk_axis(ax, risks)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylim(-3, 103)
        ax.set_title(f'{RUIN_LABELS[measure].capitalize()} ≥ {x * 100:g}%', loc='left', fontsize=11)

    _label_panels(axes, ncols, 'Risk per trade', 'Share of paths')
    _finish(fig, f'Probability of ruin by risk per trade ({RUIN_LABELS[measure]})',
            _legend_handles(families, len(ref) > 0))


def plot_ruin_curves(per_path, risks=None, measure='drawdown', max_level=None, ncols=3):
    """
    One panel per risk level: the share of paths whose worst drawdown (or loss) reached
    at least each level, one line per family. Read any ruin threshold off the x-axis:
    lower and further left is safer. Notional 1x is the dashed grey line.

    risks     : risk levels to show, e.g. [0.01, 0.02, 0.05] (default: all)
    measure   : 'drawdown' (fall from a previous peak) or 'loss' (fall below starting equity)
    max_level : right edge of the x-axis (default: just past the worst outcome in the data)
    """
    col = RUIN_COLUMNS[measure]
    meta, families = _parse(pd.DataFrame(index=pd.unique(per_path['config'])))
    ref_names = meta.index[meta['family'].isna()]
    panel_risks = np.sort(meta['risk'].dropna().unique())
    if risks is not None:
        panel_risks = [r for r in panel_risks if np.isclose(risks, r).any()]

    if max_level is None:
        max_level = min(1.0, np.ceil(per_path[col].max() * 10) / 10)
    levels = np.linspace(0, max_level, 201)
    sorted_values = {name: np.sort(v.to_numpy()) for name, v in per_path.groupby('config')[col]}

    def exceedance(values):
        """Share of paths with a value at or above each level."""
        return 1 - np.searchsorted(values, levels, side='left') / len(values)

    fig, axes, ncols = _grid(len(panel_risks), ncols)
    for ax, r in zip(axes, panel_risks):
        _style(ax)
        for name in ref_names:
            ax.plot(levels * 100, exceedance(sorted_values[name]) * 100,
                    color=REF_COLOUR, ls='--', lw=1.5, zorder=2)
        for i, fam in enumerate(families):
            match = meta[(meta['family'] == fam) & np.isclose(meta['risk'].fillna(-1), r)]
            if len(match):
                ax.plot(levels * 100, exceedance(sorted_values[match.index[0]]) * 100,
                        color=FAMILY_COLOURS[i], lw=2, zorder=3)
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_xlim(0, max_level * 100)
        ax.set_ylim(-3, 103)
        ax.set_title(f'Risk {r * 100:g}% per trade', loc='left', fontsize=11)

    _label_panels(axes, ncols, f'Worst {RUIN_LABELS[measure]}', 'Share of paths reaching it')
    _finish(fig, f'Chance of reaching each level of {RUIN_LABELS[measure]}',
            _legend_handles(families, len(ref_names) > 0, markers=False))