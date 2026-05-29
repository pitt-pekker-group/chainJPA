"""pae_explorer.py — interactive heatmap explorer for PAE parameter sweeps.

Expects a pandas DataFrame with columns 'di', 'im', 'cm', 'lm', 'pm', 'PAE'.

Two of the five parameters go on the heatmap axes (selectable via dropdown).
Each of the other three is either fixed at a user-selected value via a
dropdown OR aggregated:

  - 'im', 'cm', 'lm' that are off-axis are always fixed via dropdowns.
  - 'pm' that is off-axis is, by default, aggregated by taking the max
    across pm. Untick the "aggregate pm" checkbox to fix pm at a value
    instead.
  - 'di' that is off-axis is, by default, aggregated across di by mean or
    a chosen percentile. Untick "aggregate di" to fix di at a value instead.

PAE NaNs are treated as zero. The plot supports an optional single-level
contour overlay.

Usage in a Jupyter notebook:

    from pae_explorer import make_pae_explorer
    make_pae_explorer(df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output


PRECISION = 6
PARAMS = ['di', 'im', 'cm', 'lm', 'pm']
ALWAYS_FIX_PARAMS = ['im', 'cm', 'lm']   # never aggregated when off-axis


def _round_unique(series):
    return sorted(series.round(PRECISION).unique())


def _agg_label(kind, q_pct):
    if kind == 'mean':
        return 'mean'
    if abs(q_pct - 50) < 1e-9:
        return 'median'
    return f'p{q_pct:.0f}'


def make_pae_explorer(df, default_axis_x='im', default_axis_y='lm'):
    """Build and display the heatmap-explorer widget for `df`.

    Parameters
    ----------
    df : pandas.DataFrame
        Sweep results with columns 'di', 'im', 'cm', 'lm', 'pm', 'PAE'.
    default_axis_x, default_axis_y : str
        Initial x- and y-axis parameter names. Must differ; must each be
        one of {'di', 'im', 'cm', 'lm', 'pm'}.
    """
    missing = [c for c in PARAMS + ['PAE'] if c not in df.columns]
    if missing:
        raise KeyError(f'df is missing required columns: {missing}')
    if default_axis_x == default_axis_y:
        raise ValueError('default_axis_x and default_axis_y must differ')
    for ax in (default_axis_x, default_axis_y):
        if ax not in PARAMS:
            raise ValueError(f'invalid axis {ax!r}; must be one of {PARAMS}')

    finite_pae = df['PAE'].dropna()
    default_contour = float(finite_pae.median()) if len(finite_pae) else 0.0

    # Widgets -----------------------------------------------------------------
    axis_x = widgets.Dropdown(options=PARAMS, value=default_axis_x,
                              description='x axis:')
    axis_y = widgets.Dropdown(options=PARAMS, value=default_axis_y,
                              description='y axis:')

    # Fix dropdowns for ALL five params. Visibility flipped based on whether
    # the param is on an axis or being aggregated.
    fix_widgets = {
        p: widgets.Dropdown(options=_round_unique(df[p]), description=f'{p}:')
        for p in PARAMS
    }

    # Optional-aggregation toggles for pm and di.
    aggregate_pm = widgets.Checkbox(value=True,  description='aggregate pm (max)')
    aggregate_di = widgets.Checkbox(value=True,  description='aggregate di')

    # di-aggregation kind + percentile slider.
    agg_kind   = widgets.Dropdown(options=['mean', 'percentile'],
                                  value='percentile', description='over di:')
    percentile = widgets.FloatSlider(value=50, min=0, max=100, step=1,
                                     description='percentile:',
                                     continuous_update=False)

    contour_on    = widgets.Checkbox(value=True, description='show contour')
    contour_level = widgets.FloatText(value=default_contour, step=0.01,
                                      description='contour @ PAE:')

    out = widgets.Output()

    _suspend = False    # set True during axis change to silence intermediate redraws

    # Predicates --------------------------------------------------------------
    def on_axis_set():
        return {axis_x.value, axis_y.value}

    def pm_aggregated():
        return ('pm' not in on_axis_set()) and aggregate_pm.value

    def di_aggregated():
        return ('di' not in on_axis_set()) and aggregate_di.value

    def fixed_params():
        """Params currently driven by their fix dropdown (off-axis,
        not aggregated)."""
        on_axis = on_axis_set()
        params = []
        for p in PARAMS:
            if p in on_axis:
                continue
            if p == 'pm' and aggregate_pm.value:
                continue
            if p == 'di' and aggregate_di.value:
                continue
            params.append(p)
        return params

    # Helpers -----------------------------------------------------------------
    def update_visibility():
        on_axis = on_axis_set()
        pm_on_axis = 'pm' in on_axis
        di_on_axis = 'di' in on_axis

        # im, cm, lm fix dropdowns: shown iff off-axis.
        for p in ALWAYS_FIX_PARAMS:
            fix_widgets[p].layout.display = 'none' if p in on_axis else ''

        # pm controls: checkbox only meaningful when pm is off-axis.
        aggregate_pm.layout.display = 'none' if pm_on_axis else ''
        # pm fix dropdown: shown only when pm is off-axis AND not aggregated.
        fix_widgets['pm'].layout.display = (
            'none' if pm_on_axis or aggregate_pm.value else ''
        )

        # di controls.
        aggregate_di.layout.display = 'none' if di_on_axis else ''
        fix_widgets['di'].layout.display = (
            'none' if di_on_axis or aggregate_di.value else ''
        )
        # di-aggregation kind + percentile only visible when di is off-axis
        # AND being aggregated.
        agg_kind.layout.display = (
            'none' if (di_on_axis or not aggregate_di.value) else ''
        )
        percentile.layout.display = (
            'none' if (di_on_axis or not aggregate_di.value
                       or agg_kind.value != 'percentile')
            else ''
        )

    def compute_heat(sel, ax_x, ax_y):
        sel = sel.assign(
            x_r=sel[ax_x].round(PRECISION),
            y_r=sel[ax_y].round(PRECISION),
            di_r=sel['di'].round(PRECISION),
        )
        pm_agg = pm_aggregated()
        di_agg = di_aggregated()

        # Inner step: collapse the pm dimension. If pm is being aggregated
        # we max across pm within each (y, x[, di]) bucket. Otherwise pm has
        # already been filtered to a single value (or is on an axis) so
        # any duplicates are collapsed with mean.
        inner = 'max' if pm_agg else 'mean'
        if di_agg:
            per_di = sel.groupby(['y_r', 'x_r', 'di_r'])['PAE'].agg(inner)
            grp = per_di.groupby(level=['y_r', 'x_r'])
            if agg_kind.value == 'mean':
                heat_series = grp.mean()
            else:
                heat_series = grp.quantile(percentile.value / 100.0)
        else:
            heat_series = sel.groupby(['y_r', 'x_r'])['PAE'].agg(inner)

        return heat_series.unstack('x_r')

    def aggregation_description():
        parts = []
        if pm_aggregated():
            parts.append('max over pm')
        if di_aggregated():
            parts.append(f'{_agg_label(agg_kind.value, percentile.value)} over di')
        return parts

    # Callbacks ---------------------------------------------------------------
    def on_axis_change(*_):
        nonlocal _suspend
        _suspend = True
        try:
            update_visibility()
        finally:
            _suspend = False
        redraw()

    def on_toggle_change(*_):
        update_visibility()
        redraw()

    def redraw(*_):
        if _suspend:
            return
        with out:
            clear_output(wait=True)

            ax_x, ax_y = axis_x.value, axis_y.value
            if ax_x == ax_y:
                print('pick two different axes')
                return

            # Filter on every currently-fixed param.
            atol = 10 ** -PRECISION
            mask = np.ones(len(df), dtype=bool)
            for p in fixed_params():
                mask &= np.isclose(df[p], fix_widgets[p].value, atol=atol)
            sel = df[mask]
            if sel.empty:
                print('no matching data for this slice')
                return

            sel = sel.assign(PAE=sel['PAE'].fillna(0.0))

            heat = compute_heat(sel, ax_x, ax_y)
            if heat.empty:
                print('no data after aggregation')
                return

            Z = heat.values

            heat_str = heat.copy()
            heat_str.columns = [f'{v:.3g}' for v in heat.columns]
            heat_str.index   = [f'{v:.3g}' for v in heat.index]

            agg_parts = aggregation_description()
            cbar_label = 'PAE' + (f' ({", ".join(agg_parts)})' if agg_parts else '')

            plt.close('all')
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(heat_str, cmap='viridis', ax=ax,
                        cbar_kws={'label': cbar_label},
                        antialiased=False, linewidths=0)

            if contour_on.value and np.isfinite(Z).any():
                level = float(contour_level.value)
                zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
                if zmin <= level <= zmax:
                    # Order-0 (cell-boundary) contour: draw a line segment along
                    # every shared edge whose two neighbors straddle the level.
                    # Cell (i, j) occupies x in [j, j+1], y in [i, i+1] (seaborn
                    # heatmap convention), so a vertical edge between columns
                    # j-1 and j sits at x=j from y=i to y=i+1, and a horizontal
                    # edge between rows i-1 and i sits at y=i from x=j to x=j+1.
                    vl, vr = Z[:, :-1], Z[:,  1:]
                    vt, vb = Z[:-1, :], Z[1:,  :]
                    vcross = (np.isfinite(vl) & np.isfinite(vr)
                              & ((vl - level) * (vr - level) < 0))
                    hcross = (np.isfinite(vt) & np.isfinite(vb)
                              & ((vt - level) * (vb - level) < 0))
                    ii_v, jj_v = np.where(vcross)
                    ii_h, jj_h = np.where(hcross)
                    segments = (
                        [[(j + 1, i), (j + 1, i + 1)] for i, j in zip(ii_v, jj_v)] +
                        [[(j, i + 1), (j + 1, i + 1)] for i, j in zip(ii_h, jj_h)]
                    )
                    if segments:
                        lc = LineCollection(segments, colors='white',
                                            linewidths=1.6)
                        ax.add_collection(lc)
                else:
                    ax.text(0.02, 0.98,
                            f'contour {level:g} outside '
                            f'data range [{zmin:.3g}, {zmax:.3g}]',
                            transform=ax.transAxes, ha='left', va='top',
                            fontsize=9, color='white',
                            bbox=dict(facecolor='black', alpha=0.5,
                                      edgecolor='none', pad=2))

            ax.set_xlabel(ax_x)
            ax.set_ylabel(ax_y)
            ax.invert_yaxis()

            fix_parts = [f'{p}={fix_widgets[p].value:.4g}' for p in fixed_params()]
            ax.set_title('  '.join(fix_parts) if fix_parts else '')
            plt.tight_layout()
            plt.show()

    # Wire up observers -------------------------------------------------------
    axis_x.observe(on_axis_change, names='value')
    axis_y.observe(on_axis_change, names='value')
    for w in fix_widgets.values():
        w.observe(redraw, names='value')
    aggregate_pm.observe(on_toggle_change, names='value')
    aggregate_di.observe(on_toggle_change, names='value')
    agg_kind.observe(on_toggle_change, names='value')
    percentile.observe(redraw, names='value')
    contour_on.observe(redraw, names='value')
    contour_level.observe(redraw, names='value')

    update_visibility()

    display(widgets.VBox([
        widgets.HBox([axis_x, axis_y]),
        widgets.HBox([fix_widgets['im'], fix_widgets['cm'], fix_widgets['lm']]),
        widgets.HBox([aggregate_pm, fix_widgets['pm']]),
        widgets.HBox([aggregate_di, fix_widgets['di'], agg_kind, percentile]),
        widgets.HBox([contour_on, contour_level]),
        out,
    ]))

    redraw()