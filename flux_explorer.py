"""flux_explorer.py — interactive PAE-vs-flux explorer with series support.

Plots PAE on the y-axis against pm (flux per plaquette) on the x-axis.
For each of the three other parameters — disorder index, Ic-multiplier,
inductance-multiplier — the caller picks either a fixed value (to filter
the dataframe) or marks that parameter as the "series" (used as the hue
color of multiple lines on a single plot). At most one parameter can be
the series; the others must be fixed.

This module exports two functions:

  - ``plot_pae_vs_flux(df, fix, series=None, ...)`` — pure plotting
    function. No widgets. Useful for programmatic / scripted use,
    embedding in another figure, or in batch reports.
  - ``make_flux_explorer(df, use_design_params=False, palette=...)`` —
    interactive widget that builds the dropdowns and calls
    ``plot_pae_vs_flux`` under the hood.

Pass ``use_design_params=True`` to the widget version to filter by the
physical design quantities ``designIc`` and ``designBetaPrime`` instead
of the multipliers ``im`` and ``lm``. The DataFrame must then carry
those columns.

Usage in a Jupyter notebook:

    from flux_explorer import make_flux_explorer, plot_pae_vs_flux
    make_flux_explorer(df, use_design_params=True)

    # or, programmatically:
    plot_pae_vs_flux(df, fix={'di': 0, 'lm': 0.94}, series='im')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output


PRECISION = 6

# Display overrides for parameter names and value scaling. When a column
# appears as a dropdown, hue label, or title token, the printed label is
# taken from DISPLAY_NAMES and printed numeric values are multiplied by
# DISPLAY_SCALES. The underlying DataFrame columns are unchanged.
DISPLAY_NAMES = {
    'designIc':        'Ic [\u03BCA]',     # μA
    'designBetaPrime': "\u03B2'",          # β'
}
DISPLAY_SCALES = {
    'designIc': 1e6,                       # amps → microamps
}


def _display_name(p):
    return DISPLAY_NAMES.get(p, p)


def _display_scale(p):
    return DISPLAY_SCALES.get(p, 1.0)


def _round_sig(values, sig=PRECISION):
    """Round each value to ``sig`` significant figures (not decimal places)."""
    arr = np.asarray(values, dtype=float)
    out = arr.copy()
    nz = arr != 0
    if nz.any():
        n = sig - 1 - np.floor(np.log10(np.abs(arr[nz]))).astype(int)
        out[nz] = np.round(arr[nz] * 10.0**n) / 10.0**n
    return out


def _round_unique(series):
    return sorted(np.unique(_round_sig(series.to_numpy())))


# ---------------------------------------------------------------------------
# Pure plotting function
# ---------------------------------------------------------------------------
def plot_pae_vs_flux(df, fix=None, series=None,
                    *,
                    ax=None,
                    palette='coolwarm',
                    legend=True,
                    max_legend_entries=10,
                    xlabel='Flux per plaquette',
                    ylabel='PAE',
                    title=None):
    """Plot PAE as a function of pm. Pure function — no widgets.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'pm', 'PAE', and every column named in `fix` and
        (if given) `series`.
    fix : dict[str, float], optional
        Filter the dataframe to rows where each named column equals
        (within floating-point tolerance) the given value. A key whose
        column matches `series` is ignored (so you can pass the same
        `fix` dict regardless of which column is the series).
    series : str, optional
        Column name to use as the hue (color) variable. One line is
        drawn per unique value. If None, a single line is drawn.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes. If None, a new figure is created.
    palette : str
        seaborn / matplotlib palette name for the hue gradient.
    legend : bool
        Show the hue legend (only meaningful when `series` is given).
    max_legend_entries : int or None, default 10
        If the series has more than this many unique values, the legend
        is subsampled to this many evenly-spaced entries. All curves are
        still drawn; only the legend is thinned. Pass ``None`` to
        disable subsampling (label every curve).
    xlabel, ylabel : str
        Axis labels.
    title : str, optional
        Plot title. If None, a title is built from the fixed-parameter
        values (those not used as the series).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    if fix is None:
        fix = {}

    # Filter on every fixed column except the one used as series.
    atol = 10 ** -PRECISION
    mask = np.ones(len(df), dtype=bool)
    for col, val in fix.items():
        if col == series:
            continue
        if col not in df.columns:
            raise KeyError(f'fix references unknown column: {col!r}')
        mask &= np.isclose(df[col], val, atol=atol)
    sel = df[mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if sel.empty:
        ax.text(0.5, 0.5, 'no matching data',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='gray')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    if series is not None:
        if series not in df.columns:
            raise KeyError(f'series references unknown column: {series!r}')
        # Build hue labels: sig-fig-rounded native values, scaled to display
        # units, formatted to 4 sig figs. Categorical hue with hue_order
        # gives consistent color → value mapping when the same value appears
        # across redraws.
        scale = _display_scale(series)
        native_unique = _round_unique(sel[series])
        hue_labels = [f'{v * scale:.4g}' for v in
                      _round_sig(sel[series].to_numpy())]
        hue_order  = [f'{v * scale:.4g}' for v in native_unique]
        sns.lineplot(data=sel, x='pm', y='PAE',
                     hue=hue_labels, hue_order=hue_order,
                     palette=palette, marker='o', errorbar=None, ax=ax)
        if legend and ax.get_legend():
            handles, labels = ax.get_legend_handles_labels()
            # Subsample the legend if there are too many entries. All
            # lines are still drawn; only the legend labels are thinned.
            if (max_legend_entries is not None
                    and len(labels) > max_legend_entries):
                idx = np.linspace(0, len(labels) - 1,
                                  max_legend_entries, dtype=int)
                handles = [handles[i] for i in idx]
                labels  = [labels[i]  for i in idx]
            ax.legend(handles, labels,
                      title=_display_name(series),
                      bbox_to_anchor=(1.02, 1), loc='upper left')
        elif not legend and ax.get_legend():
            ax.get_legend().remove()
    else:
        sns.lineplot(data=sel, x='pm', y='PAE',
                     marker='o', errorbar=None, color='black', ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        parts = [
            f'{_display_name(col)}={val * _display_scale(col):.4g}'
            for col, val in fix.items()
            if col != series
        ]
        title = '  '.join(parts)
    if title:
        ax.set_title(title)

    return ax


# ---------------------------------------------------------------------------
# Interactive widget
# ---------------------------------------------------------------------------
def make_flux_explorer(df, use_design_params=False, palette='coolwarm'):
    """Build and display the interactive PAE-vs-flux explorer.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'pm', 'PAE', 'di', and either 'im' & 'lm' (when
        ``use_design_params=False``) or 'designIc' & 'designBetaPrime'
        (when ``use_design_params=True``).
    use_design_params : bool, default False
        Swap the multipliers ``im`` and ``lm`` for the physical design
        quantities ``designIc`` and ``designBetaPrime`` as the selectable
        parameters.
    palette : str
        seaborn / matplotlib palette for the hue.
    """
    if use_design_params:
        ic_col, beta_col = 'designIc', 'designBetaPrime'
    else:
        ic_col, beta_col = 'im', 'lm'
    params = ['di', ic_col, beta_col]

    required = params + ['pm', 'PAE']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'df is missing required columns: {missing}')

    # Series selector: which parameter (if any) is the hue.
    series_choice = widgets.Dropdown(
        options=[('(none — single line)', None)] +
                [(_display_name(p), p) for p in params],
        value=None,
        description='series:',
    )

    # Per-parameter value dropdowns. Each one is hidden whenever its
    # parameter is currently selected as the series.
    value_widgets = {
        p: widgets.Dropdown(
            options=[(f'{v * _display_scale(p):.4g}', v)
                     for v in _round_unique(df[p])],
            description=f'{_display_name(p)}:',
        )
        for p in params
    }

    out = widgets.Output()

    def update_visibility():
        s = series_choice.value
        for p in params:
            value_widgets[p].layout.display = 'none' if p == s else ''

    def redraw(*_):
        with out:
            clear_output(wait=True)
            s = series_choice.value
            fix = {p: value_widgets[p].value for p in params}
            plt.close('all')
            plot_pae_vs_flux(df, fix=fix, series=s, palette=palette)
            plt.tight_layout()
            plt.show()

    def on_series_change(*_):
        update_visibility()
        redraw()

    series_choice.observe(on_series_change, names='value')
    for w in value_widgets.values():
        w.observe(redraw, names='value')

    update_visibility()
    display(widgets.VBox([
        widgets.HBox([series_choice]),
        widgets.HBox([value_widgets[p] for p in params]),
        out,
    ]))
    redraw()