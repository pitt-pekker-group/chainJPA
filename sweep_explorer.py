"""sweep_explorer.py — side-by-side pump/signal sweep explorer.

Expects a pandas DataFrame with columns 'di', 'im', 'cm', 'lm', 'pm',
'sweep_pump', and 'sweep_signal'. Each sweep column holds n-by-2 numpy
arrays: column 0 is the swept amplitude, column 1 is the linear gain.

For each (di, im, cm, lm) slice, draws two log-x plots side-by-side
showing gain in dB vs amplitude — one for the pump sweep, one for the
signal sweep — with one color per pm value. Colors are mapped from a
shared normalization so identical pm values render in the same color in
both plots even if their data ranges differ.

Pass ``use_design_params=True`` to filter by the physical design
quantities ``designIc`` (in place of ``im``) and ``designBetaPrime``
(in place of ``lm``) — the DataFrame must then carry those columns.

Usage in a Jupyter notebook:

    from sweep_explorer import make_sweep_explorer
    make_sweep_explorer(df)
    make_sweep_explorer(df, use_design_params=True)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output


PRECISION = 6
PARAMS = ['di', 'im', 'cm', 'lm']
REQUIRED_COLS = PARAMS + ['pm', 'sweep_pump', 'sweep_signal']

# Display overrides. When a column appears as a dropdown or in the
# suptitle, the printed label is taken from DISPLAY_NAMES and printed
# numeric values are multiplied by DISPLAY_SCALES. The underlying
# DataFrame columns are unchanged.
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
    """Round each value to ``sig`` significant figures (not decimal places).

    Rounding to a fixed number of decimal places destroys precision for
    very small values: round(1.884e-5, 6) is 1.9e-5. Rounding to a fixed
    number of significant figures preserves precision regardless of
    magnitude, which matters for values like Ic that are stored in SI
    units (~1e-5 amps) but displayed in μA.
    """
    arr = np.asarray(values, dtype=float)
    out = arr.copy()
    nz = arr != 0
    if nz.any():
        n = sig - 1 - np.floor(np.log10(np.abs(arr[nz]))).astype(int)
        out[nz] = np.round(arr[nz] * 10.0**n) / 10.0**n
    return out


def _round_unique(series):
    return sorted(np.unique(_round_sig(series.to_numpy())))


def _prepare_long(sel, col):
    """Convert valid n-by-2 sweep arrays in `sel[col]` into long-format
    with the y-axis converted to dB."""
    valid = sel[sel[col].apply(
        lambda a: isinstance(a, np.ndarray) and a.ndim == 2 and a.shape[1] == 2
    )]
    if valid.empty:
        return None
    chunks = []
    for _, r in valid.iterrows():
        a = r[col]
        y_lin = a[:, 1]
        # 10*log10 for power gain; use 20*log10 for amplitude/voltage gain.
        y_dB = np.where(y_lin > 0, 10 * np.log10(y_lin), np.nan)
        chunks.append(pd.DataFrame({'x': a[:, 0], 'gain_dB': y_dB,
                                    'pm': r['pm']}))
    return pd.concat(chunks, ignore_index=True)


def make_sweep_explorer(df,
                        pump_ref_lines=(20.0,),
                        signal_ref_lines=(19.0, 20.0, 21.0),
                        pump_ref_xrange=(1.0, 80.0),
                        signal_ref_xrange=(1e-4, 1.0),
                        palette='coolwarm',
                        use_design_params=False):
    """Build and display the side-by-side sweep explorer.

    Parameters
    ----------
    df : pandas.DataFrame
        Sweep results with columns 'di', 'im', 'cm', 'lm', 'pm',
        'sweep_pump', 'sweep_signal' (or 'designIc' / 'designBetaPrime'
        in place of 'im' / 'lm' when ``use_design_params=True``).
    pump_ref_lines, signal_ref_lines : iterable of float
        Y-values (in dB) at which to draw horizontal reference lines on
        each subplot. Defaults: pump = (20,) drawn bold; signal =
        (19, 20, 21) drawn as thin guide lines.
    pump_ref_xrange, signal_ref_xrange : (xmin, xmax)
        Horizontal extent of the reference lines on each subplot.
    palette : str
        seaborn / matplotlib colormap name for the pm color encoding.
    use_design_params : bool, default False
        If True, swap the multipliers ``im`` and ``lm`` for the physical
        design quantities ``designIc`` and ``designBetaPrime``. The
        DataFrame must then contain those columns.
    """
    # Locally rebind the parameter list so the closures pick up whichever
    # set is active. Module-level constants are not affected.
    if use_design_params:
        PARAMS        = ['di', 'designIc', 'cm', 'designBetaPrime']
        REQUIRED_COLS = PARAMS + ['pm', 'sweep_pump', 'sweep_signal']
    else:
        PARAMS        = ['di', 'im', 'cm', 'lm']
        REQUIRED_COLS = PARAMS + ['pm', 'sweep_pump', 'sweep_signal']

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f'df is missing required columns: {missing}')

    # Widgets — dropdown labels and shown values reflect DISPLAY_NAMES /
    # DISPLAY_SCALES; underlying .value stays in native units.
    sel_widgets = {
        p: widgets.Dropdown(
            options=[(f'{v * _display_scale(p):.4g}', v) for v in _round_unique(df[p])],
            description=f'{_display_name(p)}:',
        )
        for p in PARAMS
    }
    out = widgets.Output()

    def redraw(*_):
        with out:
            clear_output(wait=True)

            atol = 10 ** -PRECISION
            mask = np.ones(len(df), dtype=bool)
            for p in PARAMS:
                mask &= np.isclose(df[p], sel_widgets[p].value, atol=atol)
            sel = df[mask]

            long_pump   = _prepare_long(sel, 'sweep_pump')
            long_signal = _prepare_long(sel, 'sweep_signal')

            if long_pump is None and long_signal is None:
                print('no matching data')
                return

            # Shared color normalization: union of pm values from both sweeps,
            # so identical pm renders as identical color in both plots.
            pm_vals = []
            if long_pump is not None:
                pm_vals.extend(long_pump['pm'].unique())
            if long_signal is not None:
                pm_vals.extend(long_signal['pm'].unique())
            pm_min, pm_max = min(pm_vals), max(pm_vals)
            if pm_min == pm_max:        # single unique pm — avoid degenerate norm
                pm_max = pm_min + 1.0
            hue_norm = (pm_min, pm_max)

            plt.close('all')
            fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(13, 4.5))

            # One shared legend, on the right plot if it has data.
            legend_host = ax_s if long_signal is not None else ax_p

            # Pump sweep (left)
            if long_pump is not None:
                sns.lineplot(data=long_pump, x='x', y='gain_dB',
                             hue='pm', palette=palette, hue_norm=hue_norm,
                             legend=('brief' if legend_host is ax_p else False),
                             ax=ax_p,
                             marker='o', linewidth=0.1, markersize=3)
            for y in pump_ref_lines:
                ax_p.plot(pump_ref_xrange, [y, y], '-k')
            ax_p.set_xlabel('pump amplitude')
            ax_p.set_ylabel('gain (dB)')
            ax_p.set_xscale('log')
            ax_p.set_title('pump sweep')

            # Signal sweep (right)
            if long_signal is not None:
                sns.lineplot(data=long_signal, x='x', y='gain_dB',
                             hue='pm', palette=palette, hue_norm=hue_norm,
                             legend=('brief' if legend_host is ax_s else False),
                             ax=ax_s,
                             marker='o', linewidth=0.1, markersize=3)
            for y in signal_ref_lines:
                ax_s.plot(signal_ref_xrange, [y, y], '-k', linewidth=0.1)
            ax_s.set_xlabel('signal amplitude')
            ax_s.set_ylabel('gain (dB)')
            ax_s.set_xscale('log')
            ax_s.set_title('signal sweep')

            if legend_host.get_legend():
                legend_host.legend(title='pm',
                                   bbox_to_anchor=(1.02, 1), loc='upper left')

            fig.suptitle('  '.join(
                f'{_display_name(p)}={sel_widgets[p].value * _display_scale(p):.4g}'
                for p in PARAMS
            ))
            plt.tight_layout()
            plt.show()

    for w in sel_widgets.values():
        w.observe(redraw, names='value')

    display(widgets.VBox([
        widgets.HBox(list(sel_widgets.values())),
        out,
    ]))

    redraw()