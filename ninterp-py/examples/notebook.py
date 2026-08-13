import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import ninterp
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, ninterp, np


@app.cell
def _(np):
    # f(x) = sin(pi * x)
    def f(x):
        return np.sin(np.pi * x)

    return (f,)


@app.cell
def _(mo):
    get_x_points, set_x_points = mo.state([0.0, 0.2, 0.5, 0.8, 1.3, 1.6, 2.0])
    return get_x_points, set_x_points


@app.cell
def _(get_x_points, mo, set_x_points):
    def _add_point():
        pts = get_x_points()
        new_x = pts[-1] + 0.2 if pts else 0.0
        set_x_points(lambda v: v + [new_x])

    def _remove_point(i):
        set_x_points(lambda v: v[:i] + v[i + 1 :])

    def _update_point(i, new_val):
        set_x_points(lambda v: [new_val if j == i else xv for j, xv in enumerate(v)])

    add_point_button = mo.ui.button(label="+ add point", on_change=lambda _: _add_point())

    _can_remove = len(get_x_points()) > 2
    x_point_rows = [
        mo.hstack(
            [
                mo.ui.number(
                    value=xv,
                    step=0.05,
                    label=f"x{i}",
                    on_change=lambda v, i=i: _update_point(i, v),
                ),
                mo.ui.button(
                    label="×",
                    disabled=not _can_remove,
                    on_change=lambda _, i=i: _remove_point(i),
                ),
            ],
            justify="start",
            gap=0.5,
        )
        for i, xv in enumerate(get_x_points())
    ]
    return add_point_button, x_point_rows


@app.cell
def _(mo):
    strategy_dropdown = mo.ui.dropdown(
        options=[
            "Linear",
            "Nearest",
            "Step (lower)",
            "Step (upper)",
            "Cubic (natural)",
            "Cubic (not-a-knot)",
        ],
        value="Linear",
        label="Strategy",
    )
    return (strategy_dropdown,)


@app.cell
def _(add_point_button, mo, strategy_dropdown, x_point_rows):
    controls = mo.vstack(
        [strategy_dropdown, add_point_button, *x_point_rows],
        gap=1,
    )
    return (controls,)


@app.cell
def _(f, get_x_points, ninterp, np, strategy_dropdown):
    _strategies = {
        "Linear": ninterp.Linear,
        "Nearest": ninterp.Nearest,
        "Step (lower)": ninterp.Step.lower,
        "Step (upper)": ninterp.Step.upper,
        "Cubic (natural)": ninterp.CubicC2.natural,
        "Cubic (not-a-knot)": ninterp.CubicC2.not_a_knot,
    }
    x = np.sort(np.array(get_x_points()))
    f_x = f(x)

    try:
        interp = ninterp.Interpolator.new_1d(x, f_x, _strategies[strategy_dropdown.value]())
    except ValueError as e:
        interp = None
        interp_error = str(e)
    else:
        interp_error = None
    return f_x, interp, interp_error, x


@app.cell
def _(controls, f, f_x, go, interp, interp_error, mo, np, strategy_dropdown, x):
    if interp is None:
        plot = mo.md(f"**Error:** {interp_error}")
    else:
        try:
            plot_x = np.linspace(x.min(), x.max(), 200)
            plot_f_x = np.array([interp.interpolate(xi) for xi in plot_x])
            true_f_x = f(plot_x)

            fig = go.Figure()
            fig.add_scatter(x=plot_x, y=plot_f_x, mode="lines", name=strategy_dropdown.value)
            fig.add_scatter(
                x=plot_x, y=true_f_x, mode="lines", name="f(x)", line=dict(dash="dash")
            )
            fig.add_scatter(x=x, y=f_x, mode="markers", name="data points", marker=dict(size=9))
            plot = mo.ui.plotly(fig)
        except Exception as e:
            # Keep controls interactive even if plotting fails (e.g. the
            # in-progress interp_diag -> interpolate rename isn't wired up
            # on the built extension yet).
            plot = mo.md(f"**Plot error:** {type(e).__name__}: {e}")

    mo.hstack([controls, plot], justify="start", align="start", gap=2)
    return


if __name__ == "__main__":
    app.run()
