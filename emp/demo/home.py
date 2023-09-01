from shiny import module, ui, reactive, render
from shinywidgets import output_widget, render_widget, register_widget
import plotly.graph_objects as go
import numpy as np
from plotly import colors
import pandas as pd
from ..metrics.maintenance import calculate_PRMC

@module.ui
def home_ui():
    return ui.nav('Home',
            ui.panel_main(
                output_widget('preds_plot'),
                ui.input_slider('threshold', 'Threshold', min=0, max=150, value=10),
                ui.output_ui("contents")
            )
        )

@module.server
def home_server(input, output, session, preds, trues, cost_reactive, cost_predictive, cost_rul):

    @output
    @render_widget
    def preds_plot():
        mach = 74
        fig_preds = go.Figure(layout={
            "hovermode": "x"
            })
        fig_preds.add_scatter(
            x=np.arange(0, preds.shape[1]),
            y=preds[mach, :],
            name="Predictions"
            )
        fig_preds.add_scatter(
            x=np.arange(0, trues.shape[1]),
            y=trues[mach, :],
            name="Ground truth",
            mode='lines'
            )
        threshold = input.threshold()
        timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
        fig_preds.add_scatter(
            x=[timestamp_maintenance, timestamp_maintenance],
            y=[0, max(trues[mach,:])],
            line={
                "color": 'black'
            },
            mode='lines',
            name="Moment of Maintenance"
        )
        fig_preds.add_vline(
            x=timestamp_maintenance,
            line_color="black"
        )
        return fig_preds

    @output
    @render.ui
    def contents():
        mach = 74
        tau = 12
        threshold = input.threshold()
        cost = np.sum(calculate_PRMC(preds[mach,:], trues[mach,:], tau, threshold, cost_reactive, cost_predictive, cost_rul[mach]))
        return ui.HTML(f"<b>COST:</b>")
