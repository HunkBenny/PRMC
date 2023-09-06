from shiny import module, ui, render
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import numpy as np
from ..metrics.maintenance import calculate_PRMC
import scipy.stats as st
from functools import partial
import scipy.integrate as integrate

@module.ui
def uncertainty_ui(preds):
    all_choices = list(set(range(preds.shape[0])))
    all_choices.sort()
    return ui.nav(
        "Uncertainty",
        ui.row(
            ui.column(
                9,
                output_widget('rul_plot'),
            ),
            ui.column(
                3,
                ui.input_select('mach', 'Select machine', all_choices),  # type: ignore
                ui.input_slider('rul_threshold', 'Threshold', 0, 50, 10),
                style="margin-top:5%"
            )
        ),
        ui.HTML(
            '<hr>'
        ),
        ui.row(
            ui.column(
                7,
                output_widget('lead_time_distr_plot'),
            ),
            ui.column(
                5,
                ui.HTML(
                    # r'\require{amsmath,amssymb}'
                    r'$$\text{Expected maintenance cost} = E[\text{PRMC}]$$'
                    r'\[\begin{align}'
                    r'E[\text{PRMC}]&='
                    r'\begin{cases}'
                    r'E[\text{Reactive cost}] & \textbf{where } \text{lead time} > RUL \\'
                    r'+\\'
                    r'E[\text{Predictive cost}] & \textbf{where } \text{lead time} \le RUL'
                    r'\end{cases}\\'
                    r'\\'
                    r'&='
                    r'\begin{cases}'
                    r'\int_{RUL}^{\infty} h(\tau) \cdot C_r\text{ }d\tau & \textbf{where } \text{lead time} > RUL \\'
                    r'+\\'
                    r'\int_{0}^{RUL} h(\tau) \cdot [C_p + (RUL - \tau) \cdot \delta] & \textbf{where } \text{lead time} \le RUL'
                    r'\end{cases}'
                    r'\end{align}\]'
                ),
            )
        ),
        ui.row(
            ui.column(
                8,
                ui.output_ui('eprmc_formula')
            )
        ),
    )


@module.server
def uncertainty_server(input, output, session, preds, trues, cost_reactive, cost_predictive, cost_rul):
    distr_pdf = partial(st.lognorm.pdf, s=1.2, scale=6)

    @output
    @render_widget  # type: ignore
    def lead_time_distr_plot():
        mach = int(input.mach())
        threshold = input.rul_threshold()
        timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
        timestamp_failure = np.amin(np.argwhere(trues[mach, :] == 0))
        if timestamp_failure < timestamp_maintenance:
            timestamp_maintenance = timestamp_failure
        rul = int(timestamp_failure - timestamp_maintenance)
        prob_repair = np.round(st.lognorm.cdf(rul, s=1.2, scale=6), 3)

        lead_time_distr_fig = go.Figure()
        x_dist = np.linspace(0, max(50, rul + 10), 150)
        y_dist = distr_pdf(x_dist)
        lead_time_distr_fig.add_scatter(
            x=x_dist,
            y=y_dist,
            name='Lead time distribution'
        )
        # add vline for rul
        lead_time_distr_fig.add_vline(
            x=rul,
            line_color='black'
        )
        lead_time_distr_fig.add_scatter(
            x=[rul, rul],
            y=[0, max(y_dist)],
            line={
                'color': 'black'
            },
            mode='lines',
            name='RUL'
        )
# add integral up until rul
        x = list(np.linspace(0, rul, 150))
        y = list(distr_pdf(x))
        x.extend([rul, 0])
        y.extend([0, 0])
        lead_time_distr_fig.add_scatter(
            x=x,
            y=y,
            fill='toself',
            mode='none',
            fillcolor="rgba(0, 255, 0, 0.5)",
            name=f'P(t ≤ RUL) = {prob_repair}',
        )
# add integral after rul
        x = list(np.linspace(rul, max(50, rul + 10), 150))
        y = list(distr_pdf(x))
        x.extend([max(50, rul + 10), rul])
        y.extend([0, 0])
        lead_time_distr_fig.add_scatter(
            x=x,
            y=y,
            fill='toself',
            mode='none',
            fillcolor="rgba(255, 0, 0, 0.5)",
            name=f'P(t > RUL) = {np.round(1 - prob_repair, 3)}',
        )
        lead_time_distr_fig.update_layout({
            "title_text": 'Lead time distribution',
            "title_x": 0.5,
            "title_font_size": 25
        })
        return lead_time_distr_fig

    @output
    @render_widget  # type: ignore
    def rul_plot():
        mach = int(input.mach())
        distr_cdf = partial(st.lognorm.cdf, s=1.2, scale=6)
        fig_preds = go.Figure(
            layout={
                "hovermode": "x",
                "title_text": "Simulation RUL for one machine",
                "title_x": 0.5,
                "title_font_size": 25,
            }
        )
        fig_preds.add_scatter(
            x=np.arange(0, preds.shape[1]),
            y=preds[mach, :],
            name="Predicted RUL"
        )
        fig_preds.add_scatter(
            x=np.arange(0, trues.shape[1]),
            y=trues[mach, :],
            name="True RUL",
            mode='lines'
        )
        threshold = input.rul_threshold()
        timestamp_failure = np.amin(np.argwhere(trues[mach, :] == 0))
        try:
            # won't work for preds < 0 so try:except
            timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
        except ValueError:
            timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] == 0))

        if timestamp_failure < timestamp_maintenance:
            timestamp_maintenance = timestamp_failure
        rul = timestamp_failure - timestamp_maintenance
        prob = distr_cdf(rul)
        if prob == 0:
            prob = 0.00001
        timestamp_after_failure = rul / prob * (1 - prob)
        fig_preds.add_scatter(
            x=[0, np.amin(np.argwhere(preds[mach, :] <= threshold))],
            y=[threshold, threshold],
            mode='lines+text',
            line={
                "width": 2,
                "color": 'purple',
                "dash": 'dot'
            },
            text=["Threshold", ""],
            textposition='top center',
            textfont={
                "color": 'purple'
            },
            name='Threshold'
        )
        fig_preds.add_scatter(
            x=[timestamp_maintenance, timestamp_maintenance],
            y=[0, max(trues[mach, :])],
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
        # Tube lead time distribution
        # Tube before failure
        fig_preds.add_scatter(
            x=[timestamp_maintenance, timestamp_maintenance + rul, timestamp_maintenance + rul, timestamp_maintenance],
            y=[15, 15, -15, -15],
            fill='toself',
            mode='none',
            name=f'Predictive Maintenance ({np.round(prob * 100, 3)}%)',
            fillcolor="rgba(0, 255, 0, 0.5)"
        )
        # Tube after failures
        fig_preds.add_scatter(
            x=[timestamp_maintenance + rul, timestamp_maintenance + rul + timestamp_after_failure, timestamp_maintenance + rul + timestamp_after_failure, timestamp_maintenance + rul],
            y=[15, 15, -15, -15],
            fill='toself',
            mode='none',
            name=f'Reactive Maintenance ({np.round((1-prob) * 100, 3)}%)',
            fillcolor="rgba(255, 0, 0, 0.5)"
        )
        return fig_preds

    @output
    @render.ui
    def eprmc_formula():
        mach = int(input.mach())
        threshold = input.rul_threshold()
        timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
        timestamp_failure = np.amin(np.argwhere(trues[mach, :] == 0))
        if timestamp_failure < timestamp_maintenance:
            timestamp_maintenance = timestamp_failure
        rul = int(timestamp_failure - timestamp_maintenance)
        prob_repair = st.lognorm.cdf(rul, s=1.2, scale=6)
        Cp = np.round(integrate.quad(lambda tau: np.multiply(calculate_PRMC(preds[mach, timestamp_maintenance], trues[mach, timestamp_maintenance], tau, threshold, cost_reactive, cost_predictive, cost_rul[mach]), distr_pdf(tau)), 0, rul)[0], 3)
        Cr = np.round((1 - prob_repair) * cost_reactive, 3)
        return ui.HTML(
                    r'\[\begin{align}'
                    r'E[\text{PRMC}]&='
                    r'\begin{cases}'
                    r'\int_{' + str(rul) + r'}^{\infty} h(\tau) \cdot C_r\text{ }d\tau & \textbf{where } \text{lead time} > RUL \\'
                    r'+\\'
                    r'\int_{0}^{' + str(rul) + r'} h(\tau) \cdot [C_p + (' + str(rul) + r' - \tau) \cdot \delta] & \textbf{where } \text{lead time} \le RUL'
                    r'\end{cases}\\'
                    r'\\'
                    r'&='
                    r'\begin{cases}'
                    fr'\text{{€ }}{Cr} &'
                    r'\textbf{where } \text{lead time} > RUL \\'
                    r'+\\'
                    fr'\text{{€ }}{Cp} &'
                    r'\textbf{where } \text{lead time} \le RUL'
                    r'\end{cases}\\'
                    r'\\'
                    r'&='
                    fr'\text{{€ }}{Cr + Cp} \text{{ (thousands)}}'
                    r'\end{align}\]'
                    r'<script>MathJax.typeset()</script>'
                )
