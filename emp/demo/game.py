from shiny import module, ui, reactive, render
from shinywidgets import output_widget, render_widget
from plotly.subplots import make_subplots
import numpy as np
from ..metrics.maintenance import calculate_PRMC
import random
import scipy.stats as st
from functools import partial

leadtime_dist = partial(st.lognorm.rvs, s=1.2, scale=6)

@module.ui
def game_ui():
    return ui.nav('Game',
                ui.layout_sidebar(
                    ui.panel_sidebar(
                            ui.output_ui('selected_tau'),
                            ui.input_switch('show_tau', 'Show lead time'),
                            ui.HTML('<hr>'),
                            ui.input_slider('game_threshold', 'Threshold', 0, 50, 10),
                            ui.input_action_button('game_submit', 'Submit!', **{"class": 'btn btn-primary'}),
                            ui.input_action_button('game_reset_selection', 'Choose new machines!', **{"class": 'btn btn-secondary'}),
                            ui.HTML('<hr>'),
                            ui.HTML('<h5 style="text-align: center">Total Cost</h5>'),
                            ui.output_ui('score_history'),
                            ui.HTML('<hr>'),
                            ui.HTML('<h5 style="text-align: center">Last Cost</h5>'),
                            ui.output_ui('score_contents'),
                            width=3
                    ),
                    ui.panel_main(
                        output_widget('plot_game_machines'),
                    )
                )
            )

@module.server
def game_server(input, output, session, preds, trues, cost_reactive, cost_predictive, cost_rul):
    machs = reactive.Value(random.sample(range(preds.shape[0]), 4))
    submit = reactive.Value(False)

    init_leadtime = int(np.round(leadtime_dist(), 0))
    tau = reactive.Value(init_leadtime)
    prev_tau = reactive.Value(init_leadtime)

    prev_score_html = reactive.Value('')

    score_user = reactive.Value(0)
    score_cpu = reactive.Value(0)

    @output
    @render_widget  # type: ignore
    def plot_game_machines():
        fig_preds = make_subplots(
            rows=4,
            subplot_titles=list(map(lambda x: "Machine: " + str(x), machs()))
        )
        threshold = input.game_threshold()
        legend = True
        for row_num, mach in enumerate(machs()):
            row_num += 1
            if row_num > 1:  # only keep legend for first figure
                legend = False
            timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
            timestamp_failure = np.amin(np.argwhere(trues[mach, :] == 0))

            fig_preds.add_scatter(
                row=row_num,
                col=1,
                x=np.arange(0, timestamp_failure),
                y=preds[mach, 0:timestamp_failure],
                name="Predicted RUL",
                line={
                    "color": 'blue'
                },
                showlegend=legend
            )
            last_pred_rul = [preds[mach, timestamp_failure]]
            last_pred_rul.extend(list(np.repeat(0, 100)))
            fig_preds.add_scatter(
                row=row_num,
                col=1,
                x=np.arange(timestamp_failure - 1, timestamp_failure + 100),
                y=last_pred_rul,
                line={
                    "color": 'blue'
                },
                showlegend=False,
                hoverinfo=' skip'
            )
            fig_preds.add_scatter(
                row=row_num,
                col=1,
                x=[0, np.amin(np.argwhere(preds[mach, :] <= threshold))],
                y=[threshold, threshold],
                mode='lines+text',
                line={
                    "width": 2,
                    "color": 'purple',
                    "dash": 'dot'
                },
                text=["Threshold", ""],
                textposition='top right',
                textfont={
                    "color": 'purple'
                },
                name='Threshold',
                showlegend=legend
            )
            fig_preds.add_scatter(
                row=row_num,
                col=1,
                x=[timestamp_maintenance, timestamp_maintenance],
                y=[0, max(trues[mach, :])],
                line={
                    "color": 'black'
                },
                mode='lines',
                name="Moment of Maintenance",
                showlegend=legend
            )
            fig_preds.add_vline(
                row=row_num,  # type: ignore
                col=1,  # type: ignore
                x=timestamp_maintenance,
                line_color="black",
            )
            if input.show_tau():
                fig_preds.add_vline(
                    row=row_num,
                    col=1,
                    x=timestamp_maintenance + tau(),
                    line_color='orange'
                )
            fig_preds.update_xaxes({'range': [0, timestamp_failure]}, row=row_num, col=1)
        # update layout
        fig_preds.update_layout(
            **{
                "height": 800,
                "hovermode": "x unified",
                "title_text": "Simulation RUL for four machines.",
                "title_x": 0.5,
                "title_font_size": 25,
            }
        )
        return fig_preds

    @reactive.Effect
    @reactive.event(input.game_reset_selection)
    def game_reset_selection():
        with reactive.isolate():
            machs.set(random.sample(range(preds.shape[0]), 4))

    @reactive.Effect
    @reactive.event(input.game_submit)
    def submit_true():
        submit.set(True)

    @reactive.Effect
    def submit_slider_change():
        input.game_threshold()  # called here so the def reacts on changes in the slider
        with reactive.isolate():
            submit.set(False)

    @output
    @render.ui
    def selected_tau():
        if submit():
            with reactive.isolate():
                prev_tau.set(tau())
            tau.set(int(np.round(leadtime_dist(), 0)))
        return ui.HTML(f'<h3>Current lead time: <a style="color:red;font-size:1.25em"> {tau()}</a></h3>')

    @output
    @render.ui
    def score_contents():
        with reactive.isolate():
            thresholds = np.linspace(0, 50, 51)
            costs = [np.round(np.sum(calculate_PRMC(preds[machs(), :], trues[machs(), :], prev_tau(), ti, cost_reactive, cost_predictive, cost_rul[machs()])), 2) for ti in thresholds]
            optimal_idx = np.argmin(costs)
            cost_optimal = np.round(costs[optimal_idx], 2)
            optimal_threshold = thresholds[optimal_idx]
            optimal_threshold = 15
            cost_player = np.round(np.sum(calculate_PRMC(preds[machs(), :], trues[machs(), :], prev_tau(), input.game_threshold(), cost_reactive, cost_predictive, cost_rul[machs()])), 2)

        if cost_optimal < cost_player:
            player_layout = 'class="table-danger"'
            cpu_layout = 'class="table-success"'
        elif cost_optimal == cost_player:
            player_layout = 'class="table-success"'
            cpu_layout = 'class="table-warning"'
        else:
            player_layout = 'class="table-success"'
            cpu_layout = 'class="table-danger"'

        if submit():
            with reactive.isolate():
                score_cpu.set(np.round(score_cpu() + cost_optimal, 2))
                score_user.set(np.round(score_user() + cost_player, 2))
                prev_score_html.set(
                            '<table class="table table-hover">'
                            f'<tr><th style="text-align:left">Lead time: {prev_tau()}</th><th>Threshold</th><th>Cost</th></tr>'
                            f'<tr {player_layout} ><td style="text-align:left">Your choice</td><td>{input.game_threshold()}</td><td>{cost_player}</td></tr>'
                            f'<tr {cpu_layout} ><td style="text-align:left">Optimum</td><td>{int(optimal_threshold)}</td><td>{cost_optimal}</td></tr>'
                            '</table>'
                )
                # new machines:
                machs.set(random.sample(range(preds.shape[0]), 4))
                return ui.HTML(prev_score_html())
        else:
            with reactive.isolate():
                return ui.HTML(prev_score_html())

    @output
    @render.ui
    def score_history():
        return ui.HTML(
            '<table class="table table-hover">'
            '<tr><th></th><th class="table-primary">CPU</th><th class="table-primary">PLAYER</th></tr>'
            f'<tr><th class="table-primary" style="text-align:left">Cost</th><td class="table-secondary">{score_cpu()}</td><td class="table-secondary">{score_user()}</td></tr>'
        )
