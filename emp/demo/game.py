from shiny import module, ui, reactive, render
import shiny.experimental as x
from shinywidgets import output_widget, render_widget
from plotly.subplots import make_subplots
import numpy as np
from ..metrics.maintenance import calculate_PRMC
import random
import scipy.stats as st
from functools import partial

leadtime_dist = partial(st.lognorm.rvs, s=1.2, scale=6)

tooltip_icon = ui.HTML('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-question-octagon" viewBox="0 0 16 16">'
    '<path d="M4.54.146A.5.5 0 0 1 4.893 0h6.214a.5.5 0 0 1 .353.146l4.394 4.394a.5.5 0 0 1 .146.353v6.214a.5.5 0 0 1-.146.353l-4.394 4.394a.5.5 0 0 1-.353.146H4.893a.5.5 0 0 1-.353-.146L.146 11.46A.5.5 0 0 1 0 11.107V4.893a.5.5 0 0 1 .146-.353L4.54.146zM5.1 1 1 5.1v5.8L5.1 15h5.8l4.1-4.1V5.1L10.9 1H5.1z"/>'
    '<path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286zm1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94z"/>'
    '</svg>'
)

@module.ui
def game_ui():
    return ui.nav('Game',
                ui.layout_sidebar(
                    ui.panel_sidebar(
                            ui.HTML(
                                "<script>"
                                "$(document).ready(function(){"
                                "$('.rules-heading').on('click', function(){"
                                    "$('.rules-body').toggle('fast');"
                                    "$('.rules-heading.expanded').toggle('fast');"
                                    "$('.rules-heading.contracted').toggle('fast');"
                                    "});"
                                "$('.btn.game-select-threshold').on('click', function(){"
                                    "$('.btn.game-restart').toggle();"
                                    "$('.btn.game-select-threshold').toggle();"
                                    "});"
                                "$('.btn.game-restart').on('click', function(){"
                                    "$('.btn.game-select-threshold').toggle();"
                                    "$('.btn.game-restart').toggle();"
                                    "});"
                                "});"
                                "</script>"
                            ),
                            ui.div(
                                ui.HTML(
                                    '<button class="btn btn-secondary" style="width:100%;padding-top:0px;padding-bottom:0px;">'
                                    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-compact-up" viewBox="0 0 16 16">'
                                    '<path fill-rule="evenodd" d="M7.776 5.553a.5.5 0 0 1 .448 0l6 3a.5.5 0 1 1-.448.894L8 6.56 2.224 9.447a.5.5 0 1 1-.448-.894l6-3z"/>'
                                    '</svg>'
                                    ' What is the objective?'
                                    '</button>'
                                ),
                                {
                                    "class": "rules-heading expanded",
                                    "style": "cursor:pointer;display:none;max-height:25px;",
                                }
                            ),
                            ui.div(
                                ui.HTML(
                                    '<button class="btn btn-info" style="width:100%;padding-top:0px;padding-bottom:0px;">'
                                    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-chevron-compact-down" viewBox="0 0 16 16">'
                                    '<path fill-rule="evenodd" d="M1.553 6.776a.5.5 0 0 1 .67-.223L8 9.44l5.776-2.888a.5.5 0 1 1 .448.894l-6 3a.5.5 0 0 1-.448 0l-6-3a.5.5 0 0 1-.223-.67z"/>'
                                    '</svg>'
                                    ' What is the objective?'
                                    '</button>'
                                ),
                                {
                                    "class": "rules-heading contracted",
                                    "style": "cursor:pointer;max-height:25px;"
                                }
                            ),
                            ui.div(
                                ui.h1(
                                    'Maintenance game'
                                ),
                                ui.p(
                                    'In the game you have to schedule maintenance for 4 machines at a time. Then, for these machines the cost of maintenance is calculated based on your selected time. '
                                    'Selecting the desired moment of maintenance is done by choosing a threshold, as soon as the predicted RUL is smaller than or equal to this threshold, maintenance is scheduled. '
                                    'After submitting your choice, 4 new machines are shown together with a new value for the lead time. '
                                ),
                                ui.h5(
                                    'Selecting a good threshold'
                                ),
                                ui.HTML(
                                    '<p>'
                                    'When setting the threshold, the current lead times should be held in to account. When it takes spareparts 15 days to be delivered, it would not make sense to set the threshold '
                                    'lower than 15 days<sup>1</sup>. Conducting maintenance too early (i.e. a high threshold) will mean that a lot of RUL is not used (i.e. high opportunity cost). On the other hand, '
                                    'Conducting maintenance too late (i.e. low threshold) will mean more machine failures (i.e. high cost of repair). So finding the optimal threshold means finding a middle ground between these two.'
                                    '</p>'
                                    '</br>'
                                    '<p style="font-size:0.9em"><i>1. Of course, as predictions are not perfect, it could still be the case that the optimal threshold is lower than the lead time. In practice, however, this is rarely the case and remains non-sensical.</i></p>'
                                ),
                                {
                                    "class": "rules-body",
                                    "style": "display:none;"
                                }
                            ),
                            ui.div(
                                ui.output_ui('selected_tau'),
                                x.ui.tooltip(
                                    ui.div(
                                        tooltip_icon,
                                        {"style": "margin:0.9em 0em 0em 0.4em"}
                                    ),
                                    'Time between scheduling maintenance and the maintenance actually happening. Keep this in mind when choosing the threshold.',
                                ),
                                {'style': "display:inline-flex;"}
                            ),
                            ui.HTML('<hr>'),
                            ui.input_slider(
                                'game_threshold',
                                ui.div(
                                    ui.h5('Threshold (in days)'),
                                    x.ui.tooltip(
                                        ui.div(
                                            tooltip_icon,
                                            {"style": "margin:0em 0em 0.15em 0.4em"}
                                        ),
                                        'Set the threshold. This decides when maintenance is scheduled. (y-axis)'
                                    ),
                                    {'style': "display:inline-flex;"}
                                ),
                                0,
                                50,
                                10
                            ),
                            ui.input_action_button('game_submit', 'Select Threshold', **{"class": 'btn btn-primary game-select-threshold'}),
                            ui.input_action_button('game_reset_selection', 'Start new round!', **{"class": 'btn btn-secondary game-restart', "style": "display:none"}),
                            ui.HTML('<hr>'),
                            ui.div(
                                ui.HTML('<h5 style="text-align: center">Cost of last round</h5>  (K€)'),
                                x.ui.tooltip(
                                    ui.div(
                                        tooltip_icon,
                                        {'style': "margin:0em 0em 0.8em 0.5em"}
                                    ),
                                    'Cost of last round.'
                                ),
                                {'style': "display:inline-flex;margin: 0% 0% 0% 15%"}
                            ),
                            ui.output_ui('score_contents'),
                            ui.HTML('<hr>'),
                            ui.div(
                                ui.HTML('<h5 style="text-align: center">Accumulated Cost</h5> (K€)'),
                                x.ui.tooltip(
                                    ui.div(
                                        tooltip_icon,
                                        {'style': "margin:0em 0em 0.8em 0.5em"}
                                    ),
                                    'Total cost over all rounds played. (cumulative)'
                                ),
                                {'style': "display:inline-flex;margin: 0% 0% 0% 15%"}
                            ),
                            ui.output_ui('score_history'),
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
    while init_leadtime > 30:
        init_leadtime = int(np.round(leadtime_dist(), 0))
    tau = reactive.Value(init_leadtime)
    prev_tau = reactive.Value(init_leadtime)

    prev_score_html = reactive.Value('')

    score_user = reactive.Value(0)
    score_cpu = reactive.Value(0)

    @reactive.Effect
    @reactive.event(input.game_submit)
    def submit_true():
        submit.set(True)

    @reactive.Effect
    @reactive.event(input.game_reset_selection)
    def game_reset_selection():
        with reactive.isolate():
            machs.set(random.sample(range(preds.shape[0]), 4))
            tau.set(int(np.round(leadtime_dist(), 0)))
            while tau() > 30:
                tau.set(int(np.round(leadtime_dist(), 0)))

            submit.set(False)

    @output
    @render.ui
    def selected_tau():
        with reactive.isolate():
            prev_tau.set(tau())
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
                name="Estimated RUL",
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
                name="Moment of scheduling maintenance",
                showlegend=legend
            )
            fig_preds.add_vline(
                row=row_num,  # type: ignore
                col=1,  # type: ignore
                x=timestamp_maintenance,
                line_color="black",
            )
            fig_preds.update_yaxes(
                title_text="Estimated RUL",
                row=row_num,
                col=1
            )
            if row_num == 4:  # xlabel only for last subplot
                fig_preds.update_xaxes(
                    title_text="Days",
                    title_font_size=30,
                    row=row_num,
                    col=1
                )
            with reactive.isolate():
                fig_preds.add_vline(
                    row=row_num,
                    col=1,
                    x=timestamp_maintenance + tau(),
                    line_color='orange'
                )
                fig_preds.add_scatter(
                    row=row_num,
                    col=1,
                    x=[timestamp_maintenance + tau(), timestamp_maintenance + tau()],
                    y=[0, max(trues[mach, :])],
                    mode='lines',
                    name='Moment of actual maintenance',
                    line_color='orange',
                    showlegend=legend
                )
            fig_preds.update_xaxes({'range': [0, timestamp_failure]}, row=row_num, col=1)
        # update layout
        fig_preds.update_layout(
            **{
                "height": 800,
                "hovermode": "x unified",
                "title_text": "Select a threshold to plan maintenance",
                "title_x": 0.5,
                "title_font_size": 25,
            }
        )
        return fig_preds

