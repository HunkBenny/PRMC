from shiny import module, ui, reactive, render
from shinywidgets import output_widget, render_widget, register_widget
import plotly.graph_objects as go
import numpy as np
from plotly import colors
import pandas as pd
from ..metrics.maintenance import calculate_PRMC

@module.ui
def info_ui():
    return ui.nav('Info',
                ui.div(
                    ui.row(
                        ui.column(8,
                            ui.img(src='DNM-at-Arrowsmith-scaled-2-875x625.jpg', style='width:100%'),
                                style="display:inline-flex;width:fit-content;flex: 1 2  66vw;height:fit-content;max-width:800px;"
                            ),
                        ui.column(4,
                            ui.h2('Problem Statement'),
                            ui.p('For many organisations in the manufacturing industry, maintenance costs are substantial. Therefore, it is preferable to optimize these costs. '
                                'Different sources estimate that proactive maintenance policies tend to be more cost-effective than reactive maintenance policies. ',
                                style='width:100%;word-wrap:break-word;'),
                            ui.HTML('<ol>'
                                    '<li>Preventive maintenance is purely based on time. Coming up with regular, intelligent maintenance intervals is at the basis of it all.</li>'
                                    '<li>Predictive maintenance is based on data! More and more machines are equipped with different types of sensors, which can in turn be used to predict when maintenance should be conducted.</li>'
                                    '</ol>'
                                    ),
                            style="display:block;width:fit-content;height:fit-content;flex: 2 1 34vw;",
                            ),
                        ),
                    ui.row(
                            ui.p('The focus of this demo is on predictive maintenance. Every day, for a machine, all sensor measurements are used in a Machine Learning model to come up with an estimate of the Remaining Useful Life (RUL). If the predicted RUL is small, then the ML-model predicts that the machine is going to fail in the nearby future.', style='width:100%;word-wrap:break-word;'),
                            style='display:block;width:fit-content;height:fit-content;'
                        ),
                    ui.row(
                        ui.h2('Costs of maintenance'),
                        ui.p('There are two possible cost scenarios in this demo:'),
                                ui.HTML(
                                    '<ol>'
                                    '<li>The machine fails and needs to be repaired. This is the cost of reactive maintenance.</li>'
                                    '<li>The machine is repaired before a failure occurs. This is the cost of predictive maintenance PLUS an opportunity cost.</li>'
                                    '</ol>'),
                                ui.h5('Cost of reactive maintenance', {"class": "text-primary"}),
                                ui.p(
                                    'If a machine fails, it needs to be repaired. This often happens at a premium price, as maintenance needs to occur urgently. '
                                    'Some sources state that this cost is on average three times larger than the cost of predictive maintenance.'
                                    ),
                                ui.h5('Cost of predictive maintenance', {"class": "text-primary"}),
                                ui.p(
                                    'If maintenance happens before failure, there are two costs:'
                                    ),
                                ui.HTML(
                                    '<ol>'
                                    '<li>Cost of the predictive maintenance itself.</li>'
                                    '<li>An opportunity cost: the cost of the lost RUL.</li>'
                                    '</ol>'
                                ),
                                ui.h6('Opportunity cost', {"class": "text-secondary"}),
                                ui.p(
                                    'The opportunity cost is equal to the amount of useful life which remained at the time of maintenance.'
                                    ' So, if a machine is repaired while it still had a RUL of 10 weeks, then the opportunity cost is equal '
                                    'to the lost value that could have been created in these 10 weeks. This lost value is of course different for every organisation. '
                                    'However, a baseline is based on the concept of depreciation. So, in order to calculate the cost of lost productivity '
                                    'only the acquisition price of a machine is needed.'
                                    ),
                        ),
                    ui.row(
                        ui.h2('Policy instrument'),
                        ui.p('In order to lower these costs, an organisation conducts maintenance when the predicted RUL of a machine is lower than or equal to a certain threshold. '
                            'This threshold is the policy instrument that the management can optimize to achieve lower maintenance costs. '
                            'It is chosen for all machines in the dataset. '),
                        ui.br(),
                        ui.p('Go ahead and fiddle around with the figure below, as you change the threshold, the moment of maintenance changes as well.'),
                        ),
                        output_widget('policy_instrument_plot'),
                        ui.div(
                            ui.input_slider('policy_instrument_threshold', 'Threshold', 0, 50, 1),
                            ui.output_ui("policy_instrument_contents"),
                            style="display:flex;"
                        ),
                    ui.row(
                        ui.h2('Lead time'),
                        ui.p('Of course, maintenance rarely happens immediately. Usually, an organization does not have i) the required parts and/or ii) '
                            'the required expertise readily available (unless a significant premium is paid). So, instead of conducting maintenance right '
                            'away, it is scheduled when the predicted RUL is lower than the threshold. Thus, the moment at which maintenance occurs also '
                            'depends on the lead time of i) and ii).'),
                        ui.p('Go ahead and fiddle around with the figure below, as you change the threshold and the lead time, the moment of maintenance changes as well.'),
                        output_widget('lead_time_plot'),
                        ui.div(
                            ui.div(
                                ui.input_slider('lead_time_threshold', 'Threshold', 0, 50, 30),
                                ui.input_slider('lead_time_tau', 'Lead Time', 0, 50, 10)
                            ),
                            ui.output_ui("lead_time_contents"),
                            style="display:flex;"
                        ),
                        ui.HTML('<i>Note: often the lead times are uncertain, so instead, a range or distribution of lead times will be used to calculate the cost of maintenance. '
                                'This uncertainty will be explained in pane 3.</i>'
                                )
                        )
                    )
                )

@module.server
def info_server(input, output, session, preds, trues, cost_reactive, cost_predictive, cost_rul):

    @output
    @render_widget
    def policy_instrument_plot():
        mach = 74
        fig_preds = go.Figure(
            layout={
                "hovermode": "x",
                "title_text": "Simulation RUL for one machine.",
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
        threshold = input.policy_instrument_threshold()
        timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
        timestamp_failure = np.amin(np.argwhere(trues[mach, :] == 0))
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
        if timestamp_maintenance < timestamp_failure:
            # LOST RUL ARROW:
            fig_preds.add_annotation(
                ax=timestamp_maintenance,
                y=0,
                x=timestamp_failure,
                ay=0,
                axref='x',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green'
            )
            fig_preds.add_annotation(
                ax=timestamp_failure,
                y=0,
                x=timestamp_maintenance,
                ay=0,
                axref='x',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green'
            )
            fig_preds.add_scatter(
                x=[timestamp_maintenance, timestamp_failure],
                y=[0, 0],
                name=f"Lost RUL",
                line={
                    "color": "green",
                    "width": 2
                },
                mode='lines'
            )
        return fig_preds

    @output
    @render.ui
    def policy_instrument_contents():
        mach = 74
        tau = 0
        threshold = input.policy_instrument_threshold()
        cost = np.sum(calculate_PRMC(preds[mach, :], trues[mach, :], tau, threshold, cost_reactive, cost_predictive, cost_rul[mach]))
        if cost_reactive == cost:
            style = 'class="text-warning"'
            text_cost = 'Machine failed. Cost of failure.'
        elif cost_reactive < cost:
            style = 'class="text-danger"'
            text_cost = 'Machine repaired before failure occured. However, it is repaired so upfront, that the cost of repair exceeds that of failure.'
        else:
            style = 'class="text-success"'
            text_cost = 'Machine repaired before failure occured.'
        return ui.HTML(f"<b {style}>COST:</b> {cost:_}</br><p {style}>{text_cost}</p>")

    @output
    @render_widget
    def lead_time_plot():
        mach = 74
        tau = input.lead_time_tau()
        fig_preds = go.Figure(
            layout={
                "hovermode": "x",
                "title_text": "Simulation RUL for one machine.",
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
        threshold = input.lead_time_threshold()
        timestamp_maintenance = np.amin(np.argwhere(preds[mach, :] <= threshold))
        timestamp_failure = np.amin(np.argwhere(trues[mach, :] == 0))
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
        if timestamp_maintenance + tau < timestamp_failure:
            # LOST RUL ARROW:
            fig_preds.add_annotation(
                ax=timestamp_maintenance+tau,
                y=0,
                x=timestamp_failure,
                ay=0,
                axref='x',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green'
            )
            fig_preds.add_annotation(
                ax=timestamp_failure,
                y=0,
                x=timestamp_maintenance+tau,
                ay=0,
                axref='x',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green'
            )
            fig_preds.add_scatter(
                x=[timestamp_maintenance+tau, timestamp_failure],
                y=[0, 0],
                name=f"Lost RUL",
                line={
                    "color": "green",
                    "width": 2
                },
                mode='lines'
            )
        # Tube for lead time
        fig_preds.add_scatter(
            x=[timestamp_maintenance, timestamp_maintenance + tau, timestamp_maintenance + tau, timestamp_maintenance],
            y=[15, 15, -15, -15],
            fill='toself',
            mode='none',
            name="Lead Time",
            fillcolor="rgba(239, 163, 29, 0.5)",
        )
        return fig_preds

    @output
    @render.ui
    def lead_time_contents():
        mach = 74
        tau = input.lead_time_tau()
        threshold = input.lead_time_threshold()
        cost = np.sum(calculate_PRMC(preds[mach, :], trues[mach, :], tau, threshold, cost_reactive, cost_predictive, cost_rul[mach]))
        if cost_reactive == cost:
            style = 'class="text-warning"'
            text_cost = 'Machine failed. Cost of failure.'
        elif cost_reactive < cost:
            style = 'class="text-danger"'
            text_cost = 'Machine repaired before failure occured. However, it is repaired so upfront, that the cost of repair exceeds that of failure.'
        else:
            style = 'class="text-success"'
            text_cost = 'Machine repaired before failure occured.'
        return ui.HTML(f"<b>COST:</b> {cost:_}</br><p {style}>{text_cost}</p>")
