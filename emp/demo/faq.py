from shiny import module, ui, render
from shinywidgets import output_widget, render_widget

import numpy as np
from ..metrics.maintenance import calculate_PRMC
import plotly.graph_objects as go

import scipy.stats as st
from functools import partial
from .simulation_faq import simulation_faq_server, simulation_faq_ui

import scipy.integrate as integrate

@module.ui
def faq_ui(preds):
    all_choices = list(set(range(preds.shape[0])))
    all_choices.sort()
    return ui.nav('FAQ',
                ui.h1('FAQ'),
                ui.hr({"style": "margin-top:0px"}),
                ui.row(
                    ui.div(
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseEight" aria-expanded="false" aria-controls="collapseEight">'
                                    '<h5>How is the optimal machine learning model selected?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingEight'
                                }
                            ),
                            ui.div(
                                ui.div(
                                    ui.p(
                                        'Model selection is done using a metric that evaluates the performance of a model. Traditionally, '
                                        'these metrics measure statistical performance such as mean squared error. However, these traditional '
                                        'metrics do not account for organisational performance (i.e costs/benefits). Using cost-based metrics will '
                                        'lead to better model selection than traditional metrics.'
                                    ),
                                    ui.h5('Cost-sensitive tuning'),
                                    ui.p(
                                        'One way is to tune the ML-models by use of a cost-sensitive metric, such as the PRMC. '
                                        'This means that the models are being evaluated by this metric instead of purely statistical metrics. '
                                        'The selected model will be the model that scores the best on this cost metric. '
                                    ),
                                    ui.h5('Cost-sensitive learning'),
                                    ui.p(
                                        'Tuning still has its limits, the models are still trained on statistical performance (i.e. the loss / objective function '
                                        'optimizes statistical performance). Instead, we want to optimize profits directly. Therefore, models can also be trained on '
                                        'custom loss-measures.  '
                                    ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseEight",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingEight',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseOne" aria-expanded="false" aria-controls="collapseOne">'
                                    '<h5>What data is being used to make predictions?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingOne'
                                }
                            ),
                            ui.div(
                                ui.div(
                                    simulation_faq_ui('simulation_faq'),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseOne",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingOne',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">'
                                    '<h5>Which maintenance costs are accounted for?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingTwo'
                                }
                            ),
                            ui.div(
                                ui.div(
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
                                    ui.h5('The Pure RUL Maintenance Cost (PRMC)', {"class": "text-primary"}),
                                    ui.p(
                                        'These costs can be combined in one metric, the so-called PRMC. '
                                        'This metric can in turn be adapted to a custom loss-function to be used during training. '
                                        'This way, instead of using a standard statistical loss (such as the RMSE), the maintenance costs can be minimized directly. '
                                        ''
                                    ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseTwo",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingTwo',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseThree" aria-expanded="false" aria-controls="collapseThree">'
                                    '<h5>What is the interplay between the threshold and the lead time?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingThree'
                                }
                            ),
                            ui.div(
                                ui.row(
                                ui.h2('Policy instrument'),
                                ui.p('In order to lower these costs, an organisation conducts maintenance when the predicted RUL of a machine is lower than or equal to a certain threshold. '
                                    'This threshold is the policy instrument that the management can optimize to achieve lower maintenance costs. '
                                    'It is only chosen once for all machines in the dataset. '
                                    'For example, if the threshold is set to 10 days, then maintenance will be scheduled as soon as the predicted RUL '
                                    '≤ 10.'
                                ),
                                ui.br(),
                                ui.p('Go ahead and fiddle around with the figure below, as you change the threshold, the moment of maintenance changes as well.'),
                                output_widget('policy_instrument_plot'),
                                ui.div(
                                    ui.input_slider('policy_instrument_threshold', 'Threshold', 0, 50, 1),
                                    ui.output_ui("policy_instrument_contents"),
                                    style="display:flex;justify-content:space-evenly"
                                ),

                                ui.HTML('<hr>'),

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
                                        ui.div(
                                            ui.output_ui("lead_time_contents"),
                                            ui.input_switch('lead_time_moment_maintenance', 'Show the moment of maintenance?')
                                        ),
                                        style="display:flex;justify-content:space-evenly;"
                                    ),
                                    ui.HTML('<i>Note: often the lead times are uncertain, so instead, a range or distribution of lead times will be used to calculate the cost of maintenance. '
                                            'This uncertainty will be explained in pane 3.</i>'
                                            ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseThree",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingThree',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseFour" aria-expanded="false" aria-controls="collapseFour">'
                                    '<h5>Why are you not using individual thresholds?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingFour'
                                }
                            ),
                            ui.div(
                                ui.div(
                                    ui.p(
                                        'Instead of setting one threshold for ALL machines in the dataset, choosing them for the machines individually (or for groups of machines) '
                                        'will probably be more effective. However, for simplicity reasons, this was left out of the current demo. '
                                        'This will definitely be explored in a later stage.'
                                    ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseFour",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingFour',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseFive" aria-expanded="false" aria-controls="collapseFive">'
                                    '<h5>Does this keep in mind the complexity of series-parallel shops?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingFive'
                                }
                            ),
                            ui.div(
                                ui.div(
                                    ui.p(
                                        'In parallel shops, when a machine fails, this has no consequences for the other machines. '
                                        'In serial shops, however, there is a dependency between the machines. This means that, '
                                        'when a machine fails, the RUL of the consecutive machines cannot change anymore, as these machines are not being used. '
                                        'This is currently not implemented in this demo. However, this is a very relevant matter that is part of the further research in this project. '
                                    ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseFive",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingFive',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSix" aria-expanded="false" aria-controls="collapseSix">'
                                    '<h5>Schedule overhaul-days for multiple machines at once instead of one machine at a time?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingSix'
                                }
                            ),
                            ui.div(
                                ui.div(
                                    ui.p(
                                        'Conducting predictive maintenance for multiple machines on the same day might also achieve a '
                                        'lower cost than conducting maintenance for each machine individually. This can be accounted for in a '
                                        'custom evaluation metric. Possible implementations of this should be researched. '
                                    ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseSix",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingSix',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        ui.div(  # accordion item
                            ui.h1(
                                ui.HTML(
                                    '<button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSeven" aria-expanded="false" aria-controls="collapseSeven">'
                                    '<h5>How can you deal with uncertainty in the lead time?</h5>'
                                    '</button>'
                                ),
                                {
                                    'class': 'accordion-header',
                                    'id': 'headingSeven'
                                }
                            ),
                            ui.div(
                                ui.div(
                                    ui.row(
                                        ui.column(
                                            9,
                                            output_widget('rul_plot_faq'),
                                        ),
                                        ui.column(
                                            3,
                                            ui.input_select('mach_uncertainty_faq', 'Select machine', all_choices),  # type: ignore
                                            ui.input_slider('rul_threshold_faq', 'Threshold', 0, 50, 10),
                                            style="margin-top:5%"
                                        )
                                    ),
                                    ui.HTML(
                                        '<hr>'
                                    ),
                                    ui.row(
                                        ui.column(
                                            7,
                                            output_widget('lead_time_distr_plot_faq'),
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
                                            ui.output_ui('eprmc_formula_faq')
                                        )
                                    ),
                                    {
                                        'class': 'accordion-body'
                                    }
                                ),
                                {
                                    'id': "collapseSeven",
                                    'class': 'accordion-collapse collapse',
                                    'aria-labelledby': 'headingSeven',
                                    'data-bs-parent': '#accordionFaq'
                                }
                            ),
                            {
                                'class': 'accordion-item',
                                'id': 'accordionFAQ'
                            },
                        ),
                        {'class': ' accordion'}
                    ),
                ),
    )

@module.server
def faq_server(input, output, session, preds, trues, cost_reactive, cost_predictive, cost_rul):
    distr_pdf = partial(st.lognorm.pdf, s=1.2, scale=6)
    simulation_faq_server('simulation_faq')


    @output
    @render_widget  # type: ignore
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
                name="Lost RUL",
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
        cost = np.round(np.sum(calculate_PRMC(preds[mach, :], trues[mach, :], tau, threshold, cost_reactive, cost_predictive, cost_rul[mach])), 3)
        if cost_reactive == cost:
            style = 'class="text-warning"'
            text_cost = 'Machine failed. Cost of failure.'
        elif cost_reactive < cost:
            style = 'class="text-danger"'
            text_cost = 'Machine repaired before failure occured. However, it is repaired so upfront, that the cost of repair exceeds that of failure.'
        else:
            style = 'class="text-success"'
            text_cost = 'Machine repaired before failure occured.'
        return ui.HTML(f"<b {style}>COST:</b> €{cost:_} (thousands)</br><p {style}>{text_cost}</p>")

    @output
    @render_widget  # type: ignore
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
                "color": 'grey'
            },
            mode='lines',
            name="Moment of scheduling maintenance"
        )
        fig_preds.add_vline(
            x=timestamp_maintenance,
            line_color="grey"
        )
        if timestamp_maintenance + tau < timestamp_failure:
            # LOST RUL ARROW:
            fig_preds.add_annotation(
                ax=timestamp_maintenance + tau,
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
                x=timestamp_maintenance + tau,
                ay=0,
                axref='x',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green'
            )
            fig_preds.add_scatter(
                x=[timestamp_maintenance + tau, timestamp_failure],
                y=[0, 0],
                name="Lost RUL",
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
        if input.lead_time_moment_maintenance():
            fig_preds.add_vline(
                x=timestamp_maintenance + tau,
                annotation_text='Moment of maintenance',
                line_color='black'
            )
        return fig_preds

    @output
    @render.ui
    def lead_time_contents():
        mach = 74
        tau = input.lead_time_tau()
        threshold = input.lead_time_threshold()
        cost = np.round(np.sum(calculate_PRMC(preds[mach, :], trues[mach, :], tau, threshold, cost_reactive, cost_predictive, cost_rul[mach])), 3)
        if cost_reactive == cost:
            style = 'class="text-warning"'
            text_cost = 'Machine failed. Cost of failure.'
        elif cost_reactive < cost:
            style = 'class="text-danger"'
            text_cost = 'Machine repaired before failure occured. However, it is repaired so upfront, that the cost of repair exceeds that of failure.'
        else:
            style = 'class="text-success"'
            text_cost = 'Machine repaired before failure occured.'
        return ui.HTML(f"<b>COST:</b> €{cost:_} (thousands)</br><p {style}>{text_cost}</p>")

    @output
    @render_widget  # type: ignore
    def lead_time_distr_plot_faq():
        mach = int(input.mach_uncertainty_faq())
        threshold = input.rul_threshold_faq()
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
    def rul_plot_faq():
        mach = int(input.mach_uncertainty_faq())
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
        threshold = input.rul_threshold_faq()
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
    def eprmc_formula_faq():
        mach = int(input.mach_uncertainty_faq())
        threshold = input.rul_threshold_faq()
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
                    r'\end{cases}'
                    r'&='
                    fr'\text{{€ }}{Cr + Cp} \text{{ (thousands)}}'
                    r'\end{align}\]'
                    r'<script>MathJax.typeset()</script>'
                )
