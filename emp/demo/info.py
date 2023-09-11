from shiny import module, ui, render
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import numpy as np
from ..metrics.maintenance import calculate_PRMC

@module.ui
def info_ui():
    return ui.nav('Info',
                ui.div(
                    ui.row(
                        ui.h1('Cost-sensitive predictive maintenance', {"class": "text-primary", "style": "text-align:center"}),
                        ui.HTML('<hr>')
                    ),
                    ui.row(
                        ui.column(
                            8,
                            ui.img(src='corrective-maintenance.jpg', style='width:100%'),
                            style="display:inline-flex;width:fit-content;flex: 1 2  66vw;height:fit-content;max-width:700px;"
                        ),
                        ui.column(
                            4,
                            ui.h2('Problem Statement'),
                            ui.HTML(
                                '<p style="width:100%;word-wrap:break-word;">'
                                'Many manufacturing organizations accrue <strong class="text-primary-emphasis">high maintenance costs</strong>. These costs can be greatly reduced by having an <strong class="text-primary-emphasis">effective maintenance policy</strong>. '
                                'Utilizing a proactive maintenance policy has been proven to significantly decrease maintenance costs due to intervening before machine '
                                'breakdown. This approach not only results in <strong class="text-primary-emphasis">fewer repair costs</strong>, but also <strong class="text-primary-emphasis">increases the machine\'s lifetime</strong>'
                                '</p>'
                            ),
                            ui.HTML(
                                '<p>'
                                'Previous work focused on estimating the Remaining Useful Life (RUL) of a machine. This approach aids you to schedule maintenance, but does not dictate which specific moment '
                                'minimizes maintenance costs nor does it consider the costs associated with maintenance. However, our proposal does suggest a specific maintenance scheduling time. This is '
                                'done by developing a <strong class="text-primary-emphasis">problem-specific cost measure</strong>. The cost measure tells the operator <strong class="text-primary-emphasis">what the cost of scheduling maintenance is</strong> at a moment in time, and in extent tells '
                                '<strong class="text-primary-emphasis">you which moment minimizes these costs</strong>. The cost measure considers the lead time and balances the probability of reacting too late (machine breaks causing sudden downtime '
                                'and increased repair costs) and reacting too early (lower repair costs but losing productive hours of the machine). Simultaneously, this cost measure can be used during the '
                                'evaluation of RUL machine learning models to choose the model which minimizes maintenance costs. This leads to <strong class="text-primary-emphasis">selecting superior machine learning models which directly take '
                                'business objectives into account.</strong>'
                                '</p>'
                            ),
                            ui.HTML(
                                'In the <strong class="text-primary-emphasis">interactive demo</strong> (second panel), you will get the chance to <strong class="text-primary-emphasis">play against our data-driven solution</strong>. You will act as the operator of several CNC machines and your '
                                'goal is to find the optimal moment to schedule maintenance.'
                            ),
                            style="display:block;width:fit-content;height:fit-content;flex: 2 1 34vw;",
                        ),
                    ),
                )
            )

@module.server
def info_server(input, output, session, preds, trues, cost_reactive, cost_predictive, cost_rul):
    pass
