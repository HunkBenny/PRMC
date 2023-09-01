from shiny import module, ui, reactive
from shinywidgets import output_widget, render_widget, register_widget
import plotly.graph_objects as go
import numpy as np

from ..metrics.maintenance import calculate_PRMC

@module.server
def prmc_server(input, output, session, preds, trues, num_thresholds):
    # PRMC PANE
    fig_prmc = go.FigureWidget(layout={
        "title_text": "Deterministic PRMC",
        "title_x": 0.5,
        "title_font_size": 40,
        "height": 500,
        "showlegend": True,
        "hovermode": "x"
    })
    register_widget('prmc_plot', fig_prmc)

    fig_cost_distribution = go.FigureWidget(layout={
        "title_text": "PRMC Cost Distribution",
        "title_x": 0.5,
        "title_font_size": 40,
        "height": 500,
        "showlegend": True,
        "xaxis": {
            "tickmode": 'array',
            "tickvals": [0, 1, 2],
            "ticktext": ['C_p <= C_r', 'C_p > C_r', 'C_r']
        }
    })
    register_widget('cost_distribution_plot', fig_cost_distribution)

    @output
    @render_widget
    @reactive.event(input.compute_prmc)
    async def prmc_plot():
        cost_rul = input.acquisition_price.get() / np.amax(trues, 1)
        thresholds = np.linspace(0.01, 150, num_thresholds)

        costs_to_plot = [np.sum(calculate_PRMC(preds, trues, input.tau.get(), ti, input.cost_reactive.get(), input.cost_predictive.get(), cost_rul)) for ti in thresholds]
        fig_prmc.add_scatter(x=thresholds,
                              y=costs_to_plot,
                              name=f"Threshold: {np.round(thresholds[np.argmin(costs_to_plot)], 2)} Cost: {np.round(np.min(costs_to_plot)):_}",
                              hovertemplate="Threshold: %{x} <br>Cost: %{y}"
                    )

        return fig_prmc

    @output
    @render_widget
    @reactive.event(input.compute_cost_distribution)
    async def cost_distribution_plot():
        # this only looks at the hypothetical moment of maintenance and failure. The moments still need to be compared
        all_PDM = list()
        all_failures = list()

        threshold = input.threshold_selected.get()
        cost_reactive = input.cost_reactive.get()
        cost_predictive = input.cost_predictive.get()
        cost_rul = input.acquisition_price.get() / np.amax(trues, 1)
        tau = input.tau.get()

        PdM = [(r, np.amin(np.where(preds <= threshold)[1][np.where(preds <= threshold)[0] == r]))for r in set(np.where(preds <= threshold)[0])]
        failures = [(r, np.amin(np.where(trues == 0)[1][np.where(trues == 0)[0] == r])) for r in set(np.where(trues == 0)[0])]
        all_PDM.append(PdM)
        all_failures.append(failures)

        cost = np.zeros((len(preds), 2))
        for idx, (true, pred) in enumerate(zip(failures, PdM)):
            diff_rul = true[1] - pred[1] - tau
            if diff_rul >= 0:
                if type(cost_rul) != np.ndarray:
                    cost[true[0]][0] = diff_rul * cost_rul[idx] + cost_predictive  # take the cost_rul of the corresponding machine
                else:
                    cost[true[0]][0] = diff_rul * cost_rul[true[0]] + cost_predictive  # take the cost_rul of the corresponding machine
            else:
                cost[true[0]][1] = cost_reactive

        failed_machs = sum(cost[:, 1]) / cost_reactive
        predicted_machs = len(cost) - failed_machs
        predicted_machs_expensive = len(cost[cost > cost_reactive])
        predicted_machs_cheap = predicted_machs - predicted_machs_expensive

        failed_machs_ratio = round(failed_machs / cost.shape[0], 4)
        predicted_machs_expensive_ratio = round(predicted_machs_expensive / cost.shape[0], 4)
        predicted_machs_cheap_ratio = round(predicted_machs_cheap / cost.shape[0], 4)

        data = [predicted_machs_cheap_ratio, predicted_machs_expensive_ratio, failed_machs_ratio]

        fig_cost_distribution.add_bar(y=data,
                                       name=f"Threshold: {threshold} <br> Lead time {tau}",
                                       customdata=np.transpose(np.array([[
                                            np.sum(cost[cost[:, 0] <= cost_reactive, 0]),
                                            np.sum(cost[cost[:, 0] > cost_reactive, 0]),
                                            failed_machs * cost_reactive
                                                        ],[
                                            int(predicted_machs_cheap),
                                            int(predicted_machs_expensive),
                                            int(failed_machs)
                                                    ]
                                                ])),
                                       hovertemplate=
                                            "Cost (M): %{customdata[0]:.2f}<br>" +
                                            "Number: %{customdata[1]}"
                                    )
        return fig_cost_distribution

@module.ui
def prmc_ui(cost_reactive, cost_predictive):
    return ui.nav("PRMC",  # PRMC tab
           ui.layout_sidebar(  # upper figure
               ui.panel_sidebar(
                   ui.input_numeric('cost_reactive', 'Cost Reactive (in millions)', cost_reactive, step=1),
                   ui.input_numeric('cost_predictive', 'Cost Predictive (in millions)', cost_predictive, step=1),
                   ui.input_numeric('tau', 'Lead Time', 12, step=1),
                   ui.input_numeric('acquisition_price', 'Acquisition Price (in millions)', 50, step=1),
                   ui.input_action_button("compute_prmc", "Compute!"),
                   width=2.5  # type: ignore
               ),
               ui.panel_main(output_widget('prmc_plot'))
           ),
           ui.layout_sidebar(  # lower figure
               ui.panel_sidebar(
                   ui.input_slider('threshold_selected', 'Threshold', 0, 150, 25),
                   ui.input_action_button("compute_cost_distribution", "Compute!"),
                   width=2.5  # type: ignore
               ),
               ui.panel_main(output_widget('cost_distribution_plot'))
           )
         )