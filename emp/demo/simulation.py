from shiny import module, ui, reactive, render
from shinywidgets import output_widget, render_widget
from plotly.subplots import make_subplots
import numpy as np
from plotly import colors
import pandas as pd


basetable = pd.read_csv('data/gold/train.csv')
all_choices = list(set(basetable['unit_ID']))
all_choices.sort()
rendered_input = False
timestep = 0
selected_cols = ["T24", "T30", "T50", "P30"]

@module.ui
def simulation_ui():
    return ui.nav('Simulation',
            ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.output_ui("contents"),
                        ui.input_switch('hold', 'Hold'),
                        ui.input_action_button('reset_clock', 'Reset history'),
                        ui.HTML("<b>"),
                        ui.input_select('selected_mach', 'Machine', all_choices, selected=all_choices[0], multiple=True),
                        ui.HTML("</b>"),
                        width=2,
                    ),
                    ui.panel_main(output_widget('simulation_plot')
                    )
                )
            )


@module.server
def simulation_server(input, output, session):
    reset = reactive.Value(False)

    @reactive.Calc
    def clock():
        if not reset():
            global timestep
            if not input.hold():
                timestep += 1
                reactive.invalidate_later(0.75)
                return timestep
            else:
                with reactive.isolate():
                    return timestep
        else:
            timestep = 0
            reset.set(not reset())
            return 0

    @output
    @render_widget
    def simulation_plot():
        global basetable
        selected_mach = input.selected_mach()[0] if len(input.selected_mach()) == 1 else input.selected_mach()  # selecting multiple is possible atm. perhaps change this def so it allows for the plotting of multiple machs
        num_rows = 4
        num_cols = int(np.ceil(len(input.columns()) / num_rows))

        if type(selected_mach) == str:
            df = basetable.loc[basetable["unit_ID"] == selected_mach, :]
            selected_mach = [selected_mach]  # turn into list for looping purposes
        else:  # if multiple selected
            df = basetable.loc[basetable["unit_ID"].isin(selected_mach), :]

        try:
            fig_simulation = make_subplots(
                                rows=num_rows,
                                cols=num_cols,
                                vertical_spacing=0.07,
                                subplot_titles=input.columns()
                            )
        except:
            fig_simulation = make_subplots(
                                rows=num_rows,
                                cols=num_cols,
                                vertical_spacing=0.07,
                                subplot_titles=["T24", "T30", "T50", "P30"]
                            )

        coords = list()
        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                coords.append((row, col))

        for idx_column, (col, coord) in enumerate(zip(input.columns(), coords)):
            for idx_color, mach in enumerate(selected_mach):
                df_mach = df[df['unit_ID'] == mach]
                fig_simulation.add_scatter(
                    x=df_mach["cycles"],
                    y=df_mach[col].values[0:timestep],
                    row=coord[0],
                    col=coord[1],
                    showlegend=True if idx_column == 0 else False,
                    name=f"{mach}",
                    marker={
                        "color": colors.DEFAULT_PLOTLY_COLORS[idx_color]
                    },
                    line={
                        "color": colors.DEFAULT_PLOTLY_COLORS[idx_color]
                    })
            fig_simulation.update_xaxes(range=(0, max(df['cycles'])),
                                        row=coord[0],
                                        col=coord[1])
            fig_simulation.update_yaxes(range=(min(df[col]), max(df[col])),
                                        row=coord[0],
                                        col=coord[1])
        clock()
        mach_text = ', '.join(f"{mach}" for mach in selected_mach)
        fig_simulation.update_layout(height=min(max(600, 200 * len(input.columns())), 800),
                                    title_text="<b>Simulation of Machine: " + mach_text + "</b>" if len(selected_mach) == 1 else "<b>Simulation of Machines: " + mach_text + "</b>")
        return fig_simulation

    @output
    @render.ui
    def contents():
        global rendered_input
        global selected_cols
        cols = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31',
                'W32']
        if not rendered_input:
            rendered_input = True
            return ui.input_checkbox_group('columns', ui.h3('Sensors'), cols, selected=selected_cols)
        else:
            return ui.input_checkbox_group('columns', ui.h3('Sensors'), cols, selected=selected_cols)

    @reactive.Effect(priority=100)
    @reactive.event(input.reset_clock, input.selected_mach)
    def reset_clock():
        reset.set(not reset())

    @reactive.Effect
    def change_selected_cols():
        # this method fixes the case in which a reload would bug out the program
        global selected_cols
        selected_cols = input.columns()
