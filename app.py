from shiny import App, ui

from emp.demo import prmc, eprmc, simulation, home, info

from pathlib import Path

import shinyswatch

import pickle

import numpy as np

label_font = {'size': 14}

# load data:
with open('data/delme/temp.pkl', 'rb') as r:
    obj = pickle.load(r)
preds = obj['preds']
trues = obj['trues']
cost_rul = np.round(obj['cost_rul'] / 1_000_000, 3)
cost_reactive = np.round(obj['cost_reactive'] / 1_000_000, 3)
cost_predictive = np.round(obj['cost_predictive'] / 1_000_000, 3)
tau = 12
num_thresholds = 40


# app logic:

# create static path
static = Path(__file__).parent / 'static'

app_ui = ui.page_fluid(
    ui.include_css(static / "custom.css"),
    shinyswatch.theme.pulse(),
    ui.navset_pill(
        info.info_ui('info_ui'),
        # home.home_ui('home_ui'),
        ui.nav_menu(
            'Extra',
            simulation.simulation_ui('simulation_ui'),
            prmc.prmc_ui('prmc_ui', cost_reactive, cost_predictive),
            eprmc.eprmc_ui('eprmc_ui', cost_reactive, cost_predictive),
            ui.nav("Lead time distribution"),
        ),
    )
)

def server(input, output, session):
    info.info_server('info_ui', preds, trues, cost_reactive, cost_predictive, cost_rul)
    # home.home_server('home_ui', preds, trues, cost_reactive, cost_predictive, cost_rul)
    prmc.prmc_server('prmc_ui', preds, trues, num_thresholds)
    eprmc.eprmc_server('eprmc_ui', preds, trues, num_thresholds)
    simulation.simulation_server('simulation_ui')


static_dir = Path(__file__).parent / 'static'
app = App(app_ui, server, static_assets=static_dir)
