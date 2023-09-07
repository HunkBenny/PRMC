from shiny import module, ui, reactive, render
from shinywidgets import output_widget, render_widget
from plotly.subplots import make_subplots
import numpy as np
from ..metrics.maintenance import calculate_PRMC
import random
import scipy.stats as st
from functools import partial

@module.ui
def faq_ui():
    return ui.nav('FAQ',
                ui.h1('FAQ'),
                ui.row(
                    ui.h5('1) Individual thresholds?'),
                    ui.p(
                        'Instead of setting one threshold for ALL machines in the dataset, choosing them for the machines individually (or for groups of machines) '
                        'will probably be more effective. However, for simplicity reasons, this was left out of the current demo. '
                        ''
                    )
                ),
                ui.row(
                    ui.h5('2) Serial vs. parallel shops?'),
                    ui.p(
                        'In parallel shops, when a machine fails, this has no consequences for the other machines. '
                        'In serial shops, however, there is a dependency between the machines. This means that, '
                        'when a machine fails, the RUL of the consecutive machines cannot change anymore, as these machines are not being used. '
                        'Again, this is possible account for, however, this was left out the demo due for simplicity reasons. '
                    )
                ),
                ui.row(
                    ui.h5('3) Schedule overhaul-days for multiple machines instead of one at a time?'),
                    ui.p(
                        'Conducting predictive maintenance for multiple machines on the same day might also achieve a lower cost than conducting '
                        'maintenance for each machine individually.'
                        'TBO => Time Between Overhauls'
                    )
                )
    )