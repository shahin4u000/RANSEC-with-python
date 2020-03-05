from bokeh.layouts import grid
from bokeh.models import Button, TextInput
from bokeh.plotting import ColumnDataSource
from bokeh.models.tools import HoverTool
from docutils.nodes import figure
import numpy as np
import pandas as pd

def modify_doc(doc):
    # same as before
    source = ColumnDataSource(df)
    p = figure(tools=tools)
    p.scatter('x','y', source=source, alpha=0.5)
    p.add_tools(
        HoverTool(
            tooltips=[('value','@value{2.2f}'), 
                      ('index', '@index')]
        )
    )
    
    # this function is called when the button is clicked
    def update():
        # number of points to be added, taken from input text box
        n = int(npoints.value)
        # new sample of points to be added. 
        # we use the a narrow gaussian centred on (-1, 1), 
        # and draw the requested number of points
        sample3 = np.random.multivariate_normal([-1,-1], [[0.05,0],[0,0.05]], n)
        df_new = pd.DataFrame(sample3, columns=('x','y'))
        df_new['value'] = np.sqrt(df['x']**2 + df['y']**2)
        # only the new data is streamed to the bokeh server, 
        # which is an efficient way to proceed
        source.stream(df_new)
    
    # GUI: 
    button = Button(label='add points:')
    npoints = TextInput(value="50")
    button.on_click(update)
    # arranging the GUI and the plot. 
    layout = grid([[button, npoints], p])
    doc.add_root(layout)

show(modify_doc)