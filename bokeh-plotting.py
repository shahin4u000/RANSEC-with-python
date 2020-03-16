#%%
from bokeh.layouts import column
from bokeh.models import CustomJS, ColumnDataSource, Slider
from bokeh.plotting import figure, output_file, show

output_file("callback.html")

x = [x*0.005 for x in range(0, 200)]
y = x
a = [1]
b = [2]
source = ColumnDataSource(data=dict(x=x, y=y))
source1 = ColumnDataSource(data=dict(x=a, y=b))
plot = figure(plot_width=400, plot_height=400)
plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
plot.scatter('x', 'y', source=source1, line_color='blue', fill_alpha=0.3, size=10)
callback = CustomJS(args=dict(source=source1), code="""
        var data = source.data;
        var f = cb_obj.value
        console.log(cb_obj.value)
        var x = data['x']
        var y = data['y']
        x.push(cb_obj.x)
        y.push(cb_obj.y)
        source.change.emit();
    """)

slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")
slider.js_on_change('value', callback)
plot.js_on_event('tap', callback)
layout = column(slider, plot)

show(layout)

#%%
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