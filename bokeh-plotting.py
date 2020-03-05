from math import sin
from random import random

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import plasma
from bokeh.plotting import figure
from bokeh.transform import transform

list_x = list(range(100))
list_y = [random() + sin(i / 20) for i in range(100)]
desc = [str(i) for i in list_y]

source = ColumnDataSource(data=dict(x=list_x, y=list_y, desc=desc))
hover = HoverTool(tooltips=[
    ("index", "$index"),
    ("(x,y)", "(@x, @y)"),
    ('desc', '@desc'),
])
mapper = LinearColorMapper(palette=plasma(256), low=min(list_y), high=max(list_y))

p = figure(plot_width=400, plot_height=400, tools=[hover], title="Belgian test")
p.circle('x', 'y', size=10, source=source,
         fill_color=transform('y', mapper))

output_file('test.html')
show(p)