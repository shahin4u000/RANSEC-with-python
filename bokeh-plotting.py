#%%
from bokeh.layouts import column
from bokeh.models import Button, CustomJS, ColumnDataSource, Slider,PointDrawTool
from bokeh.plotting import figure, output_file, show

output_file("callback.html")

x = [1, 2,  2, 2,  2,  3, 3,  5, 5,  6, 6 ,6,6,6,6,6]
y = [4, 50, 0, 10, 20, 0, 50, 0, 50, 0, 50,10,120,30,40,50]
a = []
b = []
source = ColumnDataSource(data=dict(x=x, y=y))
source1 = ColumnDataSource(data=dict(x=a, y=b))
plot = figure(plot_width=400, plot_height=400)
plot.toolbar.logo = None

plot.scatter('x', 'y', source=source,
             line_color='green', fill_alpha=0.6, size=5)
customPlot= plot.scatter('x', 'y', source=source1,
             line_color='blue', fill_alpha=0.3, size=10)
plot.patch('x', 'y',source=source1, line_width=3, color="navy", alpha=0.1)
draw_tool = PointDrawTool(renderers=[customPlot])

callback = CustomJS(args=dict(xy=source1, ab=source), code="""
        var data = xy.data;
        var data2 = ab.data;
        
        var f = cb_obj.value
        console.log(data, data2)
        var x = data['x'];
        var y = data['y'];
        var a = data2['x'];
        var b = data2['y'];
        
        if(a.includes(cb_obj.x) && b.includes(cb_obj.y)) {
            x.push(cb_obj.x)
            y.push(cb_obj.y)
            
        } else {
            x.push(cb_obj.x)
            y.push(cb_obj.y)
            console.log(cb_obj.x)
        }
        
        
        xy.change.emit();
    """)

slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")
slider.js_on_change('value', callback)
plot.js_on_event('tap', callback)
layout = column(slider, plot)
plot.add_tools(draw_tool)
plot.toolbar.active_tap = draw_tool
show(layout)

# %%


def modify_doc(doc):
    # same as before
    source = ColumnDataSource(df)
    p = figure(tools=tools)
    p.scatter('x', 'y', source=source, alpha=0.5)
    p.add_tools(
        HoverTool(
            tooltips=[('value', '@value{2.2f}'),
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
        sample3 = np.random.multivariate_normal(
            [-1, -1], [[0.05, 0], [0, 0.05]], n)
        df_new = pd.DataFrame(sample3, columns=('x', 'y'))
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

#%%
from bokeh.plotting import figure, output_file, show
from bokeh.models.tools import PointDrawTool

x = [1, 2, 3, 5, 4]
y = [6, 7, 8, 3,7]

output_file("multiple.html")

p = figure(plot_width=400, plot_height=400)

# add both a line and circles on the same plot
c1 = p.patch(x, y, line_width=2)
c2 = p.circle(x, y, fill_color="white", size=8)
tool = PointDrawTool(renderers=[c1, c2])
show(p)

# %%


output_file("slider.html")

p1 = figure(plot_width=300, plot_height=300)
p1.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)
tab1 = Panel(child=p1, title="circle")

p2 = figure(plot_width=300, plot_height=300)
p2.patch([1, 2, 3, 4, 5], [6, 7, 2, 4, 5],
         line_width=3, color="navy", alpha=0.5)
tab2 = Panel(child=p2, title="line")

tabs = Tabs(tabs=[tab1, tab2])

show(tabs)
