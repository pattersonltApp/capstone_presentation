import numpy as np
import pandas as pd

from bokeh.plotting import figure, show, output_file
import bokeh.io
bokeh.io.output_notebook()

output_file('graph.html')
df = pd.read_csv('a-04-30_145052.csv')

p = bokeh.plotting.figure(
    x_axis_label='time (s)',
    y_axis_label='Signal',
    frame_height=175,
    frame_width=500,
    x_range=[df['time (sec)'].min(), df['time (sec)'].max()],
)

for signal in [('signal 0', 'red'), ('signal 1', 'green'), ('signal 2', 'purple'), ('signal3', 'orange')]:
    p.line(source=df, x='time (sec)', y=signal[0], color=signal[1])
show(p)
