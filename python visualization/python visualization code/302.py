"""
302. GUI Tkinter
"""
import numpy as np
from ______ import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import ________
import math
# Import pandas as a alias 'pd'
import pandas as pd

MAXVAL = 0
INTERVAL = 1000

fig = Figure(figsize=(5,4), dpi=100)
ax = fig.________(111)

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017 = pd.______("./data/marathon_2015_2017.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
record = pd._________(marathon_2015_2017,columns=['5K',  '10K',  '15K',  '20K', 'Half',  '25K',  '30K',  '35K',  '40K',  'Official Time']).sort_values(by=['Official Time'])

# Dataframe to List
record_list = record.values.________

xData = [5, 10, 15, 20, 21.098, 25, 30, 35, 40, 42.195 ]

def update():
    t_a = int(t_aSpbox._____)
    MAXVAL = t_a * INTERVAL
    ax.set_xlabel('Distance(km)')
    ax.set_ylabel('Time(Second)')
    ax.set_title(str(t_a) + ' records of runners')
    
    for t in _____(0, MAXVAL, INTERVAL):
        ax._____(xData, record_list[int(t)], 'o', label=str(t+1))    

    ax.legend()
    fig.canvas._____

#main
main = Tk()
main.title("Marathon Records")
main.geometry()

label=Label(main, text='Marathon Records')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=4)

t_aVal  = DoubleVar(value=1.0)

t_aSpbox = ______(main, textvariable=t_aVal ,from_=0, to=100, increment=1, justify=RIGHT)
t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=_____(main, text='Number of runner : ')                
t_aLabel.grid(row=1,column=0)

Button(main,text="Run",width=20,height=5,command=lambda:update()).grid(row=1, column=2, columnspan=2, rowspan=1)

canvas = ___________(fig, main)
canvas.___________.grid(row=4,column=0,columnspan=3) 

main.mainloop()
