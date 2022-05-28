import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

msft_data = yf.download("MSFT", period="3mo", interval = "60m")

def plot_data():
    label_index = [str(x)[-2:] for x in msft_data.index]
    index = np.arange(0, len(msft_data.index))
    close = np.array(msft_data["Close"])

    xy_spline = make_interp_spline(index, close)

    smooth_index = np.linspace(min(index), max(index), 1000)
    smooth_close = xy_spline(smooth_index)

    plt.plot(smooth_index, smooth_close)
    plt.xlabel([])
    # plt.xticks(labels=label_index, ticks=range(134), rotation = '90')
    # plt.xlabel(f"From {msft_data.index[0]} - {msft_data.index[-1]}")
    # msft_data["Close"].plot()
    plt.show()

def splitting_data():
    split = 5  #Every 5 Hours
    # 0 1 2 3 4 (5)<--We predict from (0-4) (*5 entries -> 6th one predicted)

    xtrain = pd.DataFrame(columns=[f"Period{x}" for x in range(5)])
    ytrain = []
    close_data = np.array(msft_data["Close"])

    for iter in range(5, len(close_data)):
        xtrain.loc[iter-5] = close_data[iter-5:iter]
        ytrain.append(close_data[iter])

    xtrain["target"] = ytrain
    print(xtrain, end = "\n\n")
    # print(len(ytrain))

splitting_data()




