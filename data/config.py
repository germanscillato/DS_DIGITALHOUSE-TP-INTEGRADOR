from matplotlib.colors import ListedColormap, LinearSegmentedColormap

class config_graficos():
    def __init__(self):
        myColors = ((0.90, 0.96, 1, 1), (0.70, 0.87, 1, 1), (0, 0.40, 0.75, 0.88))
        cmap = LinearSegmentedColormap.from_list('Custom', myColors,10)