import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from mpl_toolkits.basemap import Basemap
import re

def get_traj(sub, ses):
    shortsub = re.match('sub-(\d{4})', sub)[1]
    traj_path = f'/ncf/mclaughlin_lab_tier1/STAR/8_digital/{sub}/{ses}/trajectory/{shortsub}.csv'
    try:
        traj = pd.read_csv(traj_path)
        traj = traj[(traj['y0'] > -360) & (traj['y0'] < 720)]
    except:
        print('Problem loading trajectory file')
        traj = None
    return(traj)
    
def plot_hist(traj):
    # Set up a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

    # Plot the histogram for 'y0' on the first subplot
    axes[0].hist(traj['y0'], bins=50, color='black', alpha=1)
    axes[0].set_title('y0 Histogram')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # Plot the histogram for 'x0' on the second subplot
    axes[1].hist(traj['x0'], bins=50, color='black', alpha=1)
    axes[1].set_title('x0 Histogram')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')

    # Adjust layout to ensure plots don't overlap
    plt.tight_layout()

    # Display the plots
    plt.show()
    
def plot_map(traj):
    # Create a quantilee and axis
    fig, ax = plt.subplots(figsize = (10,10))

    margin = 0.01  # Degrees latitude and longitude

    # Assuming 'x0' is latitude and 'y0' is longitude
    min_lat, max_lat = min(traj['x0']) - margin, max(traj['x0']) + margin
    min_lon, max_lon = min(traj['y0']) - margin, max(traj['y0']) + margin

    central_lat = (min_lat + max_lat) / 2
    central_lon = (min_lon + max_lon) / 2

    m = Basemap(projection='aea', 
              llcrnrlat=min_lat, urcrnrlat=max_lat, llcrnrlon=min_lon, urcrnrlon=max_lon, 
              lat_1=min_lat, lat_2=max_lat, lon_0=central_lon, lat_0=central_lat, 
              resolution='h', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray')

    # Convert lat and lon to x and y coordinates
    x, y = m(np.array(traj['y0']), np.array(traj['x0']))


    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GPS Trajectory')
    # Plot the trajectory
    ax.plot(x, y, marker='.', color='black', linestyle='-', markersize = 1, linewidth=.5, alpha = .35)