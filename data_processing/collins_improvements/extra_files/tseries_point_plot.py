import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"  # Open Plotly plots in browser


def load_and_prepare_data(csv_path, var_name="FIRA"):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df[var_name] = df[var_name].fillna(0)

    divide_ids = sorted(df['divide_id'].unique())
    divide_id_map = {id_: i for i, id_ in enumerate(divide_ids)}
    df['divide_pos'] = df['divide_id'].map(divide_id_map)

    return df, divide_ids


def render_2d_plot(df, divide_ids, var_name="FIRA"):
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        df['time'],
        df['divide_pos'],
        c=df[var_name],
        cmap='viridis',
        s=20,
        alpha=0.8,
        edgecolors='none'
    )

    plt.colorbar(scatter, label=f'{var_name} Intensity')
    plt.xlabel('Time')
    plt.ylabel('Divide ID')
    plt.title(f'Time Series of {var_name} Intensity by Divide ID')

    # Y-ticks
    step = max(1, len(divide_ids) // 20)
    ytick_locs = list(range(0, len(divide_ids), step))
    ytick_labels = [divide_ids[i] for i in ytick_locs]
    plt.yticks(ytick_locs, ytick_labels)

    # X-ticks
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()


def render_3d_matplotlib(df, divide_ids, var_name="FIRA"):
    x = mdates.date2num(df['time'])
    y = df['divide_pos']
    z = df[var_name]
    c = df[var_name]

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=10, alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Divide ID')
    ax.set_zlabel(var_name)
    ax.set_title(f'3D Time Series Scatter Plot: {var_name}')

    cb = fig.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label(f'{var_name} Intensity')

    ax.set_yticks(np.linspace(min(y), max(y), 20, dtype=int))
    ax.set_yticklabels([divide_ids[int(i)] for i in np.linspace(0, len(divide_ids) - 1, 20, dtype=int)])

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def render_3d_plotly(df, divide_ids, var_name="FIRA"):
    fig = px.scatter_3d(
        df,
        x='time',
        y='divide_pos',
        z=var_name,
        color=var_name,
        color_continuous_scale='Viridis',
        opacity=0.7,
        title=f"3D Scatter of {var_name} by Time and Divide ID",
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Divide ID',
            yaxis=dict(
                tickvals=np.linspace(0, len(divide_ids) - 1, 20, dtype=int),
                ticktext=[divide_ids[i] for i in np.linspace(0, len(divide_ids) - 1, 20, dtype=int)]
            ),
            zaxis_title=var_name
        )
    )

    fig.show()


# ============ Example usage ============
df, divide_ids = load_and_prepare_data("ldas_test_summary2_timeseries.csv", var_name="FIRA")
render_2d_plot(df, divide_ids, var_name="FIRA")

df3d, divide_ids3d = load_and_prepare_data("multidate_ldas_test_timeseries.csv", var_name="FIRA")
render_3d_matplotlib(df3d, divide_ids3d, var_name="FIRA")
render_3d_plotly(df3d, divide_ids3d, var_name="FIRA")
