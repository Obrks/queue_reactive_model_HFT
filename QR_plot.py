import QR_model as QR
import plotly.graph_objects as go
import copy
import matplotlib.pyplot as plt
import numpy as np
"""
ob = OrderBook object, the current limit order book state to simulate
T = Total simulation time horizon
dt = Time step between two simulation 
stf = Intensity scaling factor: larger -> more arrivals/removals per step
Cbound = Queue-size threshold above which growth is penalized
delta = Penalty applied when a queue size exceeds Cbound
H = Maximum allowed total incoming flow across queues
Variables are used for the "market activity intensity" but intensities themselves can be changed in QR_model.py
"""


def lob_figure(ob, t=None):
    k = ob.k
    p_ref = ob.p_ref

    bids = ob.states[:k]
    asks = ob.states[k:]

    bid_prices = [q.price for q in bids]
    ask_prices = [q.price for q in asks]

    bid_sizes = [q.size for q in bids]
    ask_sizes = [q.size for q in asks]

    bid_sizes_left = [-s for s in bid_sizes]
    max_size = max(bid_sizes + ask_sizes + [1])

    title = "Order Book Snapshot" if t is None else f"Order Book Snapshot — t={t:.2f}"

    fig = go.Figure()
    fig.add_trace(go.Bar(y=bid_prices, x=bid_sizes_left, orientation="h", name="Bid"))
    fig.add_trace(go.Bar(y=ask_prices, x=ask_sizes, orientation="h", name="Ask"))

    fig.add_hline(y=p_ref, line_dash="dash", annotation_text="p_ref", annotation_position="top right")

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis=dict(range=[-max_size * 1.2, max_size * 1.2], zeroline=True, zerolinewidth=2, title="Size (Bid negative)"),
        yaxis=dict(title="Price"),
        template="plotly_white",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def simulate_and_show(ob, T=10.0, stf=20, Cbound=200, delta=1.0, H=10_000):
    """
    Simulate from t=0 to T, with dt = 1/stf.
    Show snapshots at each integer time: 0,1,...,T.
    """
    dt = 1.0 / stf
    n_steps = int(T * stf)

    lob_figure(ob, t=0.0).show()

    next_show = 1  

    for step in range(1, n_steps + 1):
        
        ob.step(stf=stf, Cbound=Cbound, delta=delta, H=H)

        t = step * dt

        if next_show <= T and t + 1e-12 >= next_show:
            lob_figure(ob, t=float(next_show)).show()
            next_show += 1


def simulate_best_prices(ob, T=10.0, dt=0.01, stf=1, Cbound=100, delta=1.0, H=1000):
    """
    Simulate one path of the order book and record best bid / ask over time.
    """
    ob_sim = copy.deepcopy(ob)

    times = np.arange(0, T + dt, dt)
    best_bids = []
    best_asks = []

    for _ in times:
        bid, ask = ob_sim.get_best()
        best_bids.append(bid)
        best_asks.append(ask)

        ob_sim.step(stf=stf, Cbound=Cbound, delta=delta, H=H)

    return times, np.array(best_bids), np.array(best_asks)

def plot_best_prices(times, best_bids, best_asks, title="simulated bid/ask trajectory"):
    plt.figure(figsize=(8, 6))
    plt.step(times, best_bids, where="post", label="Best Bid", color="blue")
    plt.step(times, best_asks, where="post", label="Best Ask", color="red")

    plt.xlabel("time")
    plt.ylabel("price")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()