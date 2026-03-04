# Queue-Reactive Limit Order Book Simulation
Queue-reactive limit order book simulation with state-dependent intensities, market order dynamics, and endogenous price formation. Application of the paper : Simulating and analyzing order book data: The queue-reactive model Huang,  Lehalle and Rosenbaum

This repository contains a small Python project for simulating a **queue-reactive limit order book (LOB)** with **state-dependent intensities**, **market order dynamics**, and **endogenous price moves**.

The project was built to explore how local liquidity conditions in the book can drive queue evolution, bid/ask dynamics, spread changes, and reference price updates.

## Files

* **`QR_model.py`**
  Core implementation of the limit order book model:

  * queue representation
  * order book state
  * state-dependent limit/cancel/market order intensities
  * queue updates
  * best-queue depletion logic
  * endogenous reference price changes

* **`QR_plot.py`**
  Plotting and simulation utilities:

  * simulate bid/ask trajectories
  * visualize best bid and best ask over time
  * generate graphical outputs from simulated paths

* **`main.ipynb`**
  Jupyter notebook used to run experiments, test parameter regimes, and generate plots.

## What the model does

The order book is represented as a finite set of bid and ask queues.
At each simulation step:

1. queue sizes are updated using **state-dependent Poisson intensities**,
2. market orders may consume liquidity at the best quotes,
3. when the best bid or ask queue is depleted, the book is shifted,
4. the reference price is updated endogenously.

This makes it possible to simulate short-term bid/ask trajectories and study how different intensity specifications affect market behavior.

## Main features

* Queue-reactive limit order book model
* State-dependent insertion and removal intensities
* Explicit market order simulation
* Endogenous price formation through best-queue depletion
* Bid/ask trajectory plotting
* Easy parameter tuning for different market regimes

## How to use

Open **`main.ipynb`** and run the notebook cells to:

* initialize an order book,
* choose model parameters,
* simulate the book over time,
* plot bid/ask trajectories under different regimes.

A typical workflow is:

1. create an `OrderBook` object from `QR_model.py`,
2. call the simulation functions from `QR_plot.py`,
3. visualize the resulting paths.

## Example use case

This project can be used to:

* study queue depletion and price changes,
* compare low-activity and high-activity regimes,
* visualize spread dynamics,
* experiment with microstructure-inspired intensity functions.

## Notes

This is a research / educational project rather than a production-ready trading engine.
The model is intentionally stylized and designed to remain interpretable and easy to modify.

## Possible extensions

Some natural next steps would be:

* empirical calibration of the intensities,
* richer spread dynamics,
* Hawkes-process-based order flow,
* neural point process extensions,
* more advanced order book visualizations.
