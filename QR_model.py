import numpy as np
from typing import Callable, List, Tuple
from enum import Enum


class Queue:
    def __init__(self, price: float, size: int):
        self.price = price
        self.size = size

class OrderBook:
    def __init__(self, p_ref: float, states: List[Queue], k: int, tick: float):
        self.p_ref = p_ref # Ref price
        self.states = states # Bid/Ask list: 0..k-1 for one side, k..2k-1 for the other side
        self.k = k # depth of orderbook

        self.tick_size = tick #tick size

        # build/store intensities once
        self.lambda_funcs, self.mju_funcs, self.market_order_func = self.make_intensity_functions(k=k)
        self.rebuild_prices()
    
    def rebuild_prices(self):
        """
        Rebuild all queue prices from current p_ref.
        Indices:
        0..k-1   = bids
        k..2k-1  = asks
        Best bid = index k-1 = p_ref - tick/2
        Best ask = index k   = p_ref + tick/2
        """
        # bids
        for i in range(self.k):
            dist = (self.k - 1) - i
            self.states[i].price = self.p_ref - self.tick_size / 2 - dist * self.tick_size

        # asks
        for i in range(self.k, 2 * self.k):
            dist = i - self.k
            self.states[i].price = self.p_ref + self.tick_size / 2 + dist * self.tick_size
            
    def get_best_bid_index(self):
        for i in range(self.k - 1, -1, -1):
            if self.states[i].size > 0:
                return i
        return None

    def get_best_ask_index(self):
        for i in range(self.k, 2 * self.k):
            if self.states[i].size > 0:
                return i
        return None

    def get_best(self):
        """
        Looking for the current best bid/ask  
        Called functions look for the first none emptied queues of each side   
        """
        bid_idx = self.get_best_bid_index()
        ask_idx = self.get_best_ask_index()

        bid = self.states[bid_idx].price if bid_idx is not None else None
        ask = self.states[ask_idx].price if ask_idx is not None else None
        return bid, ask
    
    def sample_new_queue_size(self, side: str) -> int:
        """
        New queue inserted at the far end when book shifts.
        """
        lam = 2.0
        return max(1, np.random.poisson(lam))
    
    def update_reference_price(self):
        """
        Active p_ref updates using current ask/bid spread
        """
        bid, ask = self.get_best()
        if bid is not None and ask is not None:
            self.p_ref = 0.5 * (bid + ask)

    def shift_bid_book(self):
        """
        Best bid depleted:
        - bid side shifts toward the touch
        - insert new far bid size
        - p_ref moves up
        """
        bid_sizes = [q.size for q in self.states[:self.k]]

        # old Q-2 becomes new Q-1 etc...
        new_far_bid_size = self.sample_new_queue_size("Bid")
        new_bid_sizes = [new_far_bid_size] + bid_sizes[:-1]

        for i in range(self.k):
            self.states[i].size = new_bid_sizes[i]

        # mid price moves 
        self.p_ref -= self.tick_size / 2
        self.rebuild_prices()

    def shift_ask_book(self):
        """
        Best ask depleted:
        - ask side shifts toward the touch
        - insert new far ask size
        - p_ref moves down
        """
        ask_sizes = [q.size for q in self.states[self.k:]]

        new_far_ask_size = self.sample_new_queue_size("Ask")
        new_ask_sizes = ask_sizes[1:] + [new_far_ask_size]

        for i in range(self.k):
            self.states[self.k + i].size = new_ask_sizes[i]

        self.p_ref += self.tick_size / 2
        self.rebuild_prices()

    def handle_depletions(self):
        """
        Keep k queues per side at all times and move p_ref dynamically.
        """
        # If best bid is empty recenter downward 
        if self.states[self.k - 1].size == 0:
            self.shift_bid_book()

        # If best ask is empty recenter upward 
        if self.states[self.k].size == 0:
            self.shift_ask_book()
    

    def update_state(self, lambda_: List[Callable], mu_: List[Callable], stf: int, Cbound: int, delta: float, H: float):
        for i in range(self.k * 2):
            size = self.states[i].size

            if i < self.k:
                same_best_idx = self.k - 1
                opp_best_idx = self.k
            else:
                same_best_idx = self.k
                opp_best_idx = self.k - 1

            same_best_empty = (self.states[same_best_idx].size == 0)
            opp_best_size = self.states[opp_best_idx].size

            lam = lambda_[i](size, same_best_empty, opp_best_size)
            mu_i = mu_[i](size, same_best_empty, opp_best_size)

            s_lambda = np.random.poisson(lam * stf)
            s_mu = np.random.poisson(mu_i * stf)

            proposed_change = s_lambda - s_mu

            if size > Cbound:
                proposed_change -= delta

            total_incoming_flow = sum(
                np.random.poisson(
                    lambda_[j](
                        self.states[j].size,
                        self.states[self.k - 1].size == 0 if j < self.k else self.states[self.k].size == 0,
                        self.states[self.k].size if j < self.k else self.states[self.k - 1].size
                    ) * stf
                )
                for j in range(self.k * 2) if j != i
            )

            if total_incoming_flow > H:
                proposed_change = min(proposed_change, H - size)

            self.states[i].size = max(0, size + proposed_change)

    def make_intensity_functions(
        self,
        k: int,
        # Can be totally optimized, approx per Rosenbaum's HFT course
        # Far queues
        far_insert_floor: float = 0.35,
        far_cancel_floor: float = 0.08,
        far_cancel_slope: float = 0.015,

        # Q1
        q1_insert_low: float = 0.35,
        q1_insert_high: float = 0.82,
        q1_cancel_low: float = 0.05,
        q1_cancel_high: float = 0.95,
        q1_market_base: float = 0.04,
        q1_market_opp_boost_small: float = 0.02,
        q1_market_opp_boost_mid: float = 0.05,
        q1_market_opp_boost_large: float = 0.10,

        # Q2
        q2_insert_when_q1_empty: float = 0.82,
        q2_insert_when_q1_full: float = 0.35,
        q2_cancel_when_q1_empty: float = 1.20,
        q2_cancel_when_q1_full: float = 0.38,
        q2_market_when_q1_empty: float = 0.01,
        q2_market_when_q1_full: float = 0.00,
    ) -> Tuple[List[Callable], List[Callable], Callable]:

        lambda_: List[Callable] = []
        mu_: List[Callable] = []

        def opp_bucket(opp_best_size: int) -> int:
            if opp_best_size <= 3:
                return 0
            elif opp_best_size <= 8:
                return 1
            return 2

        for i in range(2 * k):
            if i < k:
                dist = (k - 1) - i
            else:
                dist = i - k

            # LIMIT 
            def make_lambda(dist: int):
                def lam(size: int, same_best_empty: bool, opp_best_size: int) -> float:
                    s = float(max(0, size))
                    b = opp_bucket(opp_best_size)

                    # Q1
                    if dist == 0:
                        if b == 0:
                            high = q1_insert_high
                        elif b == 1:
                            high = 0.55
                        else:
                            high = 0.58

                        val = q1_insert_low + (high - q1_insert_low) * (1.0 - np.exp(-s / 8.0))
                        val -= 0.002 * max(0.0, s - 25.0)
                        return max(0.02, val)

                    # Q2
                    elif dist == 1:
                        if same_best_empty:
                            val = q2_insert_when_q1_empty + 0.12 * (1.0 - np.exp(-s / 15.0))
                        else:
                            val = q2_insert_when_q1_full + 0.35 * np.exp(-s / 4.0)
                        return max(0.02, val)

                    # Far queues:
                    else:
                        val = far_insert_floor + 0.08 * np.exp(-s / 10.0) + 0.03 / (1.0 + dist)
                        return max(0.02, val)

                return lam

            # Removal
            def make_mu(dist: int):
                def mu(size: int, same_best_empty: bool, opp_best_size: int) -> float:
                    s = float(max(0, size))
                    b = opp_bucket(opp_best_size)

                    # Q1
                    if dist == 0:
                        cancel = q1_cancel_low + (q1_cancel_high - q1_cancel_low) * (1.0 - np.exp(-s / 7.0))

                        if b == 0:
                            market = q1_market_base + q1_market_opp_boost_small * np.exp(-s / 3.0)
                        elif b == 1:
                            market = q1_market_base + q1_market_opp_boost_mid * np.exp(-s / 3.0)
                        else:
                            market = q1_market_base + q1_market_opp_boost_large * np.exp(-s / 3.0)

                        return max(0.02, cancel + market)

                    # Q2
                    elif dist == 1:
                        if same_best_empty:
                            cancel = q2_cancel_when_q1_empty - 0.25 * np.exp(-s / 8.0)
                            market = q2_market_when_q1_empty + 0.02 * (1.0 - np.exp(-s / 20.0))
                        else:
                            cancel = q2_cancel_when_q1_full + 0.03 * (1.0 - np.exp(-s / 10.0))
                            market = q2_market_when_q1_full
                        return max(0.02, cancel + market)

                    # Far queues
                    else:
                        cancel = far_cancel_floor + far_cancel_slope * np.sqrt(s) + 0.01 * dist
                        return max(0.02, cancel)

                return mu

            lambda_.append(make_lambda(dist))
            mu_.append(make_mu(dist))

        # market orders in the book
        def market_order_func(best_size: int) -> float:
            s = float(max(0, best_size))
            return 0.08 + 2.8 * s / (10.0 + s)

        return lambda_, mu_, market_order_func


    def process_market_orders(self, market_order_func: Callable):
        """
        Process the market orders by looking at the queues
        """
        for side in ["Bid", "Ask"]:
            idx = self.k - 1 if side == "Bid" else self.k
            if self.states[idx].size > 0:
                intensity = market_order_func(self.states[idx].size)
            else:
                intensity = market_order_func(self.states[idx + (1 if side == "Ask" else -1)].size)
            
            market_orders = np.random.poisson(intensity)
            self.execute_market_orders(side, market_orders)

    def execute_market_orders(self, side, quantity: int):
        """
        Execute the orders by looking at either bid or ask
        """
        idx = self.k - 1 if side == "Bid" else self.k
        while quantity > 0 and idx >= 0 and idx < len(self.states):
            if self.states[idx].size > 0:
                executed = min(quantity, self.states[idx].size)
                self.states[idx].size -= executed
                quantity -= executed
            idx += (1 if side == "Ask" else -1)

    def step(self, stf: int, Cbound: int, delta: float, H: float):
        """
        Auto update of the order book
        """
        self.update_state(self.lambda_funcs, self.mju_funcs, stf, Cbound, delta, H)
        self.handle_depletions()
        self.process_market_orders(self.market_order_func)
        
        