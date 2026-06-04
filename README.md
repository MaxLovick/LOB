# lob.c

A high-performance limit order book in C, built to be inside of reinforcement-learning environments.

`lob.c` is a single self-contained file with no dependencies beyond the C standard library. Add, cancel, and partial match are all constant-time, and submitting an order hands you back a direct pointer into the book so that cancelling it later is O(1) with no lookup. It supports both FIFO time-price priority and pro-rata matching, each with limit prices and fill-or-kill.

## Goal
Most GPU-batched simulators (JAX-LOB and friends) win by running thousands of independent books in parallel and amortizing per-step overhead across the batch. That is the right tool when your environment is "10,000 independent traders." It is the wrong tool when your environment is a tens to hundreds of agents interacting in a few books, there is nothing to batch, and the per-step dispatch cost dominates.

`lob.c` targets that second case: a small number of books, each stepped serially at the lowest possible latency per event, cheap enough to run interesting multi-agent experiments on a normal desktop.

While it is designed to be put into a reinforcement learning environment, but without a GPU this will run faster than JAX-LOB at most reasonable batch sizes.

## Features

- **O(1) common path** for add, cancel, and partial match.
- **`OrderReference` handles** — every add returns a pointer into the book, so cancel is a direct dereference (~17 ns warm) instead of a search.
- **Two matching algorithms** — FIFO time-price priority and pro-rata, selectable per book. Both support a limit price and fill-or-kill. Pro-rata additionally takes a minimum-fill threshold and runs a FIFO leftover pass to clean up the rounding remainder.
- **Automatic bookkeeping** — best/worst price tracked on each side, per-bucket total quantity and order counts, and on-demand resizing of both the price array and the per-level order arrays.
- **Single file, no dependencies** — only `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, `<stdlib.h>`. Should compile anywhere.

## Architecture

The book is a two-level circular-buffer layout, with both layers growing on demand:

- A **`Side`** is a circular buffer of `PriceBucket`s, indexed by tick offset from the best price. Best-price advancement on a cancel or fill walks the price array forward until it finds the next non-empty bucket.
- A **`PriceBucket`** is a circular buffer of `Order`s held in time priority (FIFO within a level).

This is what keeps the hot operations constant-time: prices are reached by offset rather than by tree traversal, and orders are reached by reference rather than by search.

## Build and quick start

No build system required — just include the file.

```c
#include <stdio.h>
#include <inttypes.h>
#include "lob.c"

int main(void) {
    LimitOrderBook lob = construct_limit_order_book(/*tick_size=*/1);

    // Resting liquidity from other participants.
    add_order_to_limit_order_book(&lob, true,  99,  50);
    add_order_to_limit_order_book(&lob, true,  98,  100);
    add_order_to_limit_order_book(&lob, false, 101, 75);
    add_order_to_limit_order_book(&lob, false, 102, 50);

    // The RL agent posts its own bid. Hold on to the returned reference —
    // it is a direct pointer into the book, so cancelling is O(1) with no lookup.
    OrderReference* my_order = add_order_to_limit_order_book(&lob, true, 100, 30);
    printf("after agent quote -> best bid: %" PRId64 ", best ask: %" PRId64 "\n",
           lob.bids.best_price, lob.asks.best_price);

    // Agent re-quotes: cancel via the saved reference.
    cancel_order(&lob.bids,
                 my_order->is_bid,
                 my_order->price,
                 my_order->order_index,
                 lob.tick_size);
    printf("after agent cancel -> best bid: %" PRId64 "\n", lob.bids.best_price);

    // Agent sends a marketable sell of 70. execute_order matches against the
    // resting bids using the book's active algorithm and rests any remainder.
    execute_order(&lob, /*is_bid=*/false, /*price=*/0, /*quantity=*/70,
                  /*is_fill_or_kill=*/false, /*minimum_fill_quantity=*/0);
    printf("after market sell 70 -> best bid: %" PRId64 "\n", lob.bids.best_price);

    destroy_limit_order_book(&lob);
    return 0;
}
```

Compile:

```sh
gcc -O2 quickstart.c -o quickstart && ./quickstart
```

## API

| Function | Purpose |
|---|---|
| `LimitOrderBook construct_limit_order_book(int64_t tick_size)` | Create an empty book. |
| `OrderReference* add_order_to_limit_order_book(LimitOrderBook*, bool is_bid, int64_t price, uint64_t quantity)` | Rest an order; returns a handle for O(1) cancel. |
| `void cancel_order(Side*, bool is_bid, int64_t price, size_t order_index, int64_t tick_size)` | Remove an order via its reference fields. |
| `OrderReference* execute_order(LimitOrderBook*, bool is_bid, int64_t price, uint64_t quantity, bool is_fill_or_kill, uint64_t minimum_fill_quantity)` | Match a marketable order under the book's active algorithm, resting any unfilled remainder (unless fill-or-kill). |
| `uint64_t match_market_order_using_time_price_priority(Side*, bool is_bid, int64_t tick_size, int64_t limit_price, uint64_t quantity, bool is_fill_or_kill)` | Low-level FIFO matcher; returns the unfilled quantity. |
| `void destroy_limit_order_book(LimitOrderBook*)` | Free the book. |

The matching algorithm is selected per book via `lob.order_matching_algorithm`, set to either `ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY` or `ORDER_MATCHING_ALGORITHM_PRO_RATA`. In pro-rata mode, `execute_order`'s `minimum_fill_quantity` argument sets the per-order floor before the FIFO leftover pass distributes the rounding remainder.

## Benchmarks

Compared against [JAX-LOB](https://arxiv.org/abs/2308.13289), a JAX-based order book built for GPU-batched RL training. Both engines were run on the same machine: a single core of an Intel Xeon @ 2.80 GHz, no GPU. `lob.c` was built with `gcc -O2`; JAX was 0.10.0 on CPU.

This is a CPU comparison, and it is deliberately *not* the workload JAX-LOB was designed for (batching thousands of independent books on a GPU). The single-book numbers show the cost of JAX dispatch and array scans with nothing to amortize them against — which is exactly the regime this library targets.

**Single book, single thread, CPU:**

| Scenario | `lob.c` (ns/op) | JAX-LOB (ns/op) | Ratio |
|---|---:|---:|---:|
| add + cancel pair (warm book) | 17 | 100K – 165K | ~5,900x – 9,700x |
| add only | 330 | 123K | ~370x |
| FIFO match, per resting order consumed | 9 – 13 | 407 – 1,088 | ~40x – 100x |
| pro-rata, 100 orders @ one level | 370 | — | — |
| mixed (47% add / 48% cancel / 5% match) | 73 | — | — |
| deep add: 50K orders to one bucket (per add) | 140 | — | — |

**vmap batching (JAX-LOB's intended use case, still on CPU):**

| Batch | nOrders | Per-event amortized | Events/sec |
|---:|---:|---:|---:|
| 256 | 128 | 23,211 ns | 43,000 |
| 1024 | 128 | 40,564 ns | 25,000 |

Even amortized across 1,024 parallel books, JAX-LOB on CPU is roughly 2,400x slower per event than the C engine on a single book. On a GPU that dispatch overhead largely vanishes and batched throughput closes the gap — that is the regime JAX-LOB targets and where it is likely to win for RL training on large numbers of parallel environments. This library bets on the opposite end: a few books, stepped as fast as a single core allows.

## Roadmap

- Include normal size statistics, so the book grows and shrinks adaptively for better memory use and performance.
- Add opening and closing auctions.
- An events array tracking all book and exchange events.
- A stylized-facts display resembling a realistic exchange.
- Save/restore, for warm-starting RL episodes from a stored book state.
- A [LOBSTER](https://lobsterdata.com/) message-file driver for replay.
- A synthetic order-flow simulator driven by Hawkes processes.

## Research directions

The experiment I am most curious about is using this inside of an environment similar to the world in [*Emergent Bartering Behaviour in Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/2205.06760). 
