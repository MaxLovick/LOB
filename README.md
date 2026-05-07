# LOB

A high performance limit order book in C, designed to be used in reinforcement learning environments.

## Design choices
- **O(1) common path.** Add, cancel, and partial match are all constant-time in normal cases. The `OrderReference` pointer makes cancel especially fast: ~17 ns in the best case.
- **Two-level circular buffer layout.** A `Side` is a circular buffer of `PriceBucket`s indexed by tick offset from best price; each `PriceBucket` is a circular buffer of `Order`s in time priority. Both layers grow on demand. Best-price advancement on cancel/fill walks the price array forward until it finds the next non-empty bucket.
- **Order Reference** When an order is submitted, the function returns a pointer to the location of that order so that if it needs to be cancelled, it can be deleted in O(1).
- **Matching algorithms.** Both FIFO time-price priority and pro-rata are included, each supporting a limit price and fill-or-kill. Pro-rata also takes a minimum-fill threshold and runs a FIFO leftover pass to clean up the rounding remainder.
- **Single file, no dependencies.** Just `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, `<stdlib.h>`. Compiles on anything.


## What's included
**Core operations:** add order, cancel order, FIFO match, pro-rata match, destroy.
**Matcher options:** limit price, fill-or-kill, pro-rata minimum-fill threshold with FIFO leftover.
**Bookkeeping:** automatic best/worst price tracking on each side, automatic price-bucket and order-array resizing on demand, total quantity and order counts maintained per bucket.

## Quick start

```c
#include <stdio.h>
#include <inttypes.h>
#include "lob.c"

int main(void) {
    LimitOrderBook lob = construct_limit_order_book(1);

    // Some resting liquidity from other participants.
    add_order_to_limit_order_book(&lob, true,  99, 50);
    add_order_to_limit_order_book(&lob, true,  98, 100);
    add_order_to_limit_order_book(&lob, false, 101, 75);
    add_order_to_limit_order_book(&lob, false, 102, 50);

    // The RL agent places its own bid. Hold on to the returned reference —
    // it's a direct pointer into the book, making cancel O(1) with no lookup.
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

    // Market sell of 70 into the bid side using FIFO time priority.
    uint64_t unfilled = match_market_order_time_price_priority(
        &lob.bids, /*is_bid=*/true, lob.tick_size,
        /*quantity=*/70, /*limit_price=*/0, /*fill_or_kill=*/false);
    printf("market sell 70 -> filled %" PRIu64 ", unfilled %" PRIu64 ", best bid: %" PRId64 "\n",
           70 - unfilled, unfilled, lob.bids.best_price);

    destroy_limit_order_book(&lob);
    return 0;
}
```

## Benchmarks

Compared against JAX-LOB, a JAX-based limit order book built for GPU-batched RL training. Both engines were run on the same machine: a single core of an Intel Xeon @ 2.80 GHz, no GPU. The C engine was built with `gcc -O2`. JAX was 0.10.0 on CPU.

This is a CPU comparison and is not the workload JAX-LOB was designed for (batching thousands of independent books on a GPU). The single-book CPU numbers below show the cost of JAX dispatch and array scans without batching to amortize them.

**Single book, single thread, CPU:**

| Scenario                                       | lob.c (ns/op) | JAX-LOB (ns/op) |        Ratio |
| ---------------------------------------------- | ------------: | --------------: | -----------: |
| add + cancel pair (warm book)                  |            17 |     100K – 165K | ~5,900×–9,700× |
| add only                                       |           330 |            123K |        ~370× |
| FIFO match, per resting order consumed         |          9–13 |        407–1088 |    ~40×–100× |
| Pro-rata, 100 orders @ one level               |           370 |               — |            — |
| Mixed (47% add / 48% cancel / 5% match)        |            73 |               — |            — |
| Deep add: 50K orders to one bucket (per add)   |           140 |               — |            — |

**vmap batching (JAX-LOB's intended use case, still on CPU):**

| Batch | nOrders | Per-event amortized | Events/sec |
| ----: | ------: | ------------------: | ---------: |
|   256 |     128 |          23,211 ns  |     43,000 |
|  1024 |     128 |          40,564 ns  |     25,000 |

Even amortized across 1024 parallel books, JAX-LOB on CPU is roughly 2,400× slower per event than the C engine on a single book. On a GPU the dispatch overhead largely vanishes and the batched throughput is much closer; that's the regime JAX-LOB targets and the regime where it's likely to win for RL training on large numbers of parallel environments. But this LOB is used for multi player RL loops where there are only a few objects at a time.

## Future improvements
This is a first draft.  Planning to make the code simpler and add these features.
- Resize increment statistics to expand and shrink the order book effectively to improve performance and memory usage
- `modify_order(ref, new_qty)` with time-priority preservation rules
- Opening and closing auctions
- Add and events array to track all of the book and exchange events
- Add a stylized facts display similar to what would be shown in a realistic exchange
- Iceberg / hidden-quantity orders
- Restore for warm-starting RL episodes from a saved book state
- A LOBSTER message-file driver for replay
- A synthetic order book simulator with Hawkes Processes

## Research directions

The thing I'm most curious to try is using this as the inner loop of an RL environment similar to the paper "Emergent Bartering Behavior in Multi-Agent Reinforcement Learning", training an agent that gets the order pointer directly and seeing if it is possible to speed up the environment enough to run something interesting on a normal desktop computer. 
