# lob.c

A high-performance limit order book in C, built to be put inside reinforcement-learning environments.

## Goal
Most GPU-batched simulators (JAX-LOB and friends) win by running thousands of independent books in parallel and amortizing per-step overhead across the batch. That is the right tool when your environment is "10,000 independent traders." It is the wrong tool when your environment is tens to hundreds of agents interacting in a few books: there is nothing to batch, and the per-step dispatch cost dominates.

`lob.c` targets that second case: a small number of books, each stepped serially at the lowest possible latency per event, cheap enough to run interesting multi-agent experiments on a normal desktop.

While it is designed to be put into a reinforcement learning environment, without a GPU it runs faster than JAX-LOB at every batch size measured below.

## Features

- **Tick-offset price lookup** — a price is found by its offset from the best price, not by searching.
- **Trader accounts** — every order is owned by an account, and an account holds references to all of its resting orders. Accounts are created on first use.
- **Two matching algorithms** — FIFO time-price priority and pro-rata (with a FIFO leftover pass for the rounding remainder), both with a limit price.
- **Dynamic memory** — price levels, per-level order arrays and per-account reference arrays all grow on demand; `shrink_limit_order_book` gives memory back.
- **Automatic bookkeeping** — best/worst price tracked on each side, per-level total quantity and order counts.
- **Single file, no dependencies** — only `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, `<stdio.h>`, `<stdlib.h>`. Should compile anywhere.

## Architecture

The book is a two-level circular-buffer layout, with both layers growing on demand:

- A **`LimitOrderBook_Side`** is a circular buffer of `LimitOrderBook_PriceBucket`s, indexed by tick offset from the best price. Best-price advancement on a cancel or fill walks the price array forward until it finds the next non-empty bucket.
- A **`LimitOrderBook_PriceBucket`** is a circular buffer of order quantities held in time priority (FIFO within a level), plus a parallel array of `LimitOrderBook_OrderReference`s.
- A **`LimitOrderBook_TraderAccount`** holds pointers to its orders' references, so an agent can find and cancel its own orders.

## API

| Function | Purpose |
|---|---|
| `LimitOrderBook create_limit_order_book()` | Create an empty book (`tick_size` = 1; set `lob.tick_size` to change it). |
| `void add_order_to_limit_order_book(LimitOrderBook*, size_t trader_account_index, int8_t price_direction, int64_t price, uint64_t quantity)` | Rest an order for an account. `price_direction` is `LIMIT_ORDER_BOOK_BID` or `LIMIT_ORDER_BOOK_ASK`. |
| `void cancel_order_in_limit_order_book(LimitOrderBook*, size_t trader_account_index, size_t order_reference_index)` | Cancel the order in that slot of the account's reference array. |
| `uint64_t match_limit_order_book_order_using_time_price_priority(LimitOrderBook*, int8_t price_direction, int64_t limit_price, uint64_t quantity)` | FIFO match against the side given by `price_direction` (the resting side being consumed). Returns the unfilled quantity. |
| `uint64_t match_limit_order_book_order_using_pro_rata(LimitOrderBook*, int8_t price_direction, int64_t limit_price, uint64_t quantity)` | Pro-rata match, then a FIFO pass for the rounding remainder. Returns the unfilled quantity. |
| `uint64_t get_total_quantity_at_price_in_limit_order_book(LimitOrderBook*, int8_t price_direction, int64_t price)` | Resting quantity at one price. |
| `void shrink_limit_order_book(LimitOrderBook*)` | Trim both sides down to their live price range and compact each level. |
| `void destroy_limit_order_book(LimitOrderBook*)` | Free the book. |

## Roadmap

- Include normal size statistics, so the book grows and shrinks adaptively for better memory use and performance.
- Add opening and closing auctions.
- An events array tracking all book and exchange events.
- A stylized-facts display resembling a realistic exchange.
- Save/restore, for warm-starting RL episodes from a stored book state.

# Benchmarks

Compared against [JAX-LOB](https://github.com/KangOxford/AlphaTrade), a JAX-based order
book built for GPU-batched RL training. Both engines ran on the same machine: a single
core of an Intel Xeon @ 2.80 GHz, no GPU. `lob.c` was built with `gcc -O2`; JAX-LOB ran
on JAX 0.11.2 (CPU). This is a CPU comparison and is deliberately **not** the workload
JAX-LOB was designed for (batching thousands of books on a GPU).

Every scenario starts both engines from the **same book state**: the same resting orders at the same prices, then the same incoming message.

| Scenario | Book before the timed operation | Timed operation |
|---|---|---|
| add (existing level) | 64 bids on 64 levels | add 1 bid inside the price range |
| add (new level, book grows) | 64 bids on 64 levels | add 1 bid 100 ticks from best (outside `lob.c`'s range, so the side resizes) |
| deep add | 1,024 bids on 1,024 levels | add 1 bid inside the price range |
| add + cancel | 64 bids on 64 levels | add 1 bid, then cancel it |
| FIFO match | 100 one-lot asks at one price | 1 buy that fills the front order |
| FIFO sweep (`lob.c` only) | 100 one-lot asks at one price | 1 buy of 100 that sweeps the level |
| pro-rata (`lob.c` only) | 100 asks of 10–99 lots at one price | a buy for half the level |

JAX-LOB is measured two ways, at batch sizes 1 / 256 / 512 / 1028:

* **normal** — no vmap: the books are processed one at a time (a plain Python loop of
  single-book calls). Per-event cost is ~independent of batch; total time scales with B.
* **vmap** — `jax.vmap` across all B books in one vectorised call.

`lob.c` prefills 64 independent books and then applies the operation to each in a loop, so the books are warm in cache. Each `lob.c` number is the median of 7 isolated runs (own process). The ratio after every JAX cell is **how many times faster `lob.c` is**. Units: add / deep add = ns per order; add + cancel = ns per pair; FIFO match = ns per incoming order; FIFO sweep = ns per resting order consumed; pro-rata = ns per match call.

| Scenario | lob.c (ns) | JAX B=1 | speed-up | JAX B=256 | speed-up | JAX B=512 | speed-up | JAX B=1028 | speed-up | JAX vmap B=1 | speed-up | JAX vmap B=256 | speed-up | JAX vmap B=512 | speed-up | JAX vmap B=1028 | speed-up |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| add (existing level) | 121 | 235,859 | 1,950x | 287,937 | 2,381x | 236,265 | 1,954x | 288,998 | 2,390x | 348,893 | 2,885x | 30,125 | 249x | 35,404 | 293x | 62,087 | 513x |
| add (new level, book grows) | 4,501 | 263,148 | 58x | 254,169 | 56x | 293,858 | 65x | 295,369 | 66x | 255,767 | 57x | 30,996 | 6.9x | 35,833 | 8.0x | 60,192 | 13x |
| deep add (1,024-level book) | 617 | 743,977 | 1,206x | 848,167 | 1,375x | 777,789 | 1,261x | 785,655 | 1,274x | 776,927 | 1,259x | 950,401 | 1,541x | 984,777 | 1,596x | 1,101,341 | 1,785x |
| add + cancel (per pair) | 172 | 513,148 | 2,978x | 625,509 | 3,630x | 1,109,452 | 6,438x | 656,964 | 3,812x | 456,146 | 2,647x | 67,926 | 394x | 69,356 | 402x | 82,085 | 476x |
| FIFO match (1 incoming fills 1 resting) | 396 | 304,537 | 770x | 352,091 | 890x | 341,126 | 862x | 369,497 | 934x | 251,579 | 636x | 38,290 | 97x | 43,215 | 109x | 79,609 | 201x |
| FIFO sweep of 100 orders (per order) | 75 | n/a (stops after 1 fill) | - | n/a | - | n/a | - | n/a | - | n/a | - | n/a | - | n/a | - | n/a | - |
| pro-rata (per match call) | 1,526 | n/a (not in JAX-LOB) | - | n/a | - | n/a | - | n/a | - | n/a | - | n/a | - | n/a | - | n/a | - |

### Reading the table

* **Single-book / sequential JAX-LOB is ~1,000x–6,000x slower per event** for add and add + cancel. That is the cost of JAX dispatch plus fixed-size array scans with nothing to amortise against.
* **vmap narrows the gap at moderate batch.** At B=256 add is ~250x, add + cancel ~400x and FIFO match ~100x. On a single CPU core the gap widens again by B=1028: there is no parallel hardware, so a bigger batch just adds serial work.
* **Growing the book is `lob.c`'s most expensive add** (~4.5 µs): an order outside the current price range rebuilds the side's price array. JAX-LOB pays nothing extra here because its array is already allocated, which is why this is the closest row (~7x at vmap B=256).
* **deep add is the one case where vmap is *worse* than sequential.** JAX-LOB scans the whole fixed-size side on every message; vmapping on one core does B of those scans back to back.
* **FIFO sweep and pro-rata have no JAX-LOB equivalent.** JAX-LOB's matcher only fills one resting order per incoming message: `match_ask_order` / `match_bid_order` recompute the top-of-book index from the array as it was *before* the fill, so the `while_loop` exits after the first trade. Pro-rata is not implemented in JAX-LOB.

On a GPU the dispatch overhead behind the `normal` columns largely vanishes and the `vmap`
columns scale with the hardware — that is the regime JAX-LOB targets.

## Memory

JAX-LOB allocates a fixed-capacity array per side up front, so it has to be sized for the highest volume in the entire session. If a side fills up, the next order silently overwrites the last resting order. `lob.c` allocates as orders arrive, so you never choose a capacity and never lose an order. `shrink_limit_order_book` trims each side back to its live price range after a busy period. Per resting order, `lob.c` currently uses more memory than a fully packed JAX-LOB array; reducing per-level overhead is on the roadmap.

## Research directions

The experiment I am most curious about is using this inside of an environment similar to the world in [*Emergent Bartering Behaviour in Multi-Agent Reinforcement Learning*](https://arxiv.org/abs/2205.06760).
