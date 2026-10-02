/*
 * bench_lob.c -- benchmark harness for lob.c (trader-account API).
 *
 * Every scenario reproduces the exact book state used by the matching JAX-LOB
 * scenario in bench_jax.py, so the per-event numbers compare like with like:
 *
 *   add_only    64 resting bids on 64 levels (1,000,000 - r), add 1 bid @ 999,970
 *   add_new_lvl same book, add 1 bid @ 999,900 (outside range -> side resize)
 *   deep_add    1024 resting bids on 1024 levels,             add 1 bid @ 999,900
 *   add_cancel  same book as add_only, add 1 bid @ 999,970 then cancel (per pair)
 *   fifo_match  100 resting asks of qty 1 at ONE level, buy 1 crossing it
 *               (one incoming order filling one resting order)
 *   fifo_sweep  same book, buy 100 sweeping all of it (per resting order consumed).
 *               lob.c only: JAX-LOB's matcher stops after the first fill.
 *   prorata     100 asks of qty 10..99 at one level, consume ~half (per call)
 *
 * Like JAX-LOB's "normal" mode, NBOOKS independent books are prefilled (untimed)
 * and then the operation is applied to each book in a plain loop (timed).
 *
 * Output: one CSV line per scenario:
 *   name,ns_per_event,bytes_per_book_after,heap_bytes_leaked_per_book
 *
 * Memory accounting wraps malloc/free so we see exactly what lob.c allocates.
 */
#define _GNU_SOURCE
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- heap tracking (must come before lob.c is included) ---- */
static long long g_live_bytes = 0;
static void* tracked_malloc(size_t n) {
    void* p = malloc(n);
    if (p) g_live_bytes += (long long)malloc_usable_size(p);
    return p;
}
static void tracked_free(void* p) {
    if (p) g_live_bytes -= (long long)malloc_usable_size(p);
    free(p);
}
#define malloc tracked_malloc
#define free   tracked_free
#define main   lob_c_builtin_main   /* lob.c ships its own main() */
#include "lob.c"
#undef main
#undef malloc
#undef free

/* Bytes the book has asked for, computed from its capacities (no allocator overhead). */
static size_t lob_footprint_bytes(const LimitOrderBook* lob) {
    size_t total = sizeof(LimitOrderBook);
    const LimitOrderBook_Side* sides[2] = { &lob->bids, &lob->asks };
    for (int s = 0; s < 2; s++) {
        const LimitOrderBook_Side* side = sides[s];
        total += side->price_buckets_capacity * sizeof(LimitOrderBook_PriceBucket);
        for (size_t i = 0; i < side->price_buckets_capacity; i++) {
            const LimitOrderBook_PriceBucket* b = &side->price_buckets[i];
            if (b->orders)           total += b->orders_capacity * sizeof(uint64_t);
            if (b->order_references) total += b->order_references_capacity * sizeof(LimitOrderBook_OrderReference);
        }
    }
    total += lob->total_trader_accounts * sizeof(LimitOrderBook_TraderAccount);
    for (size_t i = 0; i < lob->total_trader_accounts; i++)
        total += lob->trader_accounts[i].order_references_capacity * sizeof(LimitOrderBook_OrderReference*);
    return total;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#define NBOOKS 64   /* small enough that prefilled books are still warm in cache, like JAX's reused state0 */
#define BASE   1000000
#define PREFILL_ACCOUNT 0
#define AGENT_ACCOUNT   1

typedef struct { double ns; size_t bytes; long long leaked; } Result;

/* Run `reps` rounds; each round prefills NBOOKS books, times op() over them. */
typedef void (*PrefillFn)(LimitOrderBook*);
typedef void (*OpFn)(LimitOrderBook*);

static Result run(PrefillFn prefill, OpFn op, int reps, double events_per_op) {
    static LimitOrderBook books[NBOOKS];
    double total = 0;
    Result r = {0};
    for (int rep = 0; rep < reps; rep++) {
        long long heap_before = g_live_bytes;
        for (int b = 0; b < NBOOKS; b++) { books[b] = create_limit_order_book(); prefill(&books[b]); }
        double t0 = now_ns();
        for (int b = 0; b < NBOOKS; b++) op(&books[b]);
        double t1 = now_ns();
        total += t1 - t0;
        r.bytes = lob_footprint_bytes(&books[0]);
        for (int b = 0; b < NBOOKS; b++) destroy_limit_order_book(&books[b]);
        r.leaked = (g_live_bytes - heap_before) / NBOOKS;
    }
    r.ns = total / ((double)reps * NBOOKS * events_per_op);
    return r;
}

/* ---- prefills (mirror bench_jax.py build_state) ---- */
static void prefill_bids_64(LimitOrderBook* lob) {
    for (int r = 0; r < 64; r++)
        add_order_to_limit_order_book(lob, PREFILL_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - r, 10);
    /* make sure the agent account exists before timing */
    add_order_to_limit_order_book(lob, AGENT_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - 63, 1);
}
static void prefill_bids_1024(LimitOrderBook* lob) {
    for (int r = 0; r < 1024; r++)
        add_order_to_limit_order_book(lob, PREFILL_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - r, 10);
    add_order_to_limit_order_book(lob, AGENT_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - 1023, 1);
}
static void prefill_asks_100_one_level(LimitOrderBook* lob) {
    for (int r = 0; r < 100; r++)
        add_order_to_limit_order_book(lob, PREFILL_ACCOUNT, LIMIT_ORDER_BOOK_ASK, BASE, 1);
}
static void prefill_prorata(LimitOrderBook* lob) {
    for (int r = 0; r < 100; r++)
        add_order_to_limit_order_book(lob, PREFILL_ACCOUNT, LIMIT_ORDER_BOOK_ASK, BASE, 10 + (uint64_t)(r % 90));
}

/* ---- timed ops ---- */
/* existing level inside the book's price range */
static void op_add(LimitOrderBook* lob) {
    add_order_to_limit_order_book(lob, AGENT_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - 30, 10);
}
/* new level 100 ticks from best: outside a 64-level book, forces the side to grow */
static void op_add_new_level(LimitOrderBook* lob) {
    add_order_to_limit_order_book(lob, AGENT_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - 100, 10);
}
static void op_add_cancel(LimitOrderBook* lob) {
    add_order_to_limit_order_book(lob, AGENT_ACCOUNT, LIMIT_ORDER_BOOK_BID, BASE - 30, 10);
    size_t idx = lob->trader_accounts[AGENT_ACCOUNT].order_references_tail_index;
    cancel_order_in_limit_order_book(lob, AGENT_ACCOUNT, idx);
}
static void op_fifo_single(LimitOrderBook* lob) {
    uint64_t left = match_limit_order_book_order_using_time_price_priority(lob, LIMIT_ORDER_BOOK_ASK, BASE, 1);
    if (left != 0) { fprintf(stderr, "fifo: expected fill\n"); exit(1); }
}
static void op_fifo(LimitOrderBook* lob) {
    /* buyer crosses the ask side: pass the resting side's direction */
    uint64_t left = match_limit_order_book_order_using_time_price_priority(lob, LIMIT_ORDER_BOOK_ASK, BASE, 100);
    if (left != 0) { fprintf(stderr, "fifo: expected full fill, %llu left\n", (unsigned long long)left); exit(1); }
}
static void op_prorata(LimitOrderBook* lob) {
    uint64_t total = lob->asks.price_buckets[lob->asks.best_price_bucket_index].total_quantity;
    match_limit_order_book_order_using_pro_rata(lob, LIMIT_ORDER_BOOK_ASK, BASE, total / 2);
}

/* ---- scaling probe: cost of one add as one account's resting order count grows ---- */
static void scaling(void) {
    long sizes[] = {1000, 2000, 4000, 8000, 16000};
    for (int s = 0; s < 5; s++) {
        long n = sizes[s];
        LimitOrderBook lob = create_limit_order_book();
        double t0 = now_ns();
        for (long i = 0; i < n; i++)  /* spread over 100 levels, one account */
            add_order_to_limit_order_book(&lob, 0, LIMIT_ORDER_BOOK_BID, BASE - (i % 100), 1);
        double t1 = now_ns();
        /* consume everything with one FIFO sweep */
        double t2 = now_ns();
        match_limit_order_book_order_using_time_price_priority(&lob, LIMIT_ORDER_BOOK_BID, BASE - 99, (uint64_t)n);
        double t3 = now_ns();
        printf("scaling,%ld,%.1f,%.1f,%zu\n", n, (t1 - t0) / n, (t3 - t2) / n, lob_footprint_bytes(&lob));
        destroy_limit_order_book(&lob);
    }
}

/* ---- memory-only probe: footprint vs resting orders (book grows with use) ---- */
static void memory_curve(void) {
    long sizes[] = {0, 1, 64, 1024, 2048};
    for (int s = 0; s < 5; s++) {
        LimitOrderBook lob = create_limit_order_book();
        for (long r = 0; r < sizes[s]; r++)
            add_order_to_limit_order_book(&lob, 0, LIMIT_ORDER_BOOK_BID, BASE - r, 10);
        printf("memory,%ld,%zu\n", sizes[s], lob_footprint_bytes(&lob));
        destroy_limit_order_book(&lob);
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* which = (argc > 1) ? argv[1] : "all";
    int all = !strcmp(which, "all");
    Result r;
#define SCN(name, pre, op, reps, ev) \
    if (all || !strcmp(which, name)) { r = run(pre, op, reps, ev); \
        printf("%s,%.2f,%zu,%lld\n", name, r.ns, r.bytes, r.leaked); }
    SCN("add_only",   prefill_bids_64,            op_add,           3000, 1)
    SCN("add_new_level", prefill_bids_64,         op_add_new_level, 3000, 1)
    SCN("deep_add",   prefill_bids_1024,          op_add,             80, 1)
    SCN("add_cancel", prefill_bids_64,            op_add_cancel,    3000, 1)
    SCN("fifo_match", prefill_asks_100_one_level, op_fifo_single,   3000, 1)
    SCN("fifo_sweep", prefill_asks_100_one_level, op_fifo,           800, 100)
    SCN("prorata",    prefill_prorata,            op_prorata,        800, 1)
    if (all || !strcmp(which, "scaling")) scaling();
    if (all || !strcmp(which, "memory"))  memory_curve();
    return 0;
}
