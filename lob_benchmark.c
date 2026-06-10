#include <stdio.h>
#include <time.h>
#include <string.h>
#include "lob.c"

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* construct_limit_order_book does not set the algorithm field, so do it here. */
static LimitOrderBook make_book(OrderMatchingAlgorithm algo, int64_t tick) {
    LimitOrderBook lob = construct_limit_order_book(tick);
    lob.tick_size = tick;
    lob.order_matching_algorithm = algo;
    return lob;
}

/* ---------- 1. add + cancel pair, warm book ---------- */
double bench_add_cancel_warm(long pairs) {
    LimitOrderBook lob = make_book(ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY, 1);
    /* Warm the book: resting orders across a band, and 10 resting at the target level
       so total_orders never drops to 0 during the measured loop (no best/worst recompute). */
    int64_t base = 1000000;
    for (int64_t p = base - 50; p <= base + 50; p++)
        add_order_to_limit_order_book(&lob, true, p, 5);
    for (int i = 0; i < 10; i++)
        add_order_to_limit_order_book(&lob, true, base, 5);

    double t0 = now_ns();
    for (long i = 0; i < pairs; i++) {
        OrderReference* ref = add_order_to_limit_order_book(&lob, true, base, 7);
        cancel_order(&lob.bids, true, ref->price, ref->order_index, lob.tick_size);
    }
    double t1 = now_ns();
    destroy_limit_order_book(&lob);
    return (t1 - t0) / (double)pairs;
}

/* ---------- 2. add only (across a band of levels) ---------- */
double bench_add_only(long n) {
    /* Re-create a fresh book in batches so memory stays bounded; time only the adds. */
    long batch = 200000;
    double total_ns = 0; long done = 0;
    int64_t base = 1000000;
    while (done < n) {
        long this_batch = (n - done < batch) ? (n - done) : batch;
        LimitOrderBook lob = make_book(ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY, 1);
        double t0 = now_ns();
        for (long i = 0; i < this_batch; i++) {
            int64_t p = base + (i % 100) - 50;   /* spread across 100 levels */
            add_order_to_limit_order_book(&lob, true, p, 3);
        }
        double t1 = now_ns();
        total_ns += (t1 - t0);
        destroy_limit_order_book(&lob);
        done += this_batch;
    }
    return total_ns / (double)n;
}

/* ---------- 3. FIFO match, per resting order consumed ---------- */
double bench_fifo_match_per_order(long resting_per_iter, long iters) {
    double total_ns = 0; long total_consumed = 0;
    for (long it = 0; it < iters; it++) {
        LimitOrderBook lob = make_book(ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY, 1);
        int64_t px = 1000000;
        for (long i = 0; i < resting_per_iter; i++)
            add_order_to_limit_order_book(&lob, false, px, 1); /* resting asks, qty 1 each */
        /* Buyer crossing: opposite side = asks, opposite_is_bid = false. */
        double t0 = now_ns();
        match_market_order_using_time_price_priority(&lob.asks, false, lob.tick_size,
                                                     px, (uint64_t)resting_per_iter, false);
        double t1 = now_ns();
        total_ns += (t1 - t0);
        total_consumed += resting_per_iter;
        destroy_limit_order_book(&lob);
    }
    return total_ns / (double)total_consumed;
}

/* ---------- 4. pro-rata, N orders @ one level (per match call) ---------- */
double bench_prorata_one_level(long orders_at_level, long iters) {
    double total_ns = 0;
    for (long it = 0; it < iters; it++) {
        LimitOrderBook lob = make_book(ORDER_MATCHING_ALGORITHM_PRO_RATA, 1);
        int64_t px = 1000000;
        uint64_t total_q = 0;
        for (long i = 0; i < orders_at_level; i++) {
            uint64_t q = 10 + (uint64_t)(i % 90); /* varied sizes */
            add_order_to_limit_order_book(&lob, false, px, q);
            total_q += q;
        }
        uint64_t incoming = total_q / 2; /* consume ~half so pro-rata branch runs */
        double t0 = now_ns();
        match_market_order_pro_rata(&lob.asks, false, lob.tick_size, px, incoming, false, 1);
        double t1 = now_ns();
        total_ns += (t1 - t0);
        destroy_limit_order_book(&lob);
    }
    return total_ns / (double)iters;
}

/* ---------- 5. mixed: 47% add / 48% cancel / 5% match ---------- */
double bench_mixed(long ops) {
    LimitOrderBook lob = make_book(ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY, 1);
    int64_t base = 1000000;
    /* seed liquidity on both sides */
    size_t cap = 200000;
    OrderReference** live = malloc(cap * sizeof(OrderReference*));
    size_t live_n = 0;
    for (int i = 0; i < 2000; i++) {
        int64_t p = base + (i % 50) - 25;
        live[live_n++] = add_order_to_limit_order_book(&lob, true, p - 100, 5);
        live[live_n++] = add_order_to_limit_order_book(&lob, false, p + 100, 5);
    }
    /* Tracked add/cancel orders live far from the touch (base +/- 75..125) so a
       small marketable order never consumes them. Match liquidity is replenished
       at base +/- 1 and is NOT tracked in live[], avoiding any use-after-free. */
    unsigned long rng = 88172645463325252ull;
    double t0 = now_ns();
    for (long i = 0; i < ops; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        unsigned r = (unsigned)(rng % 100);
        if (r < 47 || live_n == 0) {                         /* add (tracked, off-touch) */
            bool bid = (rng >> 8) & 1;
            int64_t p = base + (int64_t)((rng >> 9) % 50) - 25 + (bid ? -100 : 100);
            if (live_n < cap)
                live[live_n++] = add_order_to_limit_order_book(&lob, bid, p, 5);
        } else if (r < 95) {                                 /* cancel a random live order */
            size_t idx = (size_t)((rng >> 16) % live_n);
            OrderReference* ref = live[idx];
            Side* s = ref->is_bid ? &lob.bids : &lob.asks;
            cancel_order(s, ref->is_bid, ref->price, ref->order_index, lob.tick_size);
            live[idx] = live[--live_n];
        } else {                                             /* small marketable order */
            bool bid = (rng >> 20) & 1;
            /* seed near-touch liquidity (untracked), then cross into only that level */
            if (bid) {
                add_order_to_limit_order_book(&lob, false, base + 1, 5); /* resting ask */
                execute_order(&lob, true, base + 50, 3, false, 1);
            } else {
                add_order_to_limit_order_book(&lob, true, base - 1, 5);  /* resting bid */
                execute_order(&lob, false, base - 50, 3, false, 1);
            }
        }
    }
    double t1 = now_ns();
    free(live);
    destroy_limit_order_book(&lob);
    return (t1 - t0) / (double)ops;
}

/* ---------- 6. deep add: many orders to one bucket (per add) ---------- */
double bench_deep_add(long n) {
    LimitOrderBook lob = make_book(ORDER_MATCHING_ALGORITHM_PRICE_TIME_PRIORITY, 1);
    int64_t px = 1000000;
    double t0 = now_ns();
    for (long i = 0; i < n; i++)
        add_order_to_limit_order_book(&lob, true, px, 1);
    double t1 = now_ns();
    destroy_limit_order_book(&lob);
    return (t1 - t0) / (double)n;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    volatile double s = 0; for (int i = 0; i < 1000; i++) s += now_ns();
    const char* which = (argc > 1) ? argv[1] : "all";
    int all = (strcmp(which, "all") == 0);

    if (all || !strcmp(which, "add_only")) {
        double v = bench_add_only(2000000);
        printf("add_only,%.2f\n", v);
    }
    if (all || !strcmp(which, "deep_add")) {
        double v = bench_deep_add(50000);
        printf("deep_add,%.2f\n", v);
    }
    if (all || !strcmp(which, "add_cancel_pair")) {
        double v = bench_add_cancel_warm(20000000);
        printf("add_cancel_pair,%.2f\n", v);
    }
    if (all || !strcmp(which, "fifo_match_per_resting_order")) {
        double v = bench_fifo_match_per_order(2000, 5000);
        printf("fifo_match_per_resting_order,%.2f\n", v);
    }
    if (all || !strcmp(which, "prorata_100_orders_one_level")) {
        double v = bench_prorata_one_level(100, 200000);
        printf("prorata_100_orders_one_level,%.2f\n", v);
    }
    return 0;
}
