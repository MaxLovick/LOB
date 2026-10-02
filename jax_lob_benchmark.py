"""
Benchmark JAX-LOB (KangOxford/AlphaTrade : gymnax_exchange.jaxob) vs lob.c on
time AND memory, and emit TABLE.md, MEMORY.md and BENCHMARKS.md.

Usage:
    python3 bench_jax.py measure   # run JAX measurements -> results.json (slow)
    python3 bench_jax.py report    # build TABLE.md + MEMORY.md from results.json + ./bench_lob
    python3 bench_jax.py           # measure, then report

Every scenario uses the same book state in both engines (see bench_lob.c):
  add_only       64 resting bids on 64 levels, add 1 bid @ 999,970 (existing level)
  add_new_level  same book, add 1 bid @ 999,900 (outside lob.c's range -> side grows)
  deep_add       1024 resting bids on 1024 levels, add 1 bid @ 999,970
  add_cancel     64-level book, add 1 bid @ 999,970 then cancel it (per pair)
  fifo_match     100 asks of qty 1 at ONE price, buy 1 crossing the front one
                 (one incoming order filling one resting order)
  fifo_sweep     lob.c only: buy 100 sweeping the whole level. JAX-LOB cannot do
                 this: match_ask_order/match_bid_order recompute the top-of-book
                 index from the pre-fill array, so the while_loop exits after the
                 first fill. measure() asserts this so the note stays true.

JAX-LOB is measured at batch sizes 1/256/512/1028 two ways:
  normal : no vmap -- B single-book jitted calls in a plain Python loop.
  vmap   : jax.vmap across the B books in one vectorised call.

Memory: JAX-LOB's book is a fixed (capacity x 6) int32 array per side plus a fixed
trades array, allocated up front whatever the book holds. lob.c allocates on demand;
its bytes are read from the book's own capacities after the scenario runs.

JAX compat: gymnax_exchange/jaxob/JaxOrderBookArrays.py used
jnp.where(cond, x=..., y=...) in __removeZeroNegQuant; JAX >= 0.10 made x/y
positional-only, so that one call must use positional args.
"""
import os, sys, time, math, json, subprocess
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["OMP_NUM_THREADS"] = "1"

BATCHES = [1, 256, 512, 1028]
SMALL_NORDERS = 128      # JAX side capacity for the 64-order scenarios
DEEP_NORDERS = 2048      # JAX side capacity for the 1024-order scenario
FIFO_K = 100
OID_ADD = 555
IN_RANGE_PX = 999_970
NEW_LEVEL_PX = 999_900
RESULTS = "results.json"
JAX_ROW_BYTES = 6 * 4    # 6 int32 fields per order / trade row

def jax_state_bytes(n_orders, n_trades):
    """Bytes of one JAX-LOB book: asks + bids (n_orders rows each) + trades."""
    return (2 * n_orders + n_trades) * JAX_ROW_BYTES

def measure():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "jax-lob"))
    import numpy as np
    import jax, jax.numpy as jnp
    from gymnax_exchange.jaxob import JaxOrderBookArrays as job
    print("jax", jax.__version__, jax.devices(), file=sys.stderr)

    stepf = jax.jit(job.vcond_type_side)
    def step(state, msgs): return stepf(state, msgs)[0]

    def build_state(B, nOrders, nTrades, prefill):
        asks = np.full((B, nOrders, 6), -1, dtype=np.int32)
        bids = np.full((B, nOrders, 6), -1, dtype=np.int32)
        trades = np.full((B, nTrades, 6), -1, dtype=np.int32)
        tgt = bids if prefill["side"] == "bid" else asks
        for r in range(prefill["n"]):
            if prefill.get("one_level"):
                px = prefill["price0"]
            else:
                px = prefill["price0"] - (r if prefill["side"] == "bid" else -r)
            tgt[:, r, 0] = px
            # all 6 fields must be set: JAX-LOB finds a free row with (orderside == -1)
            # over every cell, so a -1 left in any column makes that row look free
            # and the next add overwrites it.
            tgt[:, r, 1] = prefill["qty"]; tgt[:, r, 2] = r; tgt[:, r, 3] = 1
            tgt[:, r, 4] = r; tgt[:, r, 5] = 0
        return (jnp.asarray(asks), jnp.asarray(bids), jnp.asarray(trades))

    def mb(B, row): return jnp.asarray(np.tile(np.array(row, np.int32), (B, 1)))
    small = lambda B: build_state(B, SMALL_NORDERS, 2, {"side":"bid","n":64,"price0":1_000_000,"qty":10})

    def sc_add_only(B):
        add = mb(B, [1,1,10,IN_RANGE_PX,0,OID_ADD,1000,0])
        return small(B), (lambda s: step(s, add)), 1, (SMALL_NORDERS, 2)
    def sc_add_new_level(B):
        add = mb(B, [1,1,10,NEW_LEVEL_PX,0,OID_ADD,1000,0])
        return small(B), (lambda s: step(s, add)), 1, (SMALL_NORDERS, 2)
    def sc_deep_add(B):
        st = build_state(B, DEEP_NORDERS, 2, {"side":"bid","n":DEEP_NORDERS//2,"price0":1_000_000,"qty":10})
        add = mb(B, [1,1,10,IN_RANGE_PX,0,OID_ADD,1000,0])
        return st, (lambda s: step(s, add)), 1, (DEEP_NORDERS, 2)
    def sc_add_cancel(B):
        add = mb(B, [1,1,10,IN_RANGE_PX,0,OID_ADD,1000,0])
        cxl = mb(B, [2,1,10,IN_RANGE_PX,0,OID_ADD,1001,0])
        return small(B), (lambda s: step(step(s, add), cxl)), 1, (SMALL_NORDERS, 2)
    def sc_fifo(B):
        st = build_state(B, SMALL_NORDERS, SMALL_NORDERS,
                         {"side":"ask","n":FIFO_K,"price0":1_000_000,"qty":1,"one_level":True})
        cross = mb(B, [1,1,1,1_000_000,0,OID_ADD,1000,0])
        return st, (lambda s: step(s, cross)), 1, (SMALL_NORDERS, SMALL_NORDERS)
    scenarios = [("add_only",sc_add_only),("add_new_level",sc_add_new_level),
                 ("deep_add",sc_deep_add),("add_cancel",sc_add_cancel),("fifo_match",sc_fifo)]

    # sanity: the single cross fills exactly one resting ask ...
    st, run_once, _, _ = sc_fifo(1)
    asks, bids, trades = run_once(st)
    assert int((np.asarray(trades)[0, :, 0] != -1).sum()) == 1
    assert int((np.asarray(asks)[0, :, 1] > 0).sum()) == FIFO_K - 1
    # ... and a buy for FIFO_K still only fills one (JAX-LOB matcher limitation)
    sweep = mb(1, [1,1,FIFO_K,1_000_000,0,OID_ADD,1000,0])
    asks, bids, trades = step(st, sweep)
    sweep_fills = int((np.asarray(trades)[0, :, 0] != -1).sum())
    print("JAX-LOB fills for a buy of", FIFO_K, "into", FIFO_K, "1-lot asks:", sweep_fills, file=sys.stderr)

    # overflow behaviour: fill a capacity-8 bid side, then add a 9th order
    st = build_state(1, 8, 2, {"side":"bid","n":8,"price0":1_000_000,"qty":10})
    before = np.asarray(st[1])[0].copy()
    after = np.asarray(step(st, mb(1, [1,1,7,999_000,0,OID_ADD,1000,0]))[1])[0]
    overwritten = [i for i in range(8) if (before[i] != after[i]).any()]
    overflow = {"capacity": 8, "overwritten_rows": overwritten,
                "rows_changed": int((before != after).any(axis=1).sum()),
                "last_row_before": before[-1].tolist(), "last_row_after": after[-1].tolist()}
    print("overflow:", overflow, file=sys.stderr)

    def timed_iters(fn, per_call_target_s=4.0, max_iters=100):
        fn()                                   # warm / compile
        t0 = time.perf_counter(); fn(); one = time.perf_counter() - t0
        return max(3, min(max_iters, math.ceil(per_call_target_s / max(one, 1e-9))))
    def vmap_ns(run_once, state0, events):
        def once():
            fs = run_once(state0); jax.block_until_ready(fs)
        iters = timed_iters(once)
        t0 = time.perf_counter()
        for _ in range(iters): once()
        return (time.perf_counter()-t0)/(iters*events)*1e9
    def seq_ns(make, ev, B, target=2000):
        state0, run_once, _, _ = make(1)
        fs = run_once(state0); jax.block_until_ready(fs)
        reps = max(1, math.ceil(target/B))
        t0 = time.perf_counter()
        for _ in range(reps):
            for _b in range(B): fs = run_once(state0)
            jax.block_until_ready(fs)
        return (time.perf_counter()-t0)/(reps*B*ev)*1e9

    res = {"jax_version": jax.__version__, "overflow": overflow, "sweep_fills": sweep_fills, "memory": {}}
    for name, make in scenarios:
        res[name] = {"normal":{}, "vmap":{}}
        _, _, ev, (n_orders, n_trades) = make(1)
        res["memory"][name] = jax_state_bytes(n_orders, n_trades)
        for B in BATCHES:
            st, run_once, events, _ = make(B)
            res[name]["vmap"][str(B)] = vmap_ns(run_once, st, events*B)
            del st
            res[name]["normal"][str(B)] = seq_ns(make, ev, B)
            print(f"{name:13s} B={B:4d} normal={res[name]['normal'][str(B)]:11.0f} "
                  f"vmap={res[name]['vmap'][str(B)]:11.0f}", file=sys.stderr)
        json.dump(res, open(RESULTS,"w"), indent=1)   # checkpoint after each scenario
    return res

def c_runs(scn, n=7):
    rows = []
    for _ in range(n):
        out = subprocess.run(["./bench_lob", scn], capture_output=True, text=True).stdout.strip()
        rows.append([float(x) for x in out.split(",")[1:]])
    return rows

def c_scenario(scn):
    rows = c_runs(scn)
    ns = sorted(r[0] for r in rows)[len(rows)//2]
    return {"ns": ns, "bytes": int(rows[0][1]), "leaked": int(rows[0][2])}

def c_lines(scn, n=5):
    """median of each numeric column across n runs, keyed by first numeric column"""
    runs = []
    for _ in range(n):
        out = subprocess.run(["./bench_lob", scn], capture_output=True, text=True).stdout.strip()
        runs.append([[float(x) for x in l.split(",")[1:]] for l in out.splitlines()])
    med = []
    for i in range(len(runs[0])):
        cols = list(zip(*[r[i] for r in runs]))
        med.append([sorted(c)[len(c)//2] for c in cols])
    return med

def report():
    res = json.load(open(RESULTS))
    C = {k: c_scenario(k) for k in ["add_only","add_new_level","deep_add","add_cancel","fifo_match","fifo_sweep","prorata"]}
    scaling = c_lines("scaling")
    memcurve = c_lines("memory", n=1)

    rows = [("add (existing level)","add_only"),
            ("add (new level, book grows)","add_new_level"),
            ("deep add (1,024-level book)","deep_add"),
            ("add + cancel (per pair)","add_cancel"),
            ("FIFO match (1 incoming fills 1 resting)","fifo_match"),
            ("FIFO sweep of 100 orders (per order)","fifo_sweep"),
            ("pro-rata (per match call)","prorata")]
    f = lambda n: f"{n:,.0f}"
    r = lambda j, c: f"{j/c:,.1f}x" if j/c < 10 else f"{j/c:,.0f}x"
    hdr = ["Scenario","lob.c (ns)"]
    for B in BATCHES: hdr += [f"JAX B={B}","speed-up"]
    for B in BATCHES: hdr += [f"JAX vmap B={B}","speed-up"]
    lines = ["| "+" | ".join(hdr)+" |", "|"+"|".join(["---"]*len(hdr))+"|"]
    for label, key in rows:
        c = C[key]["ns"]; cells = [label, f(c)]
        if key == "prorata":
            cells += ["n/a (not in JAX-LOB)","-"] + ["n/a","-"]*7
        elif key == "fifo_sweep":
            cells += ["n/a (stops after 1 fill)","-"] + ["n/a","-"]*7
        else:
            for mode in ("normal","vmap"):
                for B in BATCHES:
                    j = res[key][mode][str(B)]; cells += [f(j), r(j,c)]
        lines.append("| "+" | ".join(cells)+" |")
    table = "\n".join(lines)
    open("TABLE.md","w").write(table+"\n")

    kb = lambda b: f"{b/1024:,.1f} KiB"
    mrows = ["| Scenario | Resting orders | lob.c per book | JAX-LOB per book (fixed) | JAX-LOB capacity per side | JAX-LOB at B=1028 |",
             "|---|---|---|---|---|---|"]
    resting = {"add_only":66,"add_new_level":66,"deep_add":1026,"add_cancel":65,"fifo_match":99,"fifo_sweep":0,"prorata":"~50"}
    caps = {"add_only":SMALL_NORDERS,"add_new_level":SMALL_NORDERS,"deep_add":DEEP_NORDERS,
            "add_cancel":SMALL_NORDERS,"fifo_match":SMALL_NORDERS}
    for label, key in rows:
        jb = res["memory"].get(key)
        mrows.append("| " + " | ".join([label, str(resting[key]), kb(C[key]["bytes"]),
            kb(jb) if jb else "n/a", f"{caps[key]:,}" if jb else "n/a",
            f"{jb*1028/1024/1024:,.1f} MiB" if jb else "n/a"]) + " |")
    mem_table = "\n".join(mrows)

    crows = ["| Resting orders (1 per level) | lob.c | JAX-LOB, capacity 128 | JAX-LOB, capacity 2,048 |",
             "|---|---|---|---|"]
    for n, b in memcurve:
        n = int(n)
        j128 = "does not fit (overwrites)" if n > 128 else kb(jax_state_bytes(128, 2))
        crows.append(f"| {n:,} | {kb(b)} | {j128} | {kb(jax_state_bytes(2048, 2))} |")
    curve_table = "\n".join(crows)

    srows = ["| Resting orders (one account) | add, ns/op (avg) | FIFO sweep, ns/order | lob.c memory |",
             "|---|---|---|---|"]
    for n, add_ns, match_ns, b in scaling:
        srows.append(f"| {int(n):,} | {f(add_ns)} | {f(match_ns)} | {kb(b)} |")
    scaling_table = "\n".join(srows)
    open("MEMORY.md","w").write(mem_table+"\n\n"+curve_table+"\n\n"+scaling_table+"\n")

    json.dump({"c": C, "scaling": scaling, "memcurve": memcurve}, open("c_results.json","w"), indent=1)
    print(table); print(); print(mem_table); print(); print(curve_table); print(); print(scaling_table)
    print("\noverflow:", res["overflow"])
    return table, mem_table, curve_table, scaling_table, C, res

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("measure","all"): measure()
    if mode in ("report","all"): report()
