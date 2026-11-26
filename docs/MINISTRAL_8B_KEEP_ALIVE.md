# Ministral 8B keep-alive on the local inference host

**Keep the production default at 30 minutes.** Extending it to 60 minutes
avoided one of two 8B reloads in a workload designed to cross the current
timeout. Only three of 1,527 recent gateway gaps fell in the interval where
the extra half hour could prevent a reload.

## Results

| Measured on the shared host | 30 minutes | 60 minutes |
|---|---:|---:|
| Completed 8B requests scored | 30 | 30 |
| Additional incomplete 8B streams | 1 | 0 |
| Median first-token time | 0.162 s | 0.162 s |
| p95 first-token time | 2.316 s | 0.540 s |
| Within-burst p95 (27 requests) | 0.210 s | 0.193 s |
| First token after idle period 1 | 3.213 s | 0.540 s |
| First token after idle period 2 | 2.316 s | 2.383 s |
| 8B reloads after the initial load | 2 | 1 |
| Peak daemon-tree physical footprint | 19.35 GiB | 20.61 GiB |
| Peak daemon-tree RSS | 19.21 GiB | 23.74 GiB |
| Peak allocation reported by Ollama | 18.11 GiB | 31.14 GiB |

The 30-minute daemon had released 8B at the end of both idle periods. The
60-minute daemon still held it both times. On the first return, the 60-minute
daemon could load 14B alongside 8B. On the second, Ollama logged 17.2 GiB
of available system memory, evicted 8B to load 14B, then reloaded 8B when
the gateway requested it. Each daemon logged two 14B weight loads and one
initial 8B load in addition to the 8B reloads in the table.

The longer timer cut completed-stream p95 by 1.78 seconds in this workload.
There were only two opportunities to avoid expiry. With 30 observations the
single slowest request falls above the p95 rank, which is why the 60-minute
p95 is 0.540 seconds even though the second return took 2.383 seconds.

**Peak memory depends on which other models fit.** Across the two timed
idle periods, the 60-minute arm used 0.51 GiB·h more for 8B and 7.62 GiB·h
more for 14B. The latter arose because the baseline evicted 14B during the
first return. Each arm shared physical memory with the other daemon and the
rest of the host. Ollama's reported VRAM equalled its total allocation in
both arms.

After shutting down both benchmark daemons, we checked the return sequence
again with one fresh daemon at a time. We primed 8B in both cases, explicitly
unloaded it for the baseline, then sent the same native 14B and 8B requests.
Memory sampling covered only those return requests, at 250 ms intervals.
Both models fit simultaneously in both runs.

| Peak with one daemon at a time | 8B starts cold | 8B starts warm |
|---|---:|---:|
| macOS physical footprint | 20.55 GiB | 20.69 GiB |
| RSS | 33.16 GiB | 33.15 GiB |
| Ollama-reported allocation | 31.78 GiB | 31.40 GiB |

**Coverage.** One warm baseline stream timed out after 180 seconds while
reading the response body. Its client TTFT was unavailable. We preserved
the failed attempt and completed a fresh matched prompt on both arms before
starting the second idle period. The reported baseline p95 uses completed
streams. Including that attempt at any possible TTFT gives a baseline p95
between 2.316 and 3.213 seconds. Recovery used fresh HTTP connections for both
arms. Overlapping samplers preserved the memory trace across the interruption.
Each arm had 3,978 samples, with a longest interval of 1.07 seconds across
both arms. One baseline sample lacked a macOS physical-footprint reading.
All RSS and Ollama allocation samples were available. Peaks are sampled maxima.

## Workload and measurements

On 26 November 2026, two dedicated Ollama 0.32.14 daemons and two loopback
inference gateways ran on the same Apple M5 Max with 128 GiB of unified memory.
Both used the installed `ministral-3:8b` Q4_K_M manifest
(`1922accd5827ebe6829e536369195db25eaf664528dc66206d646ea3bb386b71`).
The daemons used 32,768-token context, two parallel slots and a three-model
limit. The baseline inherited the production keep-alive of 30 minutes. The
candidate used 60 minutes for 8B. Each 14B request explicitly used 30 minutes
in both arms.

Each arm started empty. One cold 8B request established the initial weight
load. Three bursts of ten matched 8B requests followed, with 31 minutes of
wall-clock idle time between bursts. After each gap, both arms also handled
one `ministral-3:14b` request before returning to 8B. Requests used short
prompts, temperature zero, seed 21 and a 24-token cap. The driver alternated
which arm received each prompt first and ran one generation at a time.

- **Time to first token (TTFT)** ran from sending the gateway's streaming
  `/v1/chat/completions` request until the first non-whitespace content delta.
  The nearest-rank p95 is the 29th of 30 sorted 8B observations per arm.
  The initial cold load and 14B requests are excluded.
- **Reloads** count daemon log events that loaded the 8B weight blob after the
  initial load. The driver also captured Ollama's process list immediately
  before and after each request.
- **Peak memory** covers each Ollama daemon and its runner processes, sampled
  once per second. macOS physical footprint, RSS and Ollama's reported
  resident bytes measure different aspects of the same workload. VRAM is
  part of the reported total on this unified-memory host. The gateways and
  other host processes are outside the daemon-tree measurement.

To reproduce the workload, start an empty dedicated Ollama daemon at each
keep-alive setting and point a separate gateway at each daemon. The primary
requests use the gateway's normal Ollama adapter and inherit the daemon
default. Set the shell variables below to the URLs, process IDs and log paths
for those benchmark instances and choose an empty output directory.
[Ollama keep-alive settings](https://docs.ollama.com/faq#how-can-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately)
and [running-model fields](https://docs.ollama.com/api/ps).

```sh
python scripts/benchmark_ollama_keep_alive.py \
  --baseline-url "$BASELINE_OLLAMA" --candidate-url "$CANDIDATE_OLLAMA" \
  --baseline-gateway-url "$BASELINE_GATEWAY" --candidate-gateway-url "$CANDIDATE_GATEWAY" \
  --baseline-pid "$BASELINE_PID" --candidate-pid "$CANDIDATE_PID" \
  --baseline-log "$BASELINE_LOG" --candidate-log "$CANDIDATE_LOG" \
  --baseline-keep-alive 30m --candidate-keep-alive 60m --cycles 2 \
  --idle-seconds 1860 --requests-per-burst 10 --secondary-model ministral-3:14b \
  --context 32768 --predict 24 --output-dir "$BENCHMARK_OUTPUT"
```

## How often the longer timer could help

The local gateway recorded 1,528 completed Ministral 8B executions between
17 November at 22:02 and 26 November at 05:09 (Asia/Shanghai). We measured
each gap from the latest preceding completion to the next start, treating
overlaps as zero. Of 1,527 gaps, only three were longer than 30 minutes and at
most 60 minutes (30.9, 57.2 and 58.0 minutes). Another 24 exceeded an hour.

If TTL expiry were the only cause of eviction, a 60-minute timer would have
avoided three reloads and added 12 hours 56 minutes of idle 8B residency
across that trace. The estimate excludes time before the first and after the
last request. Other callers, memory pressure and daemon restarts can change
actual residency.
