# bci-fault-bench

Software-in-the-Loop testbed for **fault injection in LSL-based BCI pipelines**.

It replays public EEG (BCI Competition IV 2a via MOABB) as a real-time [Lab Streaming Layer](https://labstreaminglayer.org) stream, injects controlled faults at the transport interface (sample loss, jitter, delay, disconnection), decodes with a frozen CSP+LDA model, and logs per-second telemetry plus per-trial decisions. An orchestrator runs full campaigns (subjects × runs × conditions) in parallel with fixed seeds and resumable execution; an analysis script produces the descriptive and inferential results (Friedman, Wilcoxon + Holm, effect sizes, rolling-window degradation, silent-failure divergence, threshold vs. model detectors with leave-one-subject-out validation).

Built for the undergraduate thesis *Interfaces cerebro-computadora bajo condiciones adversas: del software a la decodificación* (Agustín Tamagusuku, Software Engineering, Universidad Siglo 21, 2026).

## Components (`src/bcibench/`)

| Module | Role |
|---|---|
| `data.py` | MOABB loader (BNCI2014_001), events from the `STI` channel, artifact-flagged trials excluded from scoring |
| `decoder.py` | Reference decoder: CSP (6 components) + LDA, trained once per subject on session 1 and frozen |
| `replay.py` | Replay service: emits one run as an LSL stream at its native rate (blocks of 10 samples / 40 ms) plus a separate, unperturbed marker stream |
| `injector.py` | Fault models: `loss` (random / burst), `jitter`, `delay`, `disconnect` (the outlet is destroyed and recreated); seeded and repeatable |
| `consumer.py` | Pipeline under test + telemetry collector → `trials.csv`, `telemetry.csv` |
| `runner.py` | Campaign executor: block ordering (run-major), parallel workers, `done.json` resume |
| `metrics.py`, `stats.py` | Dependent variables, rolling balanced accuracy, reference thresholds, divergence, tests, detectors |
| `scripts/analyze.py` | End-to-end analysis → CSV/Markdown tables and APA-style figures |

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# train the 9 frozen decoders (downloads the dataset on first use)
.venv/bin/python scripts/poc1_offline.py 1 2 3 4 5 6 7 8 9
# 60-second smoke test with 5 % random sample loss
.venv/bin/python scripts/smoke_run.py --fault loss --severity 0.05 --max-seconds 60
```

Full campaign (9 subjects × evaluation runs 1-2 × 13 conditions = 234 executions, ~4.2 h with 6 workers on an 8-vCPU Linux VM):

```bash
PYTHONPATH=src .venv/bin/python -m bcibench.runner --plan campana \
  --subjects 1 2 3 4 5 6 7 8 9 --runs 0 1 --workers 6 --out results/campana
.venv/bin/python scripts/analyze.py --root results/campana --out results/analysis --window 8 --thr-rule min
```

Interrupted campaigns resume with the same command (executions with `done.json` are skipped).

## Fault models

| Model | Where it acts | Severities (exploratory) | Real-world counterpart |
|---|---|---|---|
| Sample loss | stream content (random or 100 ms bursts) | 1 %, 5 %, 10 % | wireless headset dropping samples; consumer buffer overflow |
| Jitter | block delivery time (Gaussian) | σ = 10, 50, 100 ms | network congestion, OS scheduling |
| Delay | block delivery time (constant) | 50, 100, 250 ms | sustained congestion, oversized buffers |
| Disconnection | real transport: outlet closed and recreated (5 events/run) | 0.5, 1, 3 s | link cut, producer crash; exercises LSL's declared recovery |

## Timing notes

Run the campaign on Linux: Windows' default timer granularity (~15 ms) is coarser than the smallest jitter severity (10 ms). On Ubuntu 24.04 (EC2 c6i.2xlarge) the replay's delivery-time error was p99 < 1 µs. Six parallel executions changed reference latency/inter-arrival dispersion by < 0.1 ms.

## Outputs per execution

`producer.json` (condition, seed, samples pushed/dropped, outages, timing error), `fault_log.jsonl`, `trials.csv` (label, prediction, confidence, samples, validity, decision delay), `telemetry.csv` (per nominal second: expected/received samples, gaps, latency, inter-arrival, predictions, exceptions), `consumer.json`.

## Resumen en español

Banco de experimentación *Software-in-the-Loop*: reproduce EEG público como flujo LSL en tiempo real, inyecta fallos controlados en la interfaz de transporte, decodifica con CSP+LDA congelado y registra telemetría por segundo y decisiones por ensayo. Incluye el ejecutor de campañas (semillas fijas, bloques, reanudación) y el análisis completo. Instrumento del Trabajo Final de Graduación citado arriba. Licencia MIT.
