# Architecture Review & Stabilization — EO/IR Mission Console

**Date:** 2026-06-23
**Phase:** Architecture stabilization (feature freeze). Review + planning only — **no
feature code was changed** to produce these documents.

This folder is the response to the stabilization brief: stop feature development; review
architecture, performance, latency, and repo structure; plan detector modernization
(RF-DETR via abstraction), profiling/benchmarking, and the EO/IR-Console ↔ AI-Mission-Analyst
repository split.

## Documents
1. **[01-performance-latency-report.md](01-performance-latency-report.md)** — per-stage
   analysis, ranked bottlenecks, and the profiling/dashboard instrumentation plan.
2. **[02-detector-benchmark-report.md](02-detector-benchmark-report.md)** — detector
   abstraction interface, RF-DETR integration, and the benchmark framework + report template.
3. **[03-latency-optimization-roadmap.md](03-latency-optimization-roadmap.md)** — the
   decoupled render-vs-inference architecture and a phased latency plan.
4. **[04-repository-split-plan.md](04-repository-split-plan.md)** — module-by-module split,
   communication contract, shared deps, least-disruptive migration sequence.
5. **[05-implementation-plan-and-ui.md](05-implementation-plan-and-ui.md)** — consolidated,
   prioritized milestones + the ISR operator-console UI plan + decisions needed.

## How the review was produced
Direct reading of the live source (`pipeline/console.py`, `ui/console_view.py`,
`ingestion/*`, `config.yaml`) plus two structured passes over (a) the AI/intelligence
modules for coupling and (b) the CV core for latency/blocking points. Import/coupling
claims and latency locations cite `file:line`.

## The single most important finding
The live feed is **coupled to AI inference**: the UI renders `latest_annotated`, which is
written only after the full per-frame chain (detect→track→ReID→reason→fusion→DB→draw)
completes on one thread. Decoupling display from inference (show the latest raw frame
immediately; let the overlay lag) is the highest-impact fix and the spine of the roadmap.

## Honesty / limits
- **No measured numbers yet** — the system has not been run on the GPU box with live feeds.
  Rankings are from static analysis; Milestone 0 produces real measurements.
- **mAP** needs a labeled EO/IR set; otherwise the benchmark reports relative metrics.
- These are plans. Implementation is the next, separately-approved phase.

## Open decisions
See `05-implementation-plan-and-ui.md` Part 4.
