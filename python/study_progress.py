"""Console progress for long studies, independent of optimizer verbosity."""
from __future__ import annotations

import math
import time


def duration(seconds):
    seconds = int(round(max(0., seconds)))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


class StudyProgress:
    def __init__(self, label, methods, n_runs, lengths, n_resolutions, settings, *, clock=time.perf_counter):
        self.enabled = bool(settings.get("progress", True))
        self.interval = float(settings.get("progress_interval_seconds", 10.))
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("progress_interval_seconds must be positive and finite")
        self.label, self.n_runs, self.clock = label, n_runs, clock
        self.total = len(methods)*n_runs*len(lengths)*n_resolutions
        work = n_runs*sum(lengths)*n_resolutions
        self.remaining = dict.fromkeys(methods, work)
        self.samples = dict.fromkeys(methods, 0)
        self.seconds = dict.fromkeys(methods, 0.)
        self.completed = 0
        self.started = self.last_report = clock()
        self.active_method = None
        self.method_started = self.started
        self.stage(f"Starting {n_runs} repetitions, {self.total} method fits; ETA warms up after the first fit of each method.")

    def stage(self, message, *, force=True):
        if not self.enabled:
            return
        now = self.clock()
        if force or now-self.last_report >= self.interval:
            print(f"[{self.label}] {message} | elapsed {duration(now-self.started)}", flush=True)
            self.last_report = now

    def _eta(self, now):
        if not all(value > 0 for value in self.samples.values()):
            return "warming up"
        remaining = sum(self.remaining[m]*self.seconds[m]/self.samples[m] for m in self.remaining)
        if self.active_method is not None:
            remaining -= min(now-self.method_started,
                             self.active_length*self.seconds[self.active_method]/self.samples[self.active_method])
        return "~"+duration(remaining)

    def __call__(self, event):
        if not self.enabled:
            return
        now = self.clock()
        phase, method = event["phase"], event["method"]
        if phase == "method_start":
            self.active_method, self.active_length = method, event["record_length"]
            self.method_started = now
            detail = f"starting; update budget {event['max_updates']}"
        elif phase == "iteration":
            if now-self.last_report < self.interval:
                return
            detail = f"updates {event['iteration']}/{event['max_updates']}"
        elif phase == "method_end":
            elapsed = now-self.method_started
            self.seconds[method] += elapsed
            self.samples[method] += event["record_length"]
            self.remaining[method] -= event["record_length"]
            self.completed += 1
            self.active_method = None
            detail = f"{event['status']}; {event['iteration']} accepted updates; fit time {duration(elapsed)}"
        else:
            raise ValueError(f"Unknown progress event: {phase}")
        metadata = event["run_metadata"]
        if "initialization" in self.label:
            context = f"initialization {metadata.get('initialization_index', metadata.get('mc_index', 1))}/{self.n_runs}"
        elif "sigma_nu" in metadata:
            context = (f"sigma_nu={metadata['sigma_nu']:g}; N={event['record_length']}; "
                       f"realization {metadata.get('mc_index', 1)}/{self.n_runs}")
        else:
            context = f"{metadata.get('resolution', 'recursive_PL')}; N={event['record_length']}; realization {metadata.get('mc_index', 1)}/{self.n_runs}"
        print(f"[{self.label}] {self.completed}/{self.total} fits completed ({100*self.completed/self.total:.1f}%) | "
              f"{context} | {method}: {detail} | elapsed {duration(now-self.started)} | ETA {self._eta(now)}", flush=True)
        self.last_report = now

    def finish(self, path):
        self.stage(f"Finished {self.completed}/{self.total} method fits; saved {path}")
