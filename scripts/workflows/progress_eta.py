"""Measured progress/ETA display; never estimate a benchmark accuracy."""
import statistics
import time
from collections import deque
from datetime import datetime, timedelta


def duration(seconds):
    if seconds is None:
        return '测速中'
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f'{days}天 ' if days else '') + f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class ProgressETA:
    def __init__(self, totals, mean_candidates, initial=None, emit=print, clock=time.monotonic):
        self.totals = totals
        self.work = {name: 32 + mean_candidates.get(name, 0) for name in totals}
        self.done = {name: min((initial or {}).get(name, 0), n) for name, n in totals.items()}
        self.emit, self.clock = emit, clock
        self.started = clock()
        self.samples = {}
        self.last = {}
        self.last_display = 0

    def update(self, name, status, force=False):
        if name not in self.totals:
            return
        now = self.clock()
        n = status.get('N_complete', self.done[name])
        if n < self.done[name] or n > self.totals[name]:
            raise ValueError('Progress count outside ordered full split')
        if name in self.last:
            previous_n, previous_time = self.last[name]
            if n > previous_n and now > previous_time:
                self.samples.setdefault(name, deque(maxlen=20)).append((n-previous_n, now-previous_time))
        self.last[name] = (n, now)
        self.done[name] = n
        if not force and now-self.last_display < 2 and status.get('status') != 'complete':
            return
        self.last_display = now
        samples = self.samples.get(name, [])
        sec_per_row = sum(s for _, s in samples)/sum(k for k, _ in samples) if samples else None
        remaining = self.totals[name]-n
        eta = remaining*sec_per_row if sec_per_row else None
        completed = sum(self.done.values())
        total = sum(self.totals.values())
        coefficients = [sum(s for _, s in values)/sum(k for k, _ in values)/self.work[key]
                        for key, values in self.samples.items() if values]
        # Candidate count + fixed image overhead is only a planning proxy.
        # Clearly label the cross-task estimate as coarse until each task is measured.
        coarse_eta = (statistics.median(coefficients)*sum((self.totals[key]-self.done[key])*self.work[key]
                      for key in self.totals)) if coefficients else None
        speed = f'{1/sec_per_row:.2f}题/秒' if sec_per_row else '测速中'
        finish = (datetime.now().astimezone()+timedelta(seconds=eta)).strftime('%m-%d %H:%M %Z') if eta is not None else '测速中'
        line = (f'[{datetime.now().astimezone():%H:%M:%S}] {name}: {n:,}/{self.totals[name]:,} '
                f'({100*n/self.totals[name]:.1f}%) | {speed} | 本项ETA {duration(eta)} '
                f'| 预计本项结束 {finish} | 全局 {completed:,}/{total:,} '
                f'| 整体粗估ETA {duration(coarse_eta)} | 已运行 {duration(now-self.started)}')
        self.emit(line)
        return dict(N_complete=n, N_total=self.totals[name], seconds_per_row=sec_per_row,
                    task_eta_seconds=eta, coarse_total_eta_seconds=coarse_eta,
                    eta_scope='actual recent rate for current task; candidate-weighted coarse estimate for unmeasured tasks')
