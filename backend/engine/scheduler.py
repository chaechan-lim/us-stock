"""Market-hours-aware scheduler for the trading engine.

Manages all periodic tasks:
- T1: Daily full scan (pre-market)
- T2: Intraday hot scan (during regular session, every 30min)
- T3: Sector analysis (during regular session, every 1h)
- T4: AI daily briefing (post-market)
- Strategy evaluation loop (during regular session, every 5min)
- SL/TP position check (during regular session, every 1min)
- Market state update (during regular session, every 15min)
- Health check (always, every 2min)

US market hours in KST:
  Pre-market:    18:00 ~ 23:30 (DST: 17:00 ~ 22:30)
  Regular:       23:30 ~ 06:00 (DST: 22:30 ~ 05:00)
  After-hours:   06:00 ~ 10:00 (DST: 05:00 ~ 09:00)
  Market closed: 10:00 ~ 18:00 (DST: 09:00 ~ 17:00)
"""

import asyncio
import logging
from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

from services.holiday_calendar import is_kr_holiday, is_us_holiday

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


class MarketPhase(str, Enum):
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


def get_market_phase(now: datetime | None = None) -> MarketPhase:
    """Get current US market phase based on ET time."""
    if now is None:
        now = datetime.now(ET)
    else:
        now = now.astimezone(ET)

    t = now.time()
    weekday = now.weekday()

    # Weekend
    if weekday >= 5:
        return MarketPhase.CLOSED

    # NYSE holiday — early-close days are not modeled (still REGULAR for the
    # morning, time logic below handles them well enough).
    if is_us_holiday(now.date()):
        return MarketPhase.CLOSED

    if time(4, 0) <= t < time(9, 30):
        return MarketPhase.PRE_MARKET
    elif time(9, 30) <= t < time(16, 0):
        return MarketPhase.REGULAR
    elif time(16, 0) <= t < time(20, 0):
        return MarketPhase.AFTER_HOURS
    else:
        return MarketPhase.CLOSED


def get_kr_market_phase(now: datetime | None = None) -> MarketPhase:
    """Get current KR market phase based on KST time.

    KR market hours (KST):
      Pre-market:    08:00 ~ 09:00
      Regular:       09:00 ~ 15:30
      After-hours:   15:30 ~ 18:00
      Closed:        18:00 ~ 08:00
    """
    if now is None:
        now = datetime.now(KST)
    else:
        now = now.astimezone(KST)

    t = now.time()
    weekday = now.weekday()

    if weekday >= 5:
        return MarketPhase.CLOSED

    # KR public holiday (근로자의 날 / 추석 / 신정 etc.) — KIS-sourced cache
    # populated at startup, hardcoded fallback for the known year.
    if is_kr_holiday(now.date()):
        return MarketPhase.CLOSED

    if time(8, 0) <= t < time(9, 0):
        return MarketPhase.PRE_MARKET
    elif time(9, 0) <= t < time(15, 30):
        return MarketPhase.REGULAR
    elif time(15, 30) <= t < time(18, 0):
        return MarketPhase.AFTER_HOURS
    else:
        return MarketPhase.CLOSED


def is_market_open(now: datetime | None = None) -> bool:
    return get_market_phase(now) == MarketPhase.REGULAR


def is_opening_minutes(
    market: str = "US",
    minutes: int = 30,
    now: datetime | None = None,
) -> bool:
    """True during the first N minutes after REGULAR open for `market`.

    Used to suppress BUY evaluation during the opening auction / high-vol
    first minutes — historically ~60% of live fills land in the first 30
    minutes and lose ~5% within 4h (ALM / AMPX whipsaw pattern,
    2026-04-24 post-mortem).
    """
    if market == "KR":
        now = now or datetime.now(KST)
        now_tz = now.astimezone(KST)
        phase = get_kr_market_phase(now_tz)
        open_t = time(9, 0)
    else:
        now = now or datetime.now(ET)
        now_tz = now.astimezone(ET)
        phase = get_market_phase(now_tz)
        open_t = time(9, 30)

    if phase != MarketPhase.REGULAR:
        return False
    t = now_tz.time()
    delta_min = (t.hour - open_t.hour) * 60 + (t.minute - open_t.minute)
    return 0 <= delta_min < minutes


class TaskEntry:
    """A scheduled task with phase and interval."""

    def __init__(
        self,
        name: str,
        fn,
        interval_sec: int,
        phases: list[MarketPhase] | None = None,
        recovery=None,
        market: str = "US",
    ):
        self.name = name
        self.fn = fn
        self.interval_sec = interval_sec
        self.phases = phases  # None = always run
        self.last_run: datetime | None = None
        self.recovery = recovery  # Optional TaskRecovery wrapper
        self.market = market  # "US" or "KR"

    def should_run(self, now: datetime) -> bool:
        if self.last_run and (now - self.last_run).total_seconds() < self.interval_sec:
            return False
        if self.phases is not None:
            if self.market == "KR":
                phase = get_kr_market_phase(now)
            else:
                phase = get_market_phase(now)
            if phase not in self.phases:
                return False
        return True


class TradingScheduler:
    """Orchestrates all periodic tasks with market-hours awareness."""

    def __init__(self, recovery_manager=None):
        self._tasks: list[TaskEntry] = []
        self._running = False
        self._tick_interval = 10  # check every 10 seconds
        self._recovery = recovery_manager

    def add_task(
        self,
        name: str,
        fn,
        interval_sec: int,
        phases: list[MarketPhase] | None = None,
        max_retries: int = 2,
        failure_threshold: int = 5,
        market: str = "US",
    ) -> None:
        """Register a periodic task.

        Safe to call while the scheduler is running — the :meth:`start` loop
        snapshots the task list on each tick so newly added tasks are picked up
        on the next tick without disrupting the current iteration.
        """
        recovery = None
        if self._recovery:
            recovery = self._recovery.wrap_task(
                name=name, fn=fn,
                max_retries=max_retries,
                failure_threshold=failure_threshold,
            )

        self._tasks.append(TaskEntry(name, fn, interval_sec, phases, recovery, market))
        phase_str = [p.value for p in phases] if phases else "always"
        market_str = f" [{market}]" if market != "US" else ""
        logger.info(
            "Scheduled task: %s (every %ds, phases=%s%s)",
            name, interval_sec, phase_str, market_str,
        )

    def remove_task(self, name: str) -> bool:
        """Remove a registered task by name.

        Returns *True* if the task was found and removed, *False* if no task
        with that name exists.

        Safe to call while the scheduler is running — the :meth:`start` loop
        snapshots the task list on each tick, so an in-progress tick is not
        affected by the removal; the task simply disappears from the next tick.
        """
        original_count = len(self._tasks)
        self._tasks = [t for t in self._tasks if t.name != name]
        removed = len(self._tasks) < original_count
        if removed:
            logger.info("Task removed: %s", name)
        else:
            logger.warning("remove_task: task %r not found", name)
        return removed

    def remove_tasks_by_prefix(self, prefix: str) -> int:
        """Remove all tasks whose name starts with *prefix*.

        Useful for bulk-removing the complete task set of a specific account
        when the account is deregistered.  For example, passing
        ``"ACC001:US:"`` removes all US-market tasks for account ACC001.

        Returns the number of tasks removed.

        Raises:
            ValueError: If *prefix* is empty — an empty prefix matches every
                task name and would silently wipe the entire task list, which
                is almost certainly a caller bug.

        Safe to call while the scheduler is running (same snapshot guarantee as
        :meth:`remove_task`).
        """
        if not prefix:
            raise ValueError(
                "remove_tasks_by_prefix: prefix must not be empty — "
                "an empty prefix matches all task names and would remove every task"
            )
        original_count = len(self._tasks)
        self._tasks = [t for t in self._tasks if not t.name.startswith(prefix)]
        removed = original_count - len(self._tasks)
        if removed:
            logger.info("Removed %d tasks with prefix %r", removed, prefix)
        else:
            logger.warning("remove_tasks_by_prefix: no tasks matched prefix %r", prefix)
        return removed

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info("Scheduler started with %d tasks", len(self._tasks))

        while self._running:
            now = datetime.now(ET)
            phase = get_market_phase(now)

            # Snapshot the task list at the start of each tick so that
            # concurrent calls to add_task() / remove_task() from other
            # coroutines (which can run at any ``await`` point) do not
            # mutate the collection while we iterate it.
            for task in list(self._tasks):
                if task.should_run(now):
                    if task.recovery:
                        success = await task.recovery.execute()
                        if success:
                            task.last_run = now
                    else:
                        try:
                            logger.debug("Running task: %s (phase=%s)", task.name, phase.value)
                            await task.fn()
                            task.last_run = now
                        except Exception as e:
                            logger.error("Task %s failed: %s", task.name, e)

            await asyncio.sleep(self._tick_interval)

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopped")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def task_count(self) -> int:
        """Current number of registered tasks."""
        return len(self._tasks)

    @property
    def task_names(self) -> list[str]:
        return [t.name for t in self._tasks]

    def get_status(self) -> dict:
        """Get scheduler status with task info."""
        now = datetime.now(ET)
        us_phase = get_market_phase(now)
        kr_phase = get_kr_market_phase(now)
        kr_now = now.astimezone(KST)

        tasks = []
        for t in self._tasks:
            # For display: show whether the task is in the right phase
            # (even if interval hasn't elapsed yet)
            if t.phases is not None:
                current_phase = kr_phase if t.market == "KR" else us_phase
                in_phase = current_phase in t.phases
            else:
                in_phase = True
            tasks.append({
                "name": t.name,
                "interval_sec": t.interval_sec,
                "phases": [p.value for p in t.phases] if t.phases else None,
                "last_run": t.last_run.isoformat() if t.last_run else None,
                "active": in_phase,
                "market": t.market,
                "circuit": t.recovery.circuit.get_status() if t.recovery else None,
            })

        return {
            "running": self._running,
            "market_phase": us_phase.value,
            "market_time_et": now.strftime("%Y-%m-%d %H:%M:%S ET"),
            "kr_market_phase": kr_phase.value,
            "kr_market_time_kst": kr_now.strftime("%Y-%m-%d %H:%M:%S KST"),
            "tasks": tasks,
        }
