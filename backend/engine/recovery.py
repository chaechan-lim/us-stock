"""Error recovery and circuit breaker for the trading engine.

Provides:
- CircuitBreaker: prevents cascading failures by stopping calls to failing services
- TaskRecovery: wraps scheduler tasks with retry logic and circuit breakers
"""

import asyncio
import logging
import random
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern for protecting external service calls.

    States:
      CLOSED   → normal operation, track failures
      OPEN     → failures exceeded threshold, reject all calls
      HALF_OPEN → cooldown elapsed, allow one test call
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_sec: float = 60.0,
        half_open_max: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self.cooldown_sec:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit %s: OPEN → HALF_OPEN", self.name)
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.half_open_max
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info("Circuit %s: HALF_OPEN → CLOSED (recovered)", self.name)
        else:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit %s: HALF_OPEN → OPEN (still failing)", self.name)
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit %s: CLOSED → OPEN (failures=%d)",
                self.name, self._failure_count,
            )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0
        logger.info("Circuit %s: manually reset", self.name)

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_sec": self.cooldown_sec,
        }

    async def call(self, fn, *args, **kwargs):
        """Execute fn through the circuit breaker."""
        if not self.allow_request():
            raise CircuitOpenError(
                f"Circuit '{self.name}' is OPEN "
                f"(failures={self._failure_count}, "
                f"cooldown={self.cooldown_sec}s)"
            )

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = await fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open and rejecting calls."""


class TaskRecovery:
    """Wraps a scheduler task with retry + circuit breaker logic.

    Usage:
        recovery = TaskRecovery("position_check", task_fn,
                                max_retries=2, backoff_base=5.0)
        await recovery.execute()
    """

    def __init__(
        self,
        name: str,
        fn,
        max_retries: int = 2,
        backoff_base: float = 5.0,
        backoff_max: float = 60.0,
        circuit: CircuitBreaker | None = None,
        on_failure=None,
        on_recovery=None,
    ):
        self.name = name
        self.fn = fn
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.circuit = circuit or CircuitBreaker(name)
        self._on_failure = on_failure
        self._on_recovery = on_recovery
        self._consecutive_failures = 0
        self._total_failures = 0
        self._total_successes = 0

    async def execute(self) -> bool:
        """Execute the task with retry and circuit breaker.

        Returns True if task succeeded, False otherwise.
        """
        if not self.circuit.allow_request():
            logger.debug(
                "Task %s skipped: circuit is %s",
                self.name, self.circuit.state.value,
            )
            return False

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                was_half_open = self.circuit.state == CircuitState.HALF_OPEN
                if was_half_open:
                    self.circuit._half_open_calls += 1

                await self.fn()
                self.circuit.record_success()

                # Notify on recovery from HALF_OPEN (was previously OPEN)
                if was_half_open and self._on_recovery:
                    try:
                        await self._on_recovery(self.name)
                    except Exception as e:
                        logger.warning("Recovery callback failed for %s: %s", self.name, e)

                self._consecutive_failures = 0
                self._total_successes += 1
                return True

            except Exception as e:
                last_error = e
                self._total_failures += 1

                if attempt < self.max_retries:
                    base_delay = min(
                        self.backoff_base * (2 ** attempt),
                        self.backoff_max,
                    )
                    delay = base_delay + random.uniform(0, base_delay * 0.1)
                    logger.warning(
                        "Task %s failed (attempt %d/%d): %s. Retrying in %.1fs",
                        self.name, attempt + 1, self.max_retries + 1, e, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Task %s failed after %d attempts: %s",
                        self.name, self.max_retries + 1, e,
                    )

        # All retries exhausted
        self.circuit.record_failure()
        self._consecutive_failures += 1

        if self._on_failure:
            try:
                await self._on_failure(self.name, last_error)
            except Exception as cb_err:
                logger.error("Failure callback error for %s: %s", self.name, cb_err)

        return False

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "circuit": self.circuit.get_status(),
            "consecutive_failures": self._consecutive_failures,
            "total_failures": self._total_failures,
            "total_successes": self._total_successes,
        }


class RecoveryManager:
    """Central manager for all task recovery instances.

    Integrates with the scheduler and notification service.
    """

    def __init__(self, notification=None):
        self._recoveries: dict[str, TaskRecovery] = {}
        self._notification = notification

    def wrap_task(
        self,
        name: str,
        fn,
        max_retries: int = 2,
        backoff_base: float = 5.0,
        failure_threshold: int = 5,
        cooldown_sec: float = 60.0,
    ) -> TaskRecovery:
        """Create a TaskRecovery wrapper for a scheduler task."""
        circuit = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            cooldown_sec=cooldown_sec,
        )
        recovery = TaskRecovery(
            name=name,
            fn=fn,
            max_retries=max_retries,
            backoff_base=backoff_base,
            circuit=circuit,
            on_failure=self._handle_failure,
            on_recovery=self._handle_recovery,
        )
        self._recoveries[name] = recovery
        return recovery

    async def _handle_failure(self, task_name: str, error: Exception) -> None:
        """Called when a task exhausts all retries."""
        recovery = self._recoveries.get(task_name)
        circuit_state = recovery.circuit.state.value if recovery else "unknown"

        logger.error(
            "Task %s: all retries exhausted (circuit=%s): %s",
            task_name, circuit_state, error,
        )

        if self._notification:
            await self._notification.notify_system_error(
                component=task_name,
                error=str(error),
                details=f"Circuit breaker state: {circuit_state}",
            )

    async def _handle_recovery(self, task_name: str) -> None:
        """Called when a task recovers after circuit was open."""
        logger.info("Task %s: circuit recovered", task_name)
        if self._notification:
            await self._notification.notify_system_event(
                "circuit_recovered",
                f"Task '{task_name}' has recovered and is running normally again.",
            )

    def reset_circuit(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        recovery = self._recoveries.get(name)
        if recovery:
            recovery.circuit.reset()
            return True
        return False

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for recovery in self._recoveries.values():
            recovery.circuit.reset()

    def get_status(self) -> dict:
        return {
            name: r.get_status()
            for name, r in self._recoveries.items()
        }
