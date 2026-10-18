"""Shared subscription backpressure for the two otherwise unchanged B runs."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import time
import uuid

from minimal_rag_experiments import benchmark

RUN = Path(__file__).resolve().parents[2]/'runs/semantic-tool-20261016'
FOLDER = RUN/'transport-control'
POLICY = dict(max_active_model_requests=5, minimum_start_interval_seconds=2,
              minimum_new_thread_interval_seconds=10, max_starting_threads=1,
              rate_limit_cooldown_seconds=30, maximum_cooldown_seconds=120,
              admission_order='FIFO for both startup and active-call slots',
              per_group_concurrency=5,
              scope='Experimental generation and original Enterprise judge request dispatch only',
              prompts_models_tools_call_budgets_and_timeouts_unchanged=True)


def control(action):
    with (FOLDER/'schedule.json').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        state = json.loads(stream.read() or '{}')
        result = action(state)
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream)
        stream.flush()
        return result


def reserve(state, fresh):
    start = max(time.time(), state.get('next_start', 0), state.get('cooldown_until', 0))
    if fresh:
        start = max(start, state.get('next_thread_start', 0))
        state['next_thread_start'] = start + POLICY['minimum_new_thread_interval_seconds']
    state['next_start'] = start + POLICY['minimum_start_interval_seconds']
    return start


def rate_limited(error):
    return '429' in str(error) or 'too many requests' in str(error).lower()


def event(name, **fields):
    with (RUN/'transport-events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(), pid=os.getpid(), event=name, **fields))+'\n')


def cooldown(error):
    details = getattr(error, 'codex_error', None)
    retry = re.search(r'retry[-_ ]after["\s:=]+([0-9.]+)', json.dumps(details), re.I)
    retry_after = float(retry.group(1)) if retry else 0
    def update(state):
        delay = POLICY['rate_limit_cooldown_seconds']
        if time.time() - state.get('last_429', 0) < 120:
            delay = min(POLICY['maximum_cooldown_seconds'], state.get('backoff', delay)*2)
        state.update(last_429=time.time(), backoff=delay)
        until = time.time()+max(delay, retry_after)
        state['cooldown_until'] = max(state.get('cooldown_until', 0), until)
        return state['cooldown_until']
    until = control(update)
    event('rate_limit_cooldown', until=until, error=str(error), codex_error=details)


async def fifo_slot(queue_name, names):
    ticket = {'id': uuid.uuid4().hex, 'pid': os.getpid()}
    control(lambda state: state.setdefault(queue_name, []).append(ticket))

    def admit(state):
        queue = state[queue_name]
        while queue:
            try:
                os.kill(queue[0]['pid'], 0)
                break
            except ProcessLookupError:
                queue.pop(0)
        if not queue or queue[0] != ticket:
            return None
        for name in names:
            stream = (FOLDER/name).open('a+')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                queue.pop(0)
                return stream
            except BlockingIOError:
                stream.close()
        return None

    try:
        while True:
            stream = control(admit)
            if stream is not None:
                return stream
            await asyncio.sleep(.2)
    finally:
        # A timed-out or cancelled waiter must not block the next question.
        control(lambda state: state.update({queue_name: [x for x in state[queue_name] if x != ticket]}))


async def startup_slot():
    return await fifo_slot('startup_waiters', ['thread-start.lock'])


async def slot():
    return await fifo_slot('call_waiters', [f'slot-{n}.lock' for n in range(POLICY['max_active_model_requests'])])


def install():
    from openai_codex import _run
    FOLDER.mkdir(exist_ok=True)
    (RUN/'transport-policy.json').write_text(json.dumps(POLICY, indent=2)+'\n')
    original_call = benchmark.rag.CodexModel.__call__
    original_fatal = benchmark.fatal
    original_raise = _run._raise_for_failed_turn

    def capture_error(turn):
        try:
            original_raise(turn)
        except Exception as error:
            error.codex_error = turn.error.model_dump(mode='json', by_alias=True) if turn.error else None
            event('upstream_turn_error', turn_id=turn.id, codex_error=error.codex_error)
            raise
    _run._raise_for_failed_turn = capture_error

    async def queued_call(self, messages, tools):
        queued_at = time.time()
        fresh = messages is not self._messages
        startup = await startup_slot() if fresh else None
        stream = None
        started = None
        try:
            stream = await slot()
            while True:
                start = control(lambda state: reserve(state, fresh))
                await asyncio.sleep(max(0, start-time.time()))
                if control(lambda state: state.get('cooldown_until', 0)) <= start:
                    break
            # The original adapter still owns the model timeout and records every real call.
            started = time.time()
            event('model_call_started', fresh=fresh, effort=self.effort,
                  queue_seconds=started-queued_at)
            result = await original_call(self, messages, tools)
            event('model_call_completed', fresh=fresh, seconds=time.time()-started,
                  thread_id=self._thread.id)
            return result
        except Exception as error:
            if rate_limited(error):
                cooldown(error)
            raise
        finally:
            if stream is not None:
                stream.close()
            if startup is not None:
                startup.close()

    benchmark.rag.CodexModel.__call__ = queued_call
    benchmark.fatal = lambda error: (original_fatal(error) or rate_limited(error)
                                     or 'Codex process is not running' in str(error))


def wait_upstream():
    while not (RUN/'STOP').exists():
        quota = json.loads((RUN/'quota-state.json').read_text())
        until = control(lambda state: state.get('cooldown_until', 0))
        if quota.get('ordinary_usage_allowed') is True and time.time() >= until:
            return True
        time.sleep(5)
    return False
