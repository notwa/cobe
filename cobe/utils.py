# Copyright (C) 2012 Peter Teichman

import itertools
import time


def itime(iterable, seconds):
    """Yield items from seq until enough time has passed.

    itime yields items from seq until more time than 'seconds' has
    passed. It always yields at least one item.

    This does nothing to stop a long-running sequence operation; it is
    possible that itime will take considerably longer than 'seconds'
    to yield, but it will not yield another item once that has
    occurred.

    """
    end = time.time() + seconds

    items = iter(iterable)
    yield items.next()

    for item in itertools.takewhile(lambda _: time.time() < end, items):
        yield item