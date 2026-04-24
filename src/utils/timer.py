import time


class Timer:
    def __init__(self, label="block", verbose=False):
        self.label = label
        self.verbose = verbose

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_time = time.perf_counter() - self.t0
        if self.verbose:
            print(f"[{self.label}] time: {self.elapsed_time:.3f} s")
