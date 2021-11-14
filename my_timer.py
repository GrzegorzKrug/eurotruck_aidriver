import time

from functools import wraps


def timeit_mean(N=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tm_sum = 0
            for n in range(N):
                t0 = time.time()

                out = func(*args, **kwargs)

                tend = time.time()
                dur = tend - t0
                tm_sum += dur

            dur = tm_sum / N
            name = func.__qualname__
            if dur < 1:
                print(f"Mean Execution({N}): {name:<30} was {dur * 1000:>4.3f} ms")
            elif dur > 60:
                print(f"Mean Execution({N}): {name:<30} was {dur / 60:>4.3f} m")
            else:
                print(f"Mean Execution({N}): {name:<30} was {dur:>4.3f} s")

            return out

        return wrapper

    return decorator


def single_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()

        out = func(*args, **kwargs)

        tend = time.time()
        dur = tend - t0
        name = func.__qualname__
        if dur < 1:
            print(f"Execution: {name:<30} was {dur * 1000:>4.3f} ms")
        elif dur > 60:
            print(f"Execution: {name:<30} was {dur / 60:>4.3f} m")
        else:
            print(f"Execution: {name:<30} was {dur:>4.3f} s")

        return out

    return wrapper
