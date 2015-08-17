import logging
import numpy as np
import pyopencl as cl
import pytest

from nengo.utils.compat import range

from nengo_ocl.raggedarray import RaggedArray as RA
from nengo_ocl.clraggedarray import CLRaggedArray as CLRA
from nengo_ocl.clra_gemv import (
    plan_ragged_gather_gemv, plan_many_dots, plan_reduce, plan_ref)


ctx = cl.create_some_context()
logger = logging.getLogger(__name__)


def pytest_generate_tests(metafunc):
    if "planner" in metafunc.funcargnames:
        metafunc.parametrize(
            "planner", [plan_ref, plan_reduce, plan_many_dots])


def allclose(raA, raB):
    assert len(raA) == len(raB)
    for i in range(len(raA)):
        if not np.allclose(raA[i], raB[i]):
            return False
    return True


def test_basic():
    # -- prepare initial conditions on host
    A = RA([[[0.1, .2], [.3, .4]], [[.5, .6]]])
    X = RA([[3, 5]])
    Y = RA([[0.0], [2, 3], ])
    A_js = RA([[1], [0]])
    X_js = RA([[0], [0]])
    alpha = 0.5
    beta = 0.1

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    clA_js = CLRA(queue, A_js)
    clX_js = CLRA(queue, X_js)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)
    assert allclose(A_js, clA_js)
    assert allclose(X_js, clX_js)

    # -- run cl computation
    prog = plan_ragged_gather_gemv(
        queue, alpha, clA, clA_js, clX, clX_js, beta, clY)
    plans = prog.choose_plans()
    assert len(plans) == 1
    plans[0]()

    # -- ensure they match
    for i in range(len(A_js)):
        aj, xj = int(A_js[i]), int(X_js[i])
        ref = alpha * np.dot(A[aj], X[xj]) + beta * Y[i]
        sim = clY[i]
        assert np.allclose(ref, sim)


def _test_random(k=4, p=1, m=10, n=10):
    """
    Parameters
    ----------
    k : number of operations (length of A_js)
    p : number of dots per operation (width of A_js)
    m : output dimensions
    n : input dimensions
    """

    rng = np.random.RandomState(3294)

    aa = [rng.normal(size=(m, n)) for i in range(k)]
    xx = [rng.normal(size=n) for i in range(k)]
    yy = [rng.normal(size=m) for i in range(k)]
    ajs = [rng.randint(k, size=p) for i in range(k)]
    xjs = [rng.randint(k, size=p) for i in range(k)]

    A = RA(aa)
    X = RA(xx)
    Y = RA(yy)
    A_js = RA(ajs)
    X_js = RA(xjs)
    alpha = 0.5
    beta = 0.1

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    clA_js = CLRA(queue, A_js)
    clX_js = CLRA(queue, X_js)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)
    assert allclose(A_js, clA_js)
    assert allclose(X_js, clX_js)

    # -- run cl computation
    prog = plan_ragged_gather_gemv(
        queue, alpha, clA, clA_js, clX, clX_js, beta, clY)
    plans = prog.choose_plans()

    print('-' * 5 + ' Plans ' + '-' * 45)
    for plan in plans:
        print(plan)
        plan()

    # -- ensure they match
    for i in range(k):
        ref = beta * Y[i]
        for aj, xj in zip(A_js[i], X_js[i]):
            ref += alpha * np.dot(A[aj], X[xj])
        sim = clY[i]
        assert np.allclose(ref, sim, atol=1e-3, rtol=1e-3)


def test_random_small():
    _test_random(k=4, m=10, n=10)


def test_random_large():
    _test_random(k=10, m=550, n=550)


def test_many_dots_small():
    _test_random(k=4, p=4, m=10, n=10)


def test_many_dots_large():
    # _test_random(k=4, p=4, m=550, n=550)
    _test_random(k=4, p=4, m=2000, n=1000)


def check_from_shapes(
    planner,
    alpha, beta, gamma,
    A_shapes, X_shapes,
    A_js,
    X_js,
):
    rng = np.random.RandomState(1234)
    A = RA([0.1 + rng.rand(*shp) for shp in A_shapes])
    X = RA([0.1 + rng.rand(*shp) for shp in X_shapes])
    Y = RA([0.1 + rng.rand(
        A_shapes[A_js[ii][0]][0],
        X_shapes[X_js[ii][0]][1])
        for ii in range(len(A_js))])
    A_js = RA(A_js)
    X_js = RA(X_js)
    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    clA_js = CLRA(queue, A_js)
    clX_js = CLRA(queue, X_js)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)
    assert allclose(A_js, clA_js)
    assert allclose(X_js, clX_js)

    # -- run cl computation
    prog = planner(
        queue, alpha, clA, clA_js, clX, clX_js, beta, clY, gamma=gamma)
    plans = prog.choose_plans()
    assert len(plans) == 1
    plans[0]()

    # -- ensure they match
    for i in range(len(A_js)):
        # print('gamma', gamma)
        # print('Y[i] * beta + gamma', Y[i] * beta + gamma)
        # print(A[0])
        # print(X[0])
        # print('AX', sum(
        #     [np.dot(A[aj], X[xj])
        #     for aj, xj in zip(A_js[i], X_js[i])]))
        ref = gamma + beta * Y[i] + alpha * sum(
            [np.dot(A[aj], X[xj])
             for aj, xj in zip(A_js[i], X_js[i])])
        sim = clY[i]
        if not np.allclose(ref, sim, atol=1e-3, rtol=1e-3):
            print('A_shapes',  A_shapes)
            print('X_shapes', X_shapes)
            if len(ref) > 20:
                print('ref', ref[:10], '...', ref[-10:])
                print('sim', sim[:10], '...', sim[-10:])
            else:
                print('ref', ref)
                print('sim', sim)
            assert 0


def test_one_element(planner):
    check_from_shapes(
        planner,
        0.5, 0.6, 0.7,
        A_shapes=[(1, 1)],
        X_shapes=[(1, 1)],
        A_js=[[0]],
        X_js=[[0]])


def test_one_short_segment(planner):
    check_from_shapes(
        planner,
        0.5, 0.6, 0.7,
        A_shapes=[(10, 1)],
        X_shapes=[(1, 1)],
        A_js=[[0]],
        X_js=[[0]])


def test_one_long_segment(planner):
    check_from_shapes(
        planner,
        0.5, 0.6, 0.7,
        A_shapes=[(2001, 1)],
        X_shapes=[(1, 1)],
        A_js=[[0]],
        X_js=[[0]])


def test_one_short_segment_many_dots(planner):
    for ND in 2, 20, 100:
        check_from_shapes(
            planner,
            0.5, 0.6, 0.7,
            A_shapes=[(10, 1 + ii % 2) for ii in range(ND)],
            X_shapes=[(1 + ii % 2, 1) for ii in range(ND)],
            A_js=[range(ND)],
            X_js=[range(ND)])


def test_one_short_segment_many_longer_dots(planner):
    for ND in 2, 20, 100:
        check_from_shapes(
            planner,
            0.5, 0.6, 0.7,
            A_shapes=[(2000, ii + 1) for ii in range(ND)],
            X_shapes=[(ii + 1, 1) for ii in range(ND)],
            A_js=[range(ND)],
            X_js=[range(ND)])


def test_speed():
    import time

    rng = np.random.RandomState(3294)

    k = 300
    # k = 100
    ms = [rng.randint(100, 1000) for i in range(k)]
    ns = [rng.randint(100, 1000) for i in range(k)]

    aa = [rng.normal(size=(m, n)).astype('float32') for m, n in zip(ms, ns)]
    xx = [rng.normal(size=n).astype('float32') for n in ns]
    yy = [rng.normal(size=m).astype('float32') for m in ms]
    ajs = list(range(k))
    xjs = list(range(k))
    # ajs = [rng.randint(k, size=p) for i in range(k)]
    # xjs = [rng.randint(k, size=p) for i in range(k)]

    A = RA(aa)
    X = RA(xx)
    Y = RA(yy)
    A_js = RA(ajs)
    X_js = RA(xjs)
    alpha = 0.5
    beta = 0.1

    # -- prepare initial conditions on device
    queue = cl.CommandQueue(ctx)
    # queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    clA = CLRA(queue, A)
    clX = CLRA(queue, X)
    clY = CLRA(queue, Y)
    clA_js = CLRA(queue, A_js)
    clX_js = CLRA(queue, X_js)
    assert allclose(A, clA)
    assert allclose(X, clX)
    assert allclose(Y, clY)
    assert allclose(A_js, clA_js)
    assert allclose(X_js, clX_js)

    # -- speed test in ocl blas
    from pyopencl.array import Array
    import pyopencl_blas as blas
    blas.setup()

    def array(a):
        cla = Array(queue, a.shape, a.dtype)
        cla.set(a)
        return cla

    clAs = [array(a) for a in A]
    clXs = [array(x.ravel()) for x in X]
    clYs = [array(y.ravel()) for y in Y]

    # queues = [cl.CommandQueue(ctx) for _ in range(k)]
    queues = [cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
              for _ in range(k)]

    t0 = time.time()
    # for A, X, Y in zip(clAs, clXs, clYs):
    #     blas.gemv(queue, A, X, Y)
    # queue.finish()
    for i, [A, X, Y] in enumerate(zip(clAs, clXs, clYs)):
        q = queues[i % len(queues)]
        blas.gemv(q, A, X, Y)
        q.flush()
    # for q in queues:
    #     q.flush()
    for q in queues:
        q.finish()
    t0 = time.time() - t0
    print("clBLAS: %0.3f" % t0)

    # -- run cl computation
    prog = plan_ragged_gather_gemv(
        queue, alpha, clA, clA_js, clX, clX_js, beta, clY)
    plans = prog.choose_plans()

    print('-' * 5 + ' Plans ' + '-' * 45)
    for plan in plans:
        print(plan)

    t = time.time()
    for plan in plans:
        plan()
    t = time.time() - t
    print("nengo_ocl: %0.3f" % t)

    # # -- ensure they match
    # for i in range(k):
    #     ref = beta * Y[i]
    #     for aj, xj in zip(A_js[i], X_js[i]):
    #         ref += alpha * np.dot(A[aj], X[xj])
    #     sim = clY[i]
    #     assert np.allclose(ref, sim, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
