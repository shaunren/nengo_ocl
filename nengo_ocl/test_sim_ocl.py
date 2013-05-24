
import sim_ocl
import pyopencl as cl

import test_sim_npy

ctx = cl.create_some_context()

def test_probe_with_base(show=True):
    def Simulator(*args, **kwargs):
        return sim_ocl.Simulator(ctx, *args, **kwargs)
    return test_sim_npy.test_probe_with_base(show=show,
                                             Simulator=Simulator,
                                             skip_loop_asserts=True,
                                            )
