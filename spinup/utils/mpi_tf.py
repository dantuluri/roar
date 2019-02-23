import numpy as np
import math
import tensorflow as tf
from mpi4py import MPI
from spinup.utils.mpi_tools import broadcast


def flat_concat(xs):
    return tf.concat([tf.reshape(x,(-1,)) for x in xs], axis=0)

def assign_params_from_flat(x, params):
    flat_size = lambda p : int(np.prod(p.shape.as_list())) # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([tf.assign(p, p_new) for p, p_new in zip(params, new_params)])

def sync_params(params):
    get_params = flat_concat(params)
    def _broadcast(x):
        broadcast(x)
        return x
    synced_params = tf.py_func(_broadcast, [get_params], tf.float32)
    return assign_params_from_flat(synced_params, params)

def sync_all_params():
    """Sync all tf variables across MPI processes."""
    return sync_params(tf.global_variables())


class MpiAdadeltaOptimizer(tf.train.AdadeltaOptimizer):
    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.AdadeltaOptimizer.__init__(self, **kwargs)
        print("tf.train.AdadeltaOptimizer Called")        
            

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiAdagradOptimizer(tf.train.AdagradOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.AdagradOptimizer.__init__(self, **kwargs)
        print("tf.train.AdagradOptimizer Called")    
             

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiAdamOptimizer(tf.train.AdamOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.AdamOptimizer.__init__(self, **kwargs)
        print("tf.train.AdamOptimizer Called")    

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiFtrlOptimizer(tf.train.FtrlOptimizer):


    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.FtrlOptimizer.__init__(self, **kwargs)
        print("tf.train.FtrlOptimizer Called")        

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiGradientDescentOptimizer(tf.train.GradientDescentOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.GradientDescentOptimizer.__init__(self, **kwargs)
        print("tf.train.GradientDescentOptimizer Called")    
               

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiMomentumOptimizer(tf.train.MomentumOptimizer):


    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.MomentumOptimizer.__init__(self, **kwargs)
        print("tf.train.MomentumOptimizer Called")   
               

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiProximalAdagradOptimizer(tf.train.ProximalAdagradOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.ProximalAdagradOptimizer.__init__(self, **kwargs)
        print("tf.train.ProximalAdagradOptimizer Called")       
        

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiProximalGradientDescentOptimizer(tf.train.ProximalGradientDescentOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.ProximalGradientDescentOptimizer.__init__(self, **kwargs)
        print("ProximalGradientDescentOptimizer Called")       

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiRMSPropOptimizer(tf.train.RMSPropOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.train.RMSPropOptimizer.__init__(self, **kwargs)
        print("RMSPropOptimizer Called") 
             

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiAdaMaxOptimizer(tf.contrib.opt.AdaMaxOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.AdaMaxOptimizer.__init__(self, **kwargs)
        print("tf.contrib.opt.AdaMaxOptimizer Called") 
         

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiAdamGSOptimizer(tf.contrib.opt.AdamGSOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.AdamGSOptimizer.__init__(self, **kwargs)
        print("tf.contrib.opt.AdamGSOptimizer Called")     
        

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiAdamWOptimizer(tf.contrib.opt.AdamWOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.AdamWOptimizer.__init__(self, **kwargs)
        print("tf.train.AdamWOptimizer Called")    
        
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiAddSignOptimizer(tf.contrib.opt.AddSignOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.AddSignOptimizer.__init__(self, **kwargs)
        print("tf.train.AddSignOptimizer Called")    
        
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiGGTOptimizer(tf.contrib.opt.GGTOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.GGTOptimizer.__init__(self, **kwargs)
        print("tf.train.GGTOptimizer Called")    
            train_pi = MpiAdamOptimizer(learning_rate=pi_lr,beta1=0.9,
        use_locking=False,
        name='GGT',
        window=10,
        eps=0.0001,
        svd_eps=1e-06,
        sigma_eps=0.01).minimize(pi_loss)
        train_v = MpiAdamOptimizer(learning_rate=vf_lr,beta1=0.9,
        use_locking=False,
        name='GGT',
        window=10,
        eps=0.0001,
        svd_eps=1e-06,
        sigma_eps=0.01).minimize(v_loss)

   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiLARSOptimizer(tf.contrib.opt.LARSOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.LARSOptimizer.__init__(self, **kwargs)
        print("tf.train.LARSOptimizer Called")    
        
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiLazyAdamGSOptimizer(tf.contrib.opt.LazyAdamGSOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.LazyAdamGSOptimizer.__init__(self, **kwargs)
        print("tf.train.LazyAdamGSOptimizer Called")    
            train_pi = MpiAdamOptimizer(global_step=0,
        learning_rate=pi_lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam').minimize(pi_loss)
        train_v = MpiAdamOptimizer(global_step=0,
        learning_rate=vf_lr,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam').minimize(v_loss)
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiLazyAdamOptimizer(tf.contrib.opt.LazyAdamOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.LazyAdamOptimizer.__init__(self, **kwargs)
        print("tf.train.LazyAdamOptimizer Called")    
        

   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiMomentumWOptimizer(tf.contrib.opt.MomentumWOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.MomentumWOptimizer.__init__(self, **kwargs)
        print("tf.train.AdamOptimizer Called")    
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiNadamOptimizer(tf.contrib.opt.NadamOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.NadamOptimizer.__init__(self, **kwargs)
        print("tf.contrib.opt.NadamOptimizer Called")    
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiPowerSignOptimizer(tf.contrib.opt.PowerSignOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.PowerSignOptimizer.__init__(self, **kwargs)
        print("tf.contrib.opt.PowerSignOptimizer Called")    
   

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])

class MpiShampooOptimizer(tf.contrib.opt.ShampooOptimizer):

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.ShampooOptimizer.__init__(self, **kwargs)
        print("tf.contrib.opt.ShampooOptimizer Called")           

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])


class nothing(tf.contrib.opt.ShampooOptimizer):
    """
    weight_decay should be 0.000001
    momentum=0.01

    tf.contrib.opt.RegAdagradOptimizer
    tf.contrib.opt.AdaMaxOptimizer
    tf.contrib.opt.AdamGSOptimizer
    
    tf.contrib.opt.AdamWOptimizer
    train_pi = MpiAdamOptimizer(weight_decay=0.0001,learning_rate=pi_lr).minimize(pi_loss)
    train_v = MpiAdamOptimizer(weight_decay=0.0001,learning_rate=vf_lr).minimize(v_loss)

    tf.contrib.opt.AddSignOptimizer

tf.contrib.opt.GGTOptimizer
 train_pi = MpiAdamOptimizer(learning_rate=pi_lr,beta1=0.9,
    use_locking=False,
    name='GGT',
    window=10,
    eps=0.0001,
    svd_eps=1e-06,
    sigma_eps=0.01).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr,beta1=0.9,
    use_locking=False,
    name='GGT',
    window=10,
    eps=0.0001,
    svd_eps=1e-06,
    sigma_eps=0.01).minimize(v_loss)

tf.contrib.opt.LARSOptimizer
train_pi = MpiAdamOptimizer(learning_rate=pi_lr,momentum=0.9,
    weight_decay=0.0001,
    eeta=0.001,
    epsilon=0.0,
    name='LARSOptimizer',
    skip_list=None,
    use_nesterov=False).minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr,momentum=0.9,
    weight_decay=0.0001,
    eeta=0.001,
    epsilon=0.0,
    name='LARSOptimizer',
    skip_list=None,
    use_nesterov=False).minimize(v_loss)

tf.contrib.opt.LazyAdamGSOptimizer
    train_pi = MpiAdamOptimizer(global_step=0,
    learning_rate=pi_lr,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(pi_loss)
    train_v = MpiAdamOptimizer(global_step=0,
    learning_rate=vf_lr,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(v_loss)

    tf.contrib.opt.LazyAdamOptimizer
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr,beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr,beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(v_loss)

    tf.contrib.opt.MomentumWOptimizer
    train_pi = MpiAdamOptimizer(weight_decay=0.000001,
    learning_rate=pi_lr,
    momentum=0.01,
    use_locking=False,
    name='MomentumW',
    use_nesterov=False).minimize(pi_loss)
    train_v = MpiAdamOptimizer(weight_decay=0.000001,
    learning_rate=vf_lr,
    momentum=0.01,
    use_locking=False,
    name='MomentumW',
    use_nesterov=False).minimize(v_loss)

    tf.contrib.opt.NadamOptimizer
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr,beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr,beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(v_loss)

    tf.contrib.opt.PowerSignOptimizer
    train_pi = MpiAdamOptimizer(learning_rate=pi_lr,base=math.e,
    beta=0.9,
    sign_decay_fn=None,
    use_locking=False,
    name='PowerSignOptimizer').minimize(pi_loss)
    train_v = MpiAdamOptimizer(learning_rate=vf_lr,base=math.e,
    beta=0.9,
    sign_decay_fn=None,
    use_locking=False,
    name='PowerSignOptimizer').minimize(v_loss)

    tf.contrib.opt.ShampooOptimizer CHECK LEARNING RATES
    train_pi = MpiAdamOptimizer(global_step=0,
    max_matrix_size=768,
    gbar_decay=0.0,
    gbar_weight=1.0,
    mat_gbar_decay=1.0,
    mat_gbar_weight=1.0,
    learning_rate=1.0,
    svd_interval=1,
    precond_update_interval=1,
    epsilon=0.0001,
    alpha=0.5,
    use_iterative_root=False,
    use_locking=False,
    name='Shampoo').minimize(pi_loss)
    train_v = MpiAdamOptimizer(global_step=0,
    max_matrix_size=768,
    gbar_decay=0.0,
    gbar_weight=1.0,
    mat_gbar_decay=1.0,
    mat_gbar_weight=1.0,
    learning_rate=1.0,
    svd_interval=1,
    precond_update_interval=1,
    epsilon=0.0001,
    alpha=0.5,
    use_iterative_root=False,
    use_locking=False,
    name='Shampoo').minimize(v_loss)





    The compute_gradients method is taken from Baselines `MpiAdamOptimizer`_. 
    For documentation on method arguments, see the Tensorflow docs page for 
    the base `AdamOptimizer`_.

    .. _`MpiAdamOptimizer`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py
    .. _`AdamOptimizer`: https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
    """

    def __init__(self, **kwargs):
        self.comm = MPI.COMM_WORLD
        tf.contrib.opt.ShampooOptimizer.__init__(self, **kwargs)
        print("tf.contrib.opt.ShampooOptimizer Called")        

    def compute_gradients(self, loss, var_list, **kwargs):
        """
        Same as normal compute_gradients, except average grads over processes.
        """
        grads_and_vars = super().compute_gradients(loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = flat_concat([g for g, v in grads_and_vars])
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        num_tasks = self.comm.Get_size()
        buf = np.zeros(flat_grad.shape, np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                    for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """
        Same as normal apply_gradients, except sync params after update.
        """
        opt = super().apply_gradients(grads_and_vars, global_step, name)
        with tf.control_dependencies([opt]):
            sync = sync_params([v for g,v in grads_and_vars])
        return tf.group([opt, sync])


