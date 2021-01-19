# global
import logging
import numpy as np


# Framework Getters #
# ------------------#

def try_import_ivy_jax(warn=False):
    try:
        import ivy.jax
        return ivy.jax
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\nEither jax or jaxlib appear to not be installed, '
                        'ivy.jax can therefore not be imported.\n'.format(e))


def try_import_ivy_tf(warn=False):
    try:
        import ivy.tensorflow
        return ivy.tensorflow
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\ntensorflow does not appear to be installed, '
                        'ivy.tensorflow can therefore not be imported.\n'.format(e))


def try_import_ivy_torch(warn=False):
    try:
        import ivy.torch
        return ivy.torch
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\ntorch does not appear to be installed, '
                        'ivy.torch can therefore not be imported.\n'.format(e))


def try_import_ivy_mxnd(warn=False):
    try:
        import ivy.mxnd
        return ivy.mxnd
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\nmxnet does not appear to be installed, '
                        'ivy.mxnd can therefore not be imported.\n'.format(e))


def try_import_ivy_numpy(warn=False):
    try:
        import ivy.numpy
        return ivy.numpy
    except (ImportError, ModuleNotFoundError) as e:
        if not warn:
            return
        logging.warning('{}\n\nnumpy does not appear to be installed, '
                        'ivy.numpy can therefore not be imported.\n'.format(e))


# Framework Selection #
# --------------------#

FW_DICT = {'jax': try_import_ivy_jax,
           'tensorflow': try_import_ivy_tf,
           'torch': try_import_ivy_torch,
           'mxnd': try_import_ivy_mxnd,
           'numpy': try_import_ivy_numpy}


def get_framework_from_str(f_str):
    return FW_DICT[f_str](warn=True)


def choose_random_framework(excluded=None):
    excluded = list() if excluded is None else excluded
    while True:
        if len(excluded) == 5:
            raise Exception('Unable to select framework, all backends are either excluded or not installed.')
        f_key = np.random.choice([f_srt for f_srt in list(FW_DICT.keys()) if f_srt not in excluded])
        f = get_framework_from_str(f_key)
        if f is None:
            excluded.append(f_key)
            continue
        else:
            print('\nselected framework: {}\n'.format(f_key))
            return f
