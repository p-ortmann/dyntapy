#  This file is part of the traffic assignment code base developed at KU Leuven.
#  Copyright (c) 2020 Paul Ortmann
#  License: GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, see license.txt
#  More information at: https://gitlab.kuleuven.be/ITSCreaLab
#  or contact: ITScrealab@kuleuven.be
#
#
import ctypes
import importlib
from numba.core import cgutils, config, types
from numba.core.withcontexts import bypass_context
from numba.core.extending import intrinsic, register_jitable



# printf
@intrinsic
def _printf(typingctx, format_type, *args):
    """printf that can be called from Numba jit-decorated functions.
    """
    if isinstance(format_type, types.StringLiteral):
        sig = types.void(format_type, types.BaseTuple.from_types(args))

        def codegen(context, builder, signature, args):
            cgutils.printf(builder, format_type.literal_value, *args[1:])
        return sig, codegen


@register_jitable
def printf(format_type, *args):
    if config.DISABLE_JIT:
        with bypass_context:
            print(format_type % args, end='')
    else:
        _printf(format_type, args)


def __dynamic_import(abs_module_path, class_name):
    module_object = importlib.import_module(abs_module_path)
    target_class = getattr(module_object, class_name)
    return target_class
sys_bitwidth = ctypes.sizeof(ctypes.c_voidp) * 8
int_dtype = __dynamic_import('numba.core.types', 'int' + str(sys_bitwidth))
float_dtype = __dynamic_import('numba.core.types', 'float' + str(sys_bitwidth))
