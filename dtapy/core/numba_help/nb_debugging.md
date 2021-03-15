numba debugging boundscheck is always disabled by default, enable it by changing the .numba_config.yaml
. The wdir for running needs to contain this file.
This is supposed to be extended in the future and include some best practices when debugging code that is to be jitted.
Python debuggers do not work (inside jitted functions).
By commenting out the @njit decorators you can still test functions for their functionality
The setting DISABLE_JIT in numba_config.yaml doesn't work properly for typed Lists/ Dicts .. you will have different reslts


###### Numba and dataclasses
dataclasses can be used as shown in the parameters.py file. However, global variables are not updated during 
run time and are set 
once compiled. Therefore you cannot just vary over the parameter sets without recompiling. This can be expensive, we've gotta see how
this re-compilation overhead compares to the run time of the functions. Given that we often run them iteratively 
and will rarely try out different parameters on cheap functions it should be fine.
I don't think it will matter much. See 
https://numba.pydata.org/numba-doc/dev/user/faq.html for more details ..
A hack around this is of course to pass the parameter as the default value to a function argument. We can always
overwrite the argument itself.

