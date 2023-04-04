from .pub_mod import *
from .unet import *
from .DDG import *
from .decoder import *
from .cnsn import *
def get_model(name='', max_iter=4000, use_cn=False, use_sn=True):
    model = SSAN_M(max_iter=max_iter, use_cn=use_cn, use_sn=use_sn)
    return model