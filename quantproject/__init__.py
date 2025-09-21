# 讓 'import quantproject.*' 直接等同於 'import src.*'
import importlib as _imp, sys as _sys
_pkg = _imp.import_module('src')
_sys.modules['quantproject'] = _pkg
