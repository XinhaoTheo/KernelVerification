"""Compatibility entry point: pair trials now use the shared GLM trace runner.

Historical isolated records remain in traces_fireworks/; new runs are written to
benchmark_fn_fp/traces_glm/<case>/single_call/<trial>/.
"""
from pathlib import Path
import sys

EVAL = Path(__file__).resolve().parent.parent / 'eval_scripts'
sys.path.insert(0,str(EVAL))
from run_single_fireworks import main

if __name__=='__main__':
    args=sys.argv[1:]
    if '--cases' not in args and '--all' not in args:
        args += ['--cases','case_36,case_37']
    main([*args,'--dataset','correlation_pair'])
