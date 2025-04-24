import os
import ast
import rich
import types
import inspect
import textwrap
import importlib
from rich.progress import track


def map_pyi_files_to_modules(base_dir: str):
    module_map: dict[str, str] = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pyi"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                parts = rel_path.split(os.sep)

                # Remove file extension
                if parts[-1] == "__init__.pyi":
                    parts = parts[:-1]  # drop __init__.pyi to get package name
                else:
                    parts[-1] = parts[-1][:-4]  # remove .pyi

                module_name = ".".join(parts)
                module_map[full_path] = module_name
    return module_map


fn_template = '''
def {name}{sig}:
    """
{doc}
    """
'''


rich.print("Finding stubgen stubs...")
stubfiles = map_pyi_files_to_modules("out")
for stubfile, mod in track(stubfiles.items(), description="Mapping Cython..."):
    any_mod = False
    with open(stubfile) as fi:
        stub = fi.read()
        modified_stub = []
        for line in stub.splitlines():
            if line.startswith("import _cython_3_0_"):
                continue
            try:
                st = ast.parse(line)
                if len(st.body) == 1 and isinstance(st.body[0], ast.AnnAssign):
                    ann: ast.AnnAssign = st.body[0]
                    if isinstance(ann.target, ast.Name):
                        name = ann.target.id
                        if 'cython_function' in ast.unparse(ann.annotation):
                            concrete_mod = importlib.import_module(mod)
                            fn: types.FunctionType = getattr(concrete_mod, name)
                            sig = inspect.signature(fn)
                            recon = fn_template.format(name=name, sig=sig, doc=textwrap.indent(fn.__doc__.strip(), " " * 4))
                            line = recon
                            any_mod = True
            except SyntaxError:
                pass
            modified_stub.append(line)
    if any_mod:
        modified_stub.insert(0, "from typing import *")
    output_prem = os.path.join("cuda-stubs", os.path.relpath(stubfile, "out/cuda"))
    os.makedirs(os.path.dirname(output_prem), exist_ok=True)
    with open(output_prem, "w") as fo:
        fo.write('\n'.join(modified_stub))
