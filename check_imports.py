import importlib
import sys
mods = ['torch','transformers','peft','datasets','numpy']
all_ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m} OK")
    except Exception as e:
        print(f"{m} ERROR: {e}")
        all_ok = False
if all_ok:
    print('ALL_OK')
else:
    sys.exit(1)
