
import pandas as pd
import sys
import sysconfig

df = pd.DataFrame()

print("Path", sys.path)
print("Conf1",sysconfig.get_path_names)
print("Conf2",sysconfig.get_default_scheme)