import json
import yaml
import os



CONFIG_JSON_FILE = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_YAML_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

print(CONFIG_JSON_FILE)
print(CONFIG_YAML_FILE)

print("file inbuilt: ", __file__)
print("file dirname: ", os.path.dirname(__file__))
