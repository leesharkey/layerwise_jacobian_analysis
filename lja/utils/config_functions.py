import yaml, munch
import os, shutil


def load_configs(configs_name):
    print("[Loading configurations]")
    with open("configs/configs.yml", "r") as f:
        hp = yaml.safe_load(f)[configs_name]
    for key, value in hp.items():
        print(key, ":", value)
    hp = munch.munchify(hp)
    return hp


def save_configs_file(uniq_id):
    # TODO add git hash to configs so we know what code version the experiment was run with.
    config_file_str = "configs.yml"
    config_file_path = os.path.join("configs", config_file_str)
    end = config_file_str.find(".yml")
    uniq_config_name = (
        config_file_str[:end] + "_" + str(uniq_id) + config_file_str[end:]
    )
    save_dir = "configs/config_saves"
    os.makedirs(save_dir, exist_ok=True)
    uniq_config_name = os.path.join(save_dir, uniq_config_name)
    shutil.copy(config_file_path, uniq_config_name)
    print(f"Config file saved for your records to {uniq_config_name}")
