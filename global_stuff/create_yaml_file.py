import os
from global_stuff.constants import small_troops, medium_troops, large_troops, buildings, BASE_DIR
def create_yaml_file(yaml_file, troops, dataset_name):
    if os.path.exists(yaml_file):
        os.remove(yaml_file)
    with open(yaml_file, "w") as yaml_output:
        yaml_output.write(f"path: /workspace/{dataset_name}\n")
        yaml_output.write("train: images/train\n")
        yaml_output.write("val: images/val\n\n")
        yaml_output.write(f"nc: {len(troops)}\n")
        yaml_output.write("names: \n")
        for i,unit_name in enumerate(troops):
            yaml_output.write(f"  {i}: {unit_name}\n")

if __name__ == "__main__":
    create_yaml_file(BASE_DIR/"large-troop-live/data.yaml", large_troops ,"large-troop-live")
    create_yaml_file(BASE_DIR/"small-troop-live/data.yaml", small_troops ,"small-troop-live")
    create_yaml_file(BASE_DIR/"medium-troop-live/data.yaml", medium_troops ,"medium-troop-live")
    create_yaml_file(BASE_DIR/"building-live/data.yaml", buildings ,"building-live")