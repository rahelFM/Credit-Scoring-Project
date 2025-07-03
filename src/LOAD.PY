import os

folders = [
     "C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\src",
    "C:\\Users\\Rahel\\Desktop\\KAIM5\\Week 5\\Credit-Scoring-Project\\src\\api"
]

for folder in folders:
    init_path = os.path.join(folder, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# This file makes this folder a Python package\n")
        print(f"Created: {init_path}")
    else:
        print(f"Already exists: {init_path}")
