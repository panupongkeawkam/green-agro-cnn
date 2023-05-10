import os
root_dir = "./test"

for child_dir in os.listdir(root_dir):
    current_path = os.path.join(root_dir, child_dir)
    count = 1
    for file_name in os.listdir(current_path):
        path_file = os.path.join(current_path, file_name)

        new_file_name = child_dir + "_" + str(count) + ".jpeg"

        new_path = os.path.join(current_path, new_file_name)

        try:
            os.rename(path_file, new_path)
        except:
            print("Cannot rename")

        count += 1
