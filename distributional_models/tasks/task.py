class Task:

    def __init__(self):
        pass

    @staticmethod
    def load_category_file(file_path):
        target_category_dict = {}
        with open(file_path) as f:
            for line in f:
                data = (line.strip().strip('\n').strip()).split(",")
                target_category_dict[data[0]] = data[1]
        return target_category_dict
