from display import Display

class Config(object):

    def __init__(self) -> None:
        self.mapping_path = "./data/LOC_synset_mapping.txt"
        self.images_path = "./images/test/"
        self.class_descriptions = self.get_mapping(self.mapping_path)

    def get_mapping(self, mapping_path):
        class_descriptions = {}

        with open(mapping_path, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                class_name = parts[0]
                description = ' '.join(parts[1:])
                class_descriptions[class_name] = description

        return class_descriptions


if __name__ == "__main__":
    obj_cg = Config()
    obj_display = Display()

    obj_display.print_class_descriptions(
        obj_cg.class_descriptions
    )

