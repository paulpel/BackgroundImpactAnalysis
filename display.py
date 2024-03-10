class Display(object):
    
    def print_class_descriptions(self, class_desriptions, specific=None):
        if specific:
            for id in specific:
                print(f"{id}: {class_desriptions[id]}")
        else:
            for id, description in class_desriptions.items():
                print(f"{id}: {description}")

