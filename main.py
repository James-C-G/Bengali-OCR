from demo import *


# Run application
def main(model):
    SimpleKivy4().run()
    test_image("user_image.png", model)


if __name__ == "__main__":
    main(CHAR_MODEL)



