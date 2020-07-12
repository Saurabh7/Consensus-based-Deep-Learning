
def save_weight_images():
    import pickle
    import os
    import matplotlib.pyplot as plt
    plt.tight_layout()
    weights_dir = "../../data/arcene/results/weights_2/"
    image_save_dir = os.path.join(weights_dir, "images")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    for j, f in enumerate([x for x in os.listdir(weights_dir) if x.endswith('pkl')]):
        print(j)
        path = os.path.join(weights_dir, f)
        fig, ax = plt.subplots(2, 5, figsize=(50, 20))
        weights_dict = pickle.load(open(path, "rb"))

        for i, axs in enumerate(ax.reshape(-1)):
            y = weights_dict[i]
            axs.imshow(y[1], cmap='Blues')
            axs.set_title("Model {}".format(i))
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(image_save_dir, 'weight_plot_{}'.format(j)))


if __name__ == "__main__":
    save_weight_images()