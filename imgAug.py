import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import os, glob

ia.seed(1)


def load_augment_save(path, ext):
    augpath = path + "/augmented_images"
    if not os.path.exists(augpath):
        os.mkdir(augpath)
    ext = "/*." + ext
    print(path+ext)

    images = glob.glob(path + ext)
    print("Loaded images from %a" % path + "\n")
    print(images)

    for e in images:
        img = imageio.imread(e)
        print("Working on image %s" % e)
        seq = iaa.Sequential([
            iaa.Affine(rotate=(-25, 25)),
            iaa.AdditiveGaussianNoise(scale=(10, 60)),
            iaa.Crop(percent=(0, 0.2))
        ])

        img_aug = seq(image=img)
        print("Augmented image %s" % e)

        imageio.imwrite(augpath + "/", img_aug, ext)
        print("Saved image %s on %a" % e % augpath)

    return None


load_augment_save("data/etrims-db_v1/annotations/04_etrims-ds", "png")

# annots = glob.glob("data/etrims-db_v1/annotations/08_etrims-ds/.png")
# for e in annots:
#     if not(e.endswith(".jpg") or (e.endswith(".png"))):
#         annots.remove(e)
#
#
# for image in images:
#     img = imageio.imread(image)
