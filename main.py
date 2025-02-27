import glob
from src.create_annotations import *
from src.coco_viewer import CocoDataset
# from src.coco_to_yolo import COCO2YOLO

# Label ids of the dataset
category_ids = {
    "crack": 0,
}

# Define which colors match which categories in the images
category_colors = {
    "(255, 255, 255)": 0,  # crack
}

# Define the ids that are a multiplolygon. for example: wall, roof and sky
multipolygon_ids = [0]#, 1, 2,3]


# Get "images" and "annotations" info
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    for mask_image in glob.glob(maskpath + "*.png"):
        # The mask image is *.png but the original image is *.jpg.
        # We make a reference to the original file in the COCO JSON file

        original_file_name = os.path.basename(mask_image).split(".")[0] + ".jpg"
        print(original_file_name)
        # Open the image and (to be sure) we convert it to RGB
        mask_image_open = Image.open(mask_image).convert("RGB")
        w, h = mask_image_open.size

        # "images" info
        image = create_image_annotation(original_file_name, w, h, image_id)
        images.append(image)

        sub_masks = create_sub_masks(mask_image_open, w, h)
        for color, sub_mask in sub_masks.items():
            category_id = category_colors[color]

            # "annotations" info
            polygons, segmentations = create_sub_mask_annotation(sub_mask)

            # Check if we have classes that are a multipolygon
            if category_id in multipolygon_ids:
                # Combine the polygons to calculate the bounding box and area
                multi_poly = MultiPolygon(polygons)

                annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    # Cleaner to recalculate this variable
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    print(i)
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id,
                                                          annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id


if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    mask_path = "dataset/_mask_png/"

    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    with open("output/coco_annotations_val.json", "w") as outfile:
        json.dump(coco_format, outfile)


        print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))


# # View the results
# instances_json_path = "output/train.json"
# images_path = "dataset/train"
# coco_dataset = CocoDataset(instances_json_path, images_path)
# coco_dataset.display_info()
# coco_dataset.display_licenses()
# coco_dataset.display_categories()
#
# html = coco_dataset.display_image(0)
# f = open('html.html', 'w')
# f.write(html)
# f.close()
#
# print("html created!")
# # convert COCO to YOLO file

# IPython.display.HTML(html)
