import os
import argparse
import json
import numpy as np  # (pip install numpy)
from PIL import Image  # (pip install Pillow)
from skimage import measure  # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon  # (pip install Shapely)

SPLIT_RATIO = 0.9


def create_sub_masks(mask_image):
    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width + 2, height + 2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x + 1, y + 1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation


def thread_function(obj):
    global train_img_id, valid_img_id
    global train_ann_id, valid_ann_id

    id_name, dir_name = obj[0], obj[1]
    path_to_mask_dir = os.path.join(opt.dataset, dir_name, 'masks')
    mask_image_names = [os.path.join(path_to_mask_dir, fn) for fn in os.listdir(path_to_mask_dir)]
    split_id = int(len(mask_image_names) * SPLIT_RATIO)
    train_image_names = mask_image_names[:split_id]
    valid_image_names = mask_image_names[split_id:]
    print("working on dir: {}".format(dir_name))
    # For train images
    for mask_fn in train_image_names:
        mask_image = Image.open(mask_fn)
        rgb_image_name = "frames/frame_" + os.path.basename(mask_fn).split("_")[1]
        rgb_image_name = os.path.splitext(rgb_image_name)[0] + ".jpg"
        train_images_info.append({"file_name": os.path.join(os.path.dirname(os.path.abspath(path_to_mask_dir)),
                                                            rgb_image_name),
                                  "height": mask_image.size[1],
                                  "width": mask_image.size[0],
                                  "id": train_img_id})
        sub_masks = create_sub_masks(mask_image)
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[id_name + 1][color]  # (category_id==(id_name+1))
            annotation = create_sub_mask_annotation(sub_mask, train_img_id, category_id, train_ann_id, is_crowd)
            train_annotations.append(annotation)
            train_ann_id += 1
        train_img_id += 1
    # For validation images
    for mask_fn in valid_image_names:
        mask_image = Image.open(mask_fn)
        rgb_image_name = "frames/frame_" + os.path.basename(mask_fn).split("_")[1]
        rgb_image_name = os.path.splitext(rgb_image_name)[0] + ".jpg"
        valid_images_info.append({"file_name": os.path.join(os.path.dirname(os.path.abspath(path_to_mask_dir)),
                                                            rgb_image_name),
                                  "height": mask_image.size[1],
                                  "width": mask_image.size[0],
                                  "id": valid_img_id})
        sub_masks = create_sub_masks(mask_image)
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[id_name + 1][color]  # (category_id==(id_name+1))
            annotation = create_sub_mask_annotation(sub_mask, valid_img_id, category_id, valid_ann_id, is_crowd)
            valid_annotations.append(annotation)
            valid_ann_id += 1
        valid_img_id += 1
    return


if __name__ == '__main__':

    # get command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help='path to root dir of labeled dataset')
    opt = ap.parse_args()

    # list_of_dirs = [dirname for dirname in os.listdir(opt.dataset) if dirname.split('_')[-1]!='gt']
    list_of_dirs = ["dewalt_tool", "facom_wrench", "hitachi_screw_gun",
                    "cupnoodle", "lipton", "mtrainier", "oolong",
                    "_mustard_bottle", "_potted_meat_can", "_bleach_cleanser", "_power_drill"]

    # Define which colors match which categories in the images
    category_ids = {i: {'255': i, } for i in range(1, 12)}

    is_crowd = 0

    # These ids will be automatically increased as we go
    train_ann_id, valid_ann_id = 1, 1
    train_img_id, valid_img_id = 1, 1

    # Create the annotations
    train_annotations, valid_annotations = [], []
    train_images_info, valid_images_info = [], []

    for object_id, object_dir in enumerate(list_of_dirs):
        thread_function([object_id, object_dir])

    output = {"images": train_images_info, "annotations": train_annotations}
    print("Writing to file train_labels.json...")
    with open("train_labels.json", "w") as outfile:
        json.dump(output, outfile)
        print("Done.")

    output = {"images": valid_images_info, "annotations": valid_annotations}
    print("Writing to file valid_labels.json...")
    with open("valid_labels.json", "w") as outfile:
        json.dump(output, outfile)
        print("Done.")
