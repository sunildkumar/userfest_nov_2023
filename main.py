import glob
import os
import csv
import random
from groundlight import Groundlight
import json
import time

path_to_images = "/Users/sunilkumar/Documents/userfest_nov_2023/IndustryBiscuit/Images"
# Path to the CSV file
path_to_annotations = (
    "/Users/sunilkumar/Documents/userfest_nov_2023/IndustryBiscuit/Annotations.csv"
)

random.seed(42)

# Read the CSV file and store the data in a dictionary that maps the filename to the label where True is defective and False is non-defective
data_dict = {}
with open(path_to_annotations, mode="r", encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)  # Skip the header row
    for row in csvreader:
        # Assuming the first column is the filename
        file_name = row[0]
        is_defective = row[2]

        data_dict[file_name] = bool(int(is_defective))


jpg_files = glob.glob(os.path.join(path_to_images, "*.jpg"))

images = []

for file_path in jpg_files:
    file_name = str(os.path.basename(file_path))
    images.append((str(file_path), file_name))

# shuffle the images deterministically
random.shuffle(images)

gl = Groundlight()
# now we can submit the images to groundlight for labeling
# test_query_text = "This is a test, simply answer YES"
real_query_text = "Is the biscuit defective? Please read the notes for detailed examples on what is considered defective."
name = "Biscuit Defects Detector v2"
det = gl.get_or_create_detector(
    name=name, query=real_query_text, confidence_threshold=0.9
)

results_dict = {}

num_true = 0
num_false = 0

for image_path, image_name in images:
    gt_label = data_dict[image_name]
    print(f"Submitting {image_name} for labeling with gt label {gt_label}")
    iq = gl.ask_ml(
        detector=det,
        image=image_path,
        wait=15,
    )
    result = iq.result
    predicted_label = result.label
    predicted_confidence = result.confidence

    if predicted_label == "YES":
        predicted_label = True
    elif predicted_label == "NO":
        predicted_label = False
    elif predicted_label == "UNCLEAR":
        predicted_label = None

    # if we've seen less than 10 true examples, provide a ground truth label to groundlight
    if gt_label and num_true < 10:
        gl.add_label(image_query=iq, label="YES")
        num_true += 1

    # if we've seen less than 10 false examples, provide a ground truth label to groundlight
    if not gt_label and num_false < 10:
        gl.add_label(image_query=iq, label="NO")
        num_false += 1

    print(
        f"Predicted label: {predicted_label} with confidence {predicted_confidence}. This is correct: {predicted_label == gt_label}"
    )

    results_dict[image_name] = {
        "predicted_label": predicted_label,
        "predicted_confidence": predicted_confidence,
        "gt_label": gt_label,
    }

    # Save the updated results dictionary to a JSON file after each image
    with open("results.json", "w", encoding="utf-8") as fp:
        json.dump(results_dict, fp, indent=4)

    time.sleep(5)
