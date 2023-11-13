import json
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def accuracy_score(gt_labels, pred_labels):
    return metrics.accuracy_score(gt_labels, pred_labels)


def balanced_accuracy_score(gt_labels, pred_labels):
    return metrics.balanced_accuracy_score(gt_labels, pred_labels)


def precision_score(gt_labels, pred_labels):
    return metrics.precision_score(gt_labels, pred_labels)


def recall_score(gt_labels, pred_labels):
    return metrics.recall_score(gt_labels, pred_labels)


with open("results.json", "r") as file:
    results = json.load(file)

gt_labels = []
pred_labels = []
for key, value in results.items():
    gt_labels.append(value["gt_label"])
    pred_labels.append(value["predicted_label"])

# the x axis is just the total number of elements the metric is being run on
x = list(range(1, len(gt_labels) + 1))

# metrics will be stored in these lists
accuracy_score_results = []
balanced_accuracy_score_results = []
precision_score_results = []
recall_score_results = []


# run the metric on the first element, then the first two elements, then the first three elements, etc.
for i in range(1, len(gt_labels) + 1):
    accuracy_score_results.append(accuracy_score(gt_labels[:i], pred_labels[:i]))
    balanced_accuracy_score_results.append(
        balanced_accuracy_score(gt_labels[:i], pred_labels[:i])
    )
    precision_score_results.append(precision_score(gt_labels[:i], pred_labels[:i]))
    recall_score_results.append(recall_score(gt_labels[:i], pred_labels[:i]))

accuracy_total = accuracy_score(gt_labels, pred_labels)
balanced_accuracy_total = balanced_accuracy_score(gt_labels, pred_labels)
print(f"{accuracy_total=}{balanced_accuracy_total=}")

# create a separate plot for each metric
fig, axs = plt.subplots(4, 1)
axs[0].plot(x, accuracy_score_results)
axs[0].set_title("Accuracy Score")
axs[1].plot(x, balanced_accuracy_score_results)
axs[1].set_title("Balanced Accuracy Score")
axs[2].plot(x, precision_score_results)
axs[2].set_title("Precision Score")
axs[3].plot(x, recall_score_results)
axs[3].set_title("Recall Score")

# set x axis title to be "Image Queries"
for ax in axs.flat:
    ax.set(xlabel="Image Queries")

plt.show()
