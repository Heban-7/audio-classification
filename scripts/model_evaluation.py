from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################
################################
## Built CNN Model Evaluation ##
################################
################################

def evaluate_and_compare(model, val_loader, test_loader, criterion, class_names):
    def evaluate(loader):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro')
        rec = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "labels": all_labels,
            "preds": all_preds,
            "probs": np.array(all_probs)
        }

    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)

    print("\nEvaluation Metrics Comparison:")
    print("{:<12} {:<15} {:<15}".format("Metric", "Validation (%)", "Test (%)"))
    print("-" * 42)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print("{:<12} {:<15.2f} {:<15.2f}".format(metric.capitalize(), val_metrics[metric]*100, test_metrics[metric]*100))

    # Bar plot comparison
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    val_scores = [val_metrics[m.lower()] for m in metrics_names]
    test_scores = [test_metrics[m.lower()] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, val_scores, width, label='Validation')
    bars2 = plt.bar(x + width/2, test_scores, width, label='Test')

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height()*100:.1f}%", ha='center')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height()*100:.1f}%", ha='center')

    plt.ylabel('Score')
    plt.title('Validation vs Test Metrics')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('val_test_metrics_comparison.png')
    plt.show()

    return val_metrics, test_metrics

def plot_roc_auc(y_true, y_probs, num_classes, class_names):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve([1 if label == i else 0 for label in y_true], y_probs[:, i])
        auc = roc_auc_score([1 if label == i else 0 for label in y_true], y_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("roc_curves.png")
    plt.show()


###############################
###############################
## Resnet18 Model Evaluation ##
###############################
###############################


def evaluate_and_compare_resnet(model, val_loader, test_loader, criterion, class_names):
    def evaluate(loader):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "labels": all_labels,
            "preds": all_preds,
            "probs": np.array(all_probs)
        }

    val_metrics = evaluate(val_loader)
    test_metrics = evaluate(test_loader)

    print("\nEvaluation Metrics Comparison:")
    print("{:<12} {:<15} {:<15}".format("Metric", "Validation (%)", "Test (%)"))
    print("-" * 42)
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        print("{:<12} {:<15.2f} {:<15.2f}".format(metric.capitalize(), val_metrics[metric]*100, test_metrics[metric]*100))

    # Bar plot
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    val_scores = [val_metrics[m.lower().replace(" ", "_")] for m in metrics_names]
    test_scores = [test_metrics[m.lower().replace(" ", "_")] for m in metrics_names]

    x = np.arange(len(metrics_names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, val_scores, width, label='Validation')
    bars2 = plt.bar(x + width/2, test_scores, width, label='Test')

    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height()*100:.1f}%", ha='center')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height()*100:.1f}%", ha='center')

    plt.ylabel('Score')
    plt.title('Validation vs Test Metrics')
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('val_test_metrics_comparison_resnet.png')
    plt.show()

    return val_metrics, test_metrics

def plot_roc_auc_resnet(y_true, y_probs, num_classes, class_names):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve([1 if label == i else 0 for label in y_true], y_probs[:, i])
        auc = roc_auc_score([1 if label == i else 0 for label in y_true], y_probs[:, i])
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves by Class')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("roc_curves_resnet.png")
    plt.show()
