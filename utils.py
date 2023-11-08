from sklearn import metrics
import matplotlib.pyplot as plt


def AUROC_Score(pred_in, pred_out, file):
    y_in = [1] * len(pred_in)
    y_out = [0] * len(pred_out)

    y = y_in + y_out

    pred = pred_in.tolist() + pred_out.tolist()
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    plt.plot(fpr, tpr, label=file)
    plt.savefig(file + ".png", bbox_inches="tight")
    print(metrics.roc_auc_score(y, pred))
