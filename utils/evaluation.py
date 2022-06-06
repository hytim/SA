import torch

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold
    corr = torch.sum(SR == GT, (1,2,3))
    tensor_size = SR.size(1)*SR.size(2)*SR.size(3)
    acc = corr/tensor_size

    return acc.mean(0)

def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT > threshold

    # TP : True Positive
    # FN : False Negative
    TP = (SR == 1) & (GT == 1)
    FN = (SR == 0) & (GT == 1)
    SE = (torch.sum(TP, (1,2,3)) + 1e-6)/(torch.sum(TP+FN, (1,2,3)) + 1e-6)

    return SE.mean(0)


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold

    # TN : True Negative
    # FP : False Positive
    TN = (SR == 0) & (GT == 0)
    FP = (SR == 1) & (GT == 0)
    SP = (torch.sum(TN, (1,2,3)) + 1e-6)/(torch.sum(TN+FP, (1,2,3)) + 1e-6)

    return SP.mean(0)


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT > threshold

    # TP : True Positive
    # FP : False Positive
    TP = (SR == 1) & (GT == 1)
    FP = (SR == 1) & (GT == 0)

    PC = (torch.sum(TP, (1,2,3)) + 1e-6)/(torch.sum(TP+FP, (1,2,3)) + 1e-6)
    return PC.mean(0)

def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT > threshold
    Inter = torch.sum((SR == 1) & (GT == 1), (1,2,3))
    Union = torch.sum((SR == 1) | (GT == 1), (1,2,3))

    JS = (Inter + 1e-6)/(Union + 1e-6)

    return JS.mean(0)


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT > threshold

    Inter = torch.sum((SR == 1) & (GT == 1), (1,2,3))
    DC = (2*Inter + 1e-6)/(torch.sum(SR, (1,2,3))+torch.sum(GT, (1,2,3)) + 1e-6)

    return DC.mean(0)

def get_He_sensitivity(SR, GT, data, threshold=0.5, he_threshold=40):
    hematoma = ((GT==1) & (data>((he_threshold/255-0.5)/0.5)))
    filter_label = GT * hematoma
    filter_prediction = SR * hematoma
    return get_sensitivity(filter_prediction, filter_label)