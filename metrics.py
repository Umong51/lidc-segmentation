import torch


@torch.no_grad()
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


@torch.no_grad()
def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1)
    target = target.view(-1)
    intersection = (output * target).sum()

    return (2.0 * intersection + smooth) / (
        output.sum() + target.sum() + smooth
    )
