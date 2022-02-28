import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device, unet_option):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    criterion = nn.CrossEntropyLoss()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']

        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        print(mask_true.shape)
        
        mask_true = F.one_hot(mask_true, num_classes=net.num_classes).float()

        print("eval mask true shape ",mask_true.shape)

        #mask_true = mask_true.reshape(1,4,256,256)

        image = torch.reshape(image, (1,3,256,256))
        mask_true = torch.reshape(mask_true, (1,4,256,256))

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            if unet_option == 'unet' or unet_option == 'unet_1' or unet_option == 'simple_unet':
                mask_pred = mask_pred
            else:
                mask_pred = mask_pred[0]

            print("eval mask shape: ", mask_true.shape)
            #print("eval mask pred shape: ",mask_pred.shape)

            #kl_loss = torch.mean(mask_pred[3])

            # convert to one-hot format
            if net.num_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).permute(0, 3, 1, 2).float()
                #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.num_classes).float()

                print("eval mask pred shape: ",mask_pred.shape)
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
        #return kl_loss
    return dice_score / num_val_batches
    #return kl_loss