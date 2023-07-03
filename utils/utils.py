import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
import matplotlib.patches as mpatches
from collections import deque

def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            #ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_img_and_mask_3(image, mask_true, mask_pred, accuracy=0):
    #classes = ['Tree', 'Grass', 'Soil', 'Concrete'] # 6-Cloud not present
    #colors = ['forestgreen', 'lawngreen', 'brown', 'orange']

    classes = ['Tree', 'Grass','Concrete'] # 6-Cloud not present
    colors = ['forestgreen', 'lawngreen', 'orange']
    colormap = pltc.ListedColormap(colors)

    # mask_true[mask_true == 0]=1
    # mask_true = mask_true-1

    #mask_pred[mask_pred==1]==0
    #mask_pred[mask_pred==2]==1
    #mask_pred[mask_pred==1]==0

     # lets plot some information here
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 60), sharex=True, sharey=True)
    axes[0].title.set_text("Image")
    axes[0].imshow(image)
    axes[1].title.set_text("Ground Truth")
    axes[1].imshow(mask_true, cmap=colormap)
    axes[2].title.set_text(str("Prediction "+ str(accuracy)))
    axes[2].imshow(mask_pred, cmap=colormap)
    fig.tight_layout()
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i])) for i in range(len(classes))]
    # put those patched as legend-handles into the legend
    plt.show()

def plot_img_and_mask_2(image, mask_pred):
    classes = ['Tree', 'Grass', 'Soil', 'Concrete'] # 6-Cloud not present
    colors = ['forestgreen', 'lawngreen', 'brown', 'orange']
    colormap = pltc.ListedColormap(colors)

    #mask_pred[mask_pred==1]==0
    #mask_pred[mask_pred==2]==1
    #mask_pred[mask_pred==1]==0

     # lets plot some information here
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    axes[0].title.set_text("Image")
    axes[0].imshow(image)
    axes[1].title.set_text("Prediction")
    axes[1].imshow(mask_pred, cmap=colormap)

    fig.tight_layout()
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i])) for i in range(len(classes))]
    # put those patched as legend-handles into the legend

    box0 = axes[0].get_position()
    box1 = axes[1].get_position()
    axes[0].set_position([box0.x0, box0.y0 + box0.height * 0.1,
                    box0.width, box0.height * 0.9])
    axes[1].set_position([box1.x0, box1.y0 + box1.height * 0.1,
                    box1.width, box1.height * 0.9])

    # Put a legend below current axis
    axes[1].legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    #plt.colorbar()
    plt.show()

def plot_img_and_mask_4(image, mask_true, mask_pred):
    #classes = ['Tree', 'Grass', 'Soil', 'Concrete'] # 6-Cloud not present
    #colors = ['forestgreen', 'lawngreen', 'brown', 'orange']

    # classes = ['1','2','3'] # 6-Cloud not present
    # colors = ['blue', 'red', 'lawngreen']
    # colormap = pltc.ListedColormap(colors)

    # mask_true[mask_true == 1] = 3
    # mask_true = mask_true-2

    #mask_pred[mask_pred==1]==0
    #mask_pred[mask_pred==2]==1
    #mask_pred[mask_pred==1]==0

     # lets plot some information here
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 60))
    axes[0].title.set_text("Image")
    axes[0].imshow(image)
    axes[1].title.set_text("Ground Truth")
    axes[1].imshow(mask_true)
    axes[2].title.set_text("Prediction")
    axes[2].imshow(mask_pred)
    fig.tight_layout()
    # create a patch (proxy artist) for every color 
    #patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i])) for i in range(len(classes))]
    # put those patched as legend-handles into the legend
    plt.show()

def plot_img_and_mask_5(image, mask_true, mask_pred):
    #classes = ['Tree', 'Grass', 'Soil', 'Concrete'] # 6-Cloud not present
    #colors = ['forestgreen', 'lawngreen', 'brown', 'orange']

    classes = ['Tree', 'Grass','Concrete'] # 6-Cloud not present
    colors = ['forestgreen', 'lawngreen', 'orange']
    colormap = pltc.ListedColormap(colors)

    #mask_true[mask_true == 1] = 3
    #mask_true = mask_true-2

     # lets plot some information here
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 60))
    axes[0].title.set_text("Image")
    axes[0].imshow(image)
    axes[1].title.set_text("Ground Truth")
    axes[1].imshow(mask_true, cmap=colormap)
    axes[2].title.set_text("Prediction")
    axes[2].imshow(mask_pred)
    fig.tight_layout()
    # create a patch (proxy artist) for every color 
    # put those patched as legend-handles into the legend
    plt.show()

def plot_img_and_mask_recon(img, mask, name):
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Sat")
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    #values = np.unique(y.ravel())
    plt.imshow(mask)
    plt.savefig(f"/home/geoint/tri/stacked-unetvae-hls-video/{name}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_3D(image, preds):

    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    zline = image[:,0]
    xline = image[:,1]
    yline = image[:,2]
    ax.plot3D(xline, yline, zline, 'gray')

    ax1 = plt.axes(projection='3d')
    #for i in range(preds.shape[2]):
    z = preds[:,:,0]
    x = preds[:,:,1]
    y = preds[:,:,2]
    ax1.scatter3D(x, y, z, 'gray')
    plt.show()

def plot_accu_map(image, mask_true, accu_map):
    classes = ['Tree', 'Grass','Concrete'] # 6-Cloud not present
    colors = ['forestgreen', 'lawngreen', 'orange']
    colormap = pltc.ListedColormap(colors)

     # lets plot some information here
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
    axes[0,0].title.set_text("Image")
    im0=axes[0,0].imshow(image)
    axes[0,1].title.set_text("Ground Truth")
    im1=axes[0,1].imshow(mask_true, cmap=colormap)
    axes[1,0].title.set_text("Image")
    im2=axes[1,0].imshow(image)
    axes[1,1].title.set_text("Accuracy Map ")
    im3=axes[1,1].imshow(accu_map, cmap="coolwarm", interpolation='nearest')
    fig.colorbar(im3)
    fig.tight_layout()
    # put those patched as legend-handles into the legend
    plt.show()

def plot_pred_only(mask_pred, image_name, accuracy=0):
    #classes = ['Tree', 'Grass', 'Soil', 'Concrete'] # 6-Cloud not present
    #colors = ['forestgreen', 'lawngreen', 'brown', 'orange']

    classes = ['Tree', 'Grass','Concrete'] # 6-Cloud not present
    colors = ['forestgreen', 'lawngreen', 'orange']
    colormap = pltc.ListedColormap(colors)

     # lets plot some information here
    #fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 60), sharex=True, sharey=True)
    # plt.title(str("Prediction "+ str(accuracy)))
    plt.imshow(mask_pred, cmap=colormap)
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i])) for i in range(len(classes))]
    # put those patched as legend-handles into the legend
    plt.axis('off')
    plt.savefig(image_name, bbox_inches='tight')
    plt.show()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)