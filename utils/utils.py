import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import numpy as np
import matplotlib.patches as mpatches

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

    classes = ['1','2','3'] # 6-Cloud not present
    colors = ['blue', 'red', 'lawngreen']
    colormap = pltc.ListedColormap(colors)

    mask_true[mask_true == 1] = 3
    mask_true = mask_true-2

    #mask_pred[mask_pred==1]==0
    #mask_pred[mask_pred==2]==1
    #mask_pred[mask_pred==1]==0

     # lets plot some information here
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(60, 60))
    axes[0].title.set_text("Image")
    axes[0].imshow(image)
    axes[1].title.set_text("Ground Truth")
    axes[1].imshow(mask_true, cmap=colormap)
    axes[2].title.set_text("Prediction")
    axes[2].imshow(mask_pred, cmap=colormap)
    fig.tight_layout()
    # create a patch (proxy artist) for every color 
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=classes[i])) for i in range(len(classes))]
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

def plot_img_and_mask_recon(img, mask):
    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.title("Sat")
    plt.imshow(img)
    plt.subplot(1,2,2)
    plt.title("Reconstruction")
    #values = np.unique(y.ravel())
    plt.imshow(mask)
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