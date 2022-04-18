import matplotlib.pyplot as plt
import pickle
import numpy as np


if __name__ == '__main__':

    unet_jaxony_dict_file = '/home/geoint/tri/github_files/unet_jaxony_dictionary_4-12.pickle'
    unet_vae_rq_dict_file = '/home/geoint/tri/github_files/unet_vae_RQ_dictionary_4-12.pickle'
    unet_rq_dict_file = '/home/geoint/tri/github_files/unet_RQ_dictionary_4-12.pickle'
    
    # read the pickle object
    with open(unet_jaxony_dict_file, 'rb') as input_file:
        unet_jaxony_dict = pickle.load(input_file)
    with open(unet_vae_rq_dict_file, 'rb') as input_file:
        unet_vae_rq_dict = pickle.load(input_file)
    with open(unet_rq_dict_file, 'rb') as input_file:
        unet_rq_dict = pickle.load(input_file)

    sigma_range = np.arange(0,0.1,0.01)
    sigma_acc_unet_jaxony = []
    sigma_acc_unet_vae_rq = []
    sigma_acc_unet_rq = []
    sigma_acc_unet_vae_rq_1 = []
    sigma_acc_unet_vae_rq_2 = []
    sigma_acc_unet_vae_rq_3 = []
    sigma_acc_unet_vae_rq_4 = []
    for i in sigma_range:
        sigma_acc_unet_jaxony.append(unet_jaxony_dict[i][0]['avg_balanced_accuracy'])
        sigma_acc_unet_vae_rq.append(unet_vae_rq_dict[i][0.5]['avg_balanced_accuracy'])
        sigma_acc_unet_rq.append(unet_rq_dict[i][0.5]['avg_balanced_accuracy'])
        sigma_acc_unet_vae_rq_1.append(unet_vae_rq_dict[i][0.1]['avg_balanced_accuracy'])
        #sigma_acc_unet_vae_rq_2.append(unet_vae_rq_dict[i][0.2]['avg_balanced_accuracy'])
        #sigma_acc_unet_vae_rq_3.append(unet_vae_rq_dict[i][0.3]['avg_balanced_accuracy'])
        #sigma_acc_unet_vae_rq_4.append(unet_vae_rq_dict[i][0.4]['avg_balanced_accuracy'])

    std_unet_jaxony = []
    std_unet_vae_rq = []
    std_unet_rq = []
    for i in sigma_range:
        std_unet_jaxony.append(unet_jaxony_dict[i][0]['balanced_std'])
        std_unet_vae_rq.append(unet_vae_rq_dict[i][0.5]['balanced_std'])
        std_unet_rq.append(unet_rq_dict[i][0.5]['balanced_std'])

    sigma_acc_unet_jaxony = np.array(sigma_acc_unet_jaxony)
    sigma_acc_unet_vae_rq = np.array(sigma_acc_unet_vae_rq)
    sigma_acc_unet_rq = np.array(sigma_acc_unet_rq)

    sigma_acc_unet_vae_rq_1 = np.array(sigma_acc_unet_vae_rq_1)
    #sigma_acc_unet_vae_rq_2 = np.array(sigma_acc_unet_vae_rq_2)
    #sigma_acc_unet_vae_rq_3 = np.array(sigma_acc_unet_vae_rq_3)
    #sigma_acc_unet_vae_rq_4 = np.array(sigma_acc_unet_vae_rq_4)
    std_unet_jaxony = np.array(std_unet_jaxony)
    std_unet_vae_rq = np.array(std_unet_vae_rq)
    std_unet_rq = np.array(std_unet_rq)

    plt.title('Sigma vs Class-balanced Accuracy')
    plt.plot(sigma_range, sigma_acc_unet_jaxony, label = 'typical Unet')
    plt.fill_between(sigma_range, sigma_acc_unet_jaxony-std_unet_jaxony, sigma_acc_unet_jaxony+std_unet_jaxony, alpha=0.5)
    plt.plot(sigma_range, sigma_acc_unet_vae_rq, label = 'RQUnet-VAE alpha=0.5')
    plt.fill_between(sigma_range, sigma_acc_unet_vae_rq-std_unet_vae_rq, sigma_acc_unet_vae_rq+std_unet_vae_rq, alpha=0.5)
    #plt.plot(sigma_range, sigma_acc_unet_rq, label = 'RQUnet alpha=0.5')
    #plt.fill_between(sigma_range, sigma_acc_unet_rq-std_unet_rq, sigma_acc_unet_rq+std_unet_rq, alpha=0.5)
    plt.plot(sigma_range, sigma_acc_unet_vae_rq_1, label = 'RQUNet-VAE alpha=0.1')
    #plt.plot(sigma_range, sigma_acc_unet_vae_rq_2, label = 'Unet VAE RQ 0.2')
    #plt.plot(sigma_range, sigma_acc_unet_vae_rq_3, label = 'Unet VAE RQ 0.3')
    #plt.plot(sigma_range, sigma_acc_unet_vae_rq_4, label = 'Unet VAE RQ 0.4')
    plt.legend()
    plt.show()