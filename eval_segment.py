import matplotlib.pyplot as plt
import pickle
import numpy as np

def get_max_accuracy(acc_dict, alpha_range, sigma_range):
    max_value_list = []
    max_acc = (0,0,0,0)
    for j in sigma_range:
        #print("sigma: ", j)
        local_max = (0,0)
        local_std = 0
        for i in alpha_range:
            #print("alpha value: ", i)
            #print(acc_dict[j][i]['avg_balanced_accuracy'])
            i = round(i,1)
            
            if acc_dict[j][i]['avg_balanced_accuracy'] > local_max[1]:
                local_max = (i,acc_dict[j][i]['avg_balanced_accuracy'])
                local_std = acc_dict[j][i]['balanced_std']
                
            max_acc = (j, local_max[0], local_max[1], local_std)
        max_value_list.append(max_acc)

    return max_value_list

if __name__ == '__main__':

    unet_jaxony_dict_file = '/home/geoint/tri/github_files/unet_jaxony_dictionary_5-19.pickle'
    unet_vae_rq_dict_file = '/home/geoint/tri/github_files/unet_vae_RQ_dictionary_5-19.pickle'
    # unet_rq_dict_file = '/home/geoint/tri/github_files/unet_RQ_dictionary_5-19.pickle'
    
    # read the pickle object
    with open(unet_jaxony_dict_file, 'rb') as input_file:
        unet_jaxony_dict = pickle.load(input_file)
    with open(unet_vae_rq_dict_file, 'rb') as input_file:
        unet_vae_rq_dict = pickle.load(input_file)
    # with open(unet_rq_dict_file, 'rb') as input_file:
    #     unet_rq_dict = pickle.load(input_file)


    # get max accuracy for each sigma
    alpha_range = np.arange(0,1.1,0.1)
    sigma_range = np.arange(0.0,0.2,0.01)
    max_lst = get_max_accuracy(unet_vae_rq_dict, alpha_range, sigma_range)
    print(max_lst)

    # visualization

    sigma_range = np.arange(0,0.2,0.01)
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
        # sigma_acc_unet_rq.append(unet_rq_dict[i][0.5]['avg_balanced_accuracy'])
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
        # std_unet_rq.append(unet_rq_dict[i][0.5]['balanced_std'])

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

    name = '/home/geoint/tri/github_files/results_paper1/avg_accuracy_rqunet_vs_unet_plot.png'
    plt.title('Sigma vs Class-balanced Accuracy')
    plt.plot(sigma_range, sigma_acc_unet_jaxony, label = 'typical UNet')
    plt.fill_between(sigma_range, sigma_acc_unet_jaxony-std_unet_jaxony, sigma_acc_unet_jaxony+std_unet_jaxony, alpha=0.5)
    plt.plot(sigma_range, sigma_acc_unet_vae_rq, label = 'RQUNet-VAE alpha=0.5')
    plt.fill_between(sigma_range, sigma_acc_unet_vae_rq-std_unet_vae_rq, sigma_acc_unet_vae_rq+std_unet_vae_rq, alpha=0.5)
    #plt.plot(sigma_range, sigma_acc_unet_rq, label = 'RQUnet alpha=0.5')
    #plt.fill_between(sigma_range, sigma_acc_unet_rq-std_unet_rq, sigma_acc_unet_rq+std_unet_rq, alpha=0.5)
    plt.plot(sigma_range, sigma_acc_unet_vae_rq_1, label = 'RQUNet-VAE alpha=0.1')
    #plt.plot(sigma_range, sigma_acc_unet_vae_rq_2, label = 'Unet VAE RQ 0.2')
    #plt.plot(sigma_range, sigma_acc_unet_vae_rq_3, label = 'Unet VAE RQ 0.3')
    #plt.plot(sigma_range, sigma_acc_unet_vae_rq_4, label = 'Unet VAE RQ 0.4')
    plt.xlabel('noise level')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.show()