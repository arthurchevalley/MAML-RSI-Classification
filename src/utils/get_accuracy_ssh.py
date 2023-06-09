
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import matplotlib
import os
import csv
from PIL import Image
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

train_file_names = [
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.01_ss0.25.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.01_ss0.285.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.01_ss0.32.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.005_ss0.25.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.005_ss0.285.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.005_ss0.32.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_train_5shot_4way_lr0.0001_ss0.4.csv',
    '/home/arthur/OVA/1_way/OVA_results_train_5shot_1way_lr0.01_ss0.285.csv'

]
test_file_names = [
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.01_ss0.25.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.01_ss0.285.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.01_ss0.32.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.005_ss0.25.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.005_ss0.285.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.005_ss0.32.csv',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/results_test_5shot_4way_lr0.0001_ss0.4.csv',
    '/home/arthur/OVA/1_way/OVA_results_test_5shot_1way_lr0.01_ss0.285.csv'
]

CM_filenames = [
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.01_ss0.25.png',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.01_ss0.285.png',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.01_ss0.32.png',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.005_ss0.25.png',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.005_ss0.285.png',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.005_ss0.32.png',
    '/home/arthur/res/5steps_16tasks_no_portalfredSouthAfrica_hp/ConfusionMatrix_5Shots_4Ways_lr0.0001_ss0.4.png',
    '/home/arthur/OVA/1_way/OVA_ConfusionMatrix_5Shots_1Ways_lr0.01_ss0.285_Class0.png',
    '/home/arthur/OVA/1_way/OVA_ConfusionMatrix_5Shots_1Ways_lr0.01_ss0.285_Class1.png',
    '/home/arthur/OVA/1_way/OVA_ConfusionMatrix_5Shots_1Ways_lr0.01_ss0.285_Class2.png',
    '/home/arthur/OVA/1_way/OVA_ConfusionMatrix_5Shots_1Ways_lr0.01_ss0.285_Class3.png'

]


train_file_names = []#'/Users/cheva/Desktop/DATA/DFC_train/OVA_DFC_results_train_1shot_1way.csv']


min_GS = 6#14 #5
max_GS = 26
min_way = 3
max_way = 6
min_region = 0
max_region = 7
test_file_names = []
lr_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
lr_list = [5e-3]
shots_list = [10]#1,5] #5
model_list = [0,1,2,5]#3]
#for lr in range(len(lr_list)):
for w in range(min_region,max_region):#way or region
    for gs in shots_list: #[1,5]:#range(min_GS,max_GS): # gs or shot
        for model in range(len(model_list)):
            #test_file_names.append('/Users/cheva/Desktop/DATA/DFC_train_multiple_ma/PRED_OVA_results_test_{:}shot_6way_GS{:}.csv'.format(w,gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/DFC_train_floating/CSV/PRED_OVA_results_test_{:}shot_10way_GS{:}.csv'.format(w,gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/DFC_train_floating/PRED_OVA_results_test_{:}shot_10way_GS{:}.csv'.format(w,gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/DFC_train_floating/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/DFC_train_multiple_ma/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/pretrained/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/all/FO/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/all/DFC/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/all/pretrain/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))

            #test_file_names.append('/Users/cheva/Desktop/DATA/all/all/FO/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))
            #test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/retrained_DFC/new_results/Region_{:}_results_test_1shot_lr{:}_GS20.csv'.format(w, lr))
            #test_file_names.append('/Users/cheva/Desktop/DATA/all/all/pretrain/Region_{:}_results_test_{:}shot_10way_GS20.csv'.format(w, gs))

            #test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/retrained_DFC/plots/test_models/retrained/Region_{0}_results_test_{1}shot_GS20_model{2}.csv'.format(w,gs,model))
            #test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/retrained_DFC/new_results/garbage/Region_{0}_results_test_{1}shot_GS20_model{2}.csv'.format(w,gs,model))
            test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/comparison/Region_{0}_results_test_{1}shot_GS20_model{2}.csv'.format(w,gs,model))

            # retrained for various lr
            # test_file_names.append('/Users/cheva/Desktop/DATA/other_regions/retrained_DFC/new_results/Region_{0}_results_test_1shot_lr{1}_GS20.csv'.format(w, lr))

CM_filenames = []
train_set_repeat = 3
if len(train_file_names) >= 1:
    for n in range(len(train_file_names)):
        train_file_name = train_file_names[n]
        data = genfromtxt(train_file_name, delimiter=',')
        if n//3 >= 1:
            data = data[0,1:].tolist()
        else:
            data = data[1:].tolist()


        N = [10]
        train_separations = []
        for k in range(1,train_set_repeat):
            train_separations.append(k*len(data)/train_set_repeat)
        for i in range(len(N)):
            log_acc = running_mean(data, N[i])
            plt.figure()
            plt.plot(log_acc)
            plt.title('Running mean over {:} samples at training.\n Max accuracy: {:0.1f}, Min accuracy: {:0.1f}, Mean accuracy: {:0.1f}'.format(N[i], np.max(log_acc)*100, np.min(log_acc)*100, np.mean(log_acc)*100))
            plt.hlines(np.max(log_acc),0,len(data), colors='red')
           # for k in range(len(train_separations)):
            #    plt.vlines(train_separations[k],np.min(log_acc),np.max(log_acc), colors='purple')
            plt.show()

num_shots = 0
output_folder = '/Users/cheva/Desktop/DATA/DFC_train_multiple_ma/results'
#output_folder = '/Users/cheva/Desktop/DATA/DFC_train_floating/results'
output_folder = '/Users/cheva/Desktop/DATA/all/results'

#
output_folder = '/Users/cheva/Desktop/DATA/other_regions/retrained_DFC/plots'

total_acc_plot = []
total_acc_plot_all_mc = []
total_acc_plot_2_mc = []
show_all = False
show_recap = True
show_lr = False
GS = 13
plot_test = True
names_acc = ['Top class']#, 'Two top class', 'All classes']
num_shots = 2
max_gradient_steps = max_GS - min_GS #12#21

if plot_test:
    n_test = [i for i in range(1,41, 2)]
    n_idx = 0
    n_balance = True
    for n in range(len(test_file_names)):
        test_file_name = test_file_names[n]
        data_test_all = genfromtxt(test_file_name, delimiter=',')
        if n//3 not in n_test:
            size = len(data_test_all.shape)
            if size>= 2:
                data_test = data_test_all[0,1:].tolist()
            else:
                data_test = data_test_all[1:].tolist()

        else:
            size = len(data_test_all.shape)
            if size >= 2:
                data_test = data_test_all[0, 1:].tolist()
            else:
                data_test = data_test_all[1:].tolist()
        #data_test_all_mc = data_test_all[1,1:].tolist()
        #data_test_2_mc = data_test_all[2,1:].tolist()

        N_test = [1]
        for i in range(len(N_test)):
            log_acc_test = running_mean(data_test, N_test[i])
            #log_acc_test_all_mc = running_mean(data_test_all_mc, N_test[i])
            #log_acc_test_2_mc = running_mean(data_test_2_mc, N_test[i])

        GS_2 = n % max_gradient_steps
        if GS_2 == 0:
            num_shots += 1
        GS = GS_2 + min_GS #14
        total_acc_plot.append(log_acc_test)
        #total_acc_plot_2_mc.append(log_acc_test_2_mc)
        #total_acc_plot_all_mc.append(log_acc_test_all_mc)

        if show_all:
            plt.figure()
            plt.plot(log_acc_test, label='Only the most present class at testing')
            plt.plot(log_acc_test_2_mc, label='Only the two most present class at testing')
            plt.plot(log_acc_test_all_mc, label='All the present class at testing')
            plt.title('Running mean over {:} samples. {:} Shot with {:} GS.\n Accuracies: Max: {:0.1f} %, Min: {:0.1f} %, Mean: {:0.1f} %'.format(N_test[i], num_shots, GS, np.max(log_acc_test)*100, np.min(log_acc_test)*100, np.mean(log_acc_test)*100))
            plt.hlines(np.mean(data_test),0,len(data_test), colors='red',linewidth=1)
            plt.hlines(np.mean(data_test_all_mc),0,len(data_test_all_mc), colors='red',linewidth=1)
            plt.hlines(np.mean(data_test_2_mc),0,len(data_test_2_mc), colors='red',linewidth=1)
            plt.legend(loc='best')

            image_name = os.path.join(output_folder,
                                      'Test curves with running mean over {:} samples for {:} shots with {:} GS.png'.format(N_test[i], num_shots,GS))
            plt.savefig(image_name)
            plt.show()


    all_acc_plot = [total_acc_plot]#, total_acc_plot_2_mc, total_acc_plot_all_mc]
    for k in range(len(all_acc_plot)):
        num_shots = 2
        labels = []
        res = [[],[],[],[],[],[],[],[],[],[]]
        res_param = [[],[],[],[],[],[],[],[],[],[]]
        max_shots = 3
        for i in range(10):
            val_compare = i*0.1
            GS = 0
            for idx, val in enumerate(all_acc_plot[k]):
                if val_compare < np.mean(val) < val_compare + 0.1:
                    res[i].append(idx)
                    GS_2 = idx % max_gradient_steps
                    if GS_2 == 0:
                        num_shots += 1
                    GS = GS_2 + min_GS  # 14
                    to_add = ['{:}Shots'.format(num_shots), 'GS{:}'.format(GS)]
                    res_param[i].append(to_add)

        s = False
        if s:
            for i in range(10):
                file_name = os.path.join(output_folder, 'Resulting accuracies for {:}.csv'.format(names_acc[k]))
                with open(file_name, 'a') as f:
                    write = csv.writer(f)
                    if len(res_param[i]) > 0:
                        print("\nHP for accuracy higher than {:.1f} % :\n".format(i*10))
                        write.writerow(["HP for accuracy higher than {:.1f} % :".format(i*10)])
                        for j in range(len(res_param[i])):
                            print("{:} Shots & {:} GS.".format(res_param[i][j][0],res_param[i][j][1]))
                            write.writerow(["{:} & {:}".format(res_param[i][j][0],res_param[i][j][1])])

    if show_recap:
        mean_1 = [0,0,0,0]
        mean_2 = [0,0,0,0]
        mean_all = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
        all_acc_plot = [total_acc_plot]#,total_acc_plot_2_mc,total_acc_plot_all_mc]
        for k in range(len(names_acc)):
            num_shots = 2
            labels = []
            labels2 = []
            fig = plt.figure(figsize=(25, 15))
            gs = 1
            if gs == 1:
                ax = plt.subplot(111)
            if gs != 1:
                ax = plt.subplot(121)
                ax2 = plt.subplot(122)
            gs_n_shots = False
            if gs_n_shots:
                for i in range(len(all_acc_plot[k])):
                    GS_2 = i % max_gradient_steps
                    if GS_2 == 0:
                        num_shots += 1
                    GS = GS_2 +min_GS #14
                    #plt.plot(total_acc_plot[i])
                    #plt.errorbar(np.min(total_acc_plot[i]), np.mean(total_acc_plot[i]), np.max(total_acc_plot[i]), linestyle='None', marker='^')
                    ax.errorbar(i,np.mean(all_acc_plot[k][i]),yerr=([[np.mean(all_acc_plot[k][i])-np.min(all_acc_plot[k][i])],[np.max(all_acc_plot[k][i])-np.mean(all_acc_plot[k][i])]]),marker='^')#, label='{:} GS & {:} shots'.format(GS, num_shots))
                    labels.append('{:} GS & {:} shots'.format(GS, num_shots))
                    plt.grid()
            else:
                #for i in range(len(all_acc_plot[k])):
                model_name= ['pretrained','DFC','FO','DFC All']
                region = -1
                shots_to = 0
                i1 = 0
                i2 = 0
                for i in range(len(all_acc_plot[k])):
                    new_region = i%(len(model_name)*len(shots_list))
                    model = i%(len(model_name))
                    if model == 0:
                        shots_to += 1
                        #shots = 5-(shots_to%2)*4
                        if shots_to == 1:
                            shots = 1
                        elif shots_to == 2:
                            shots = 5
                        elif shots_to == 3:
                            shots = 10
                        shots = shots_list[0]
                    if new_region == 0:
                        region += 1
                    #plt.plot(total_acc_plot[i])
                    #plt.errorbar(np.min(total_acc_plot[i]), np.mean(total_acc_plot[i]), np.max(total_acc_plot[i]), linestyle='None', marker='^')
                    mean = np.mean(all_acc_plot[k][i])
                    if shots == 1 or shots == 10 or shots == 5:
                        mean_1[model] += mean
                        mean_all[model][region] = mean
                        ax.errorbar(i1,mean,yerr=([[np.mean(all_acc_plot[k][i])-np.min(all_acc_plot[k][i])],[np.max(all_acc_plot[k][i])-np.mean(all_acc_plot[k][i])]]),marker='^')#, label='{:} GS & {:} shots'.format(GS, num_shots))
                        i1 += 1
                        if (i1+1)%(len(model_name)) == 0 and i1 != 0:
                            ax.axvline(i1,color='gray')
                        labels.append('Model {:}, Region {:}'.format(model_name[model],region))
                        plt.grid()
                    else:
                        mean_2[model] += mean
                        mean_all[model][region] = mean

                        ax2.errorbar(i2,mean,yerr=([[np.mean(all_acc_plot[k][i])-np.min(all_acc_plot[k][i])],[np.max(all_acc_plot[k][i])-np.mean(all_acc_plot[k][i])]]),marker='^')#, label='{:} GS & {:} shots'.format(GS, num_shots))
                        i2 += 1
                        if (i2+1)%3 == 0 and i2 != 0:
                            ax2.axvline(i2, color='gray')
                        labels2.append('Model {:}, Region {:}'.format(model_name[model],region))
                        plt.grid()

            ax.set_xticks([i for i in range(int(np.floor(len(test_file_names)/1)))]) #90
            ax.set_xticklabels(labels, rotation=45, ha='right')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.tick_params(labelbottom=True, labelleft=True, labelright=True)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.025))
            ax.title.set_text('{:} Shots, Pretrain mean {:.2f}, DFC mean {:.2f}, FO mean {:.2f}'.format(shots_list[0],mean_1[0]*100/6,mean_1[1]*100/6,mean_1[2]*100/6))
            ax.grid(visible=True, axis='y')
            print(mean_all)
            if shots != 1 and shots != 10 and shots != 5:
                ax2.set_xticks([i for i in range(int(np.floor(len(test_file_names)/2)))])# 90
                ax2.set_xticklabels(labels2, rotation=45, ha='right')
                box2 = ax2.get_position()
                ax2.set_position([box2.x0, box2.y0, box2.width * 0.9, box2.height])
                ax2.tick_params(labelbottom=True, labelleft=True, labelright=True)
                ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.025))
                ax2.title.set_text('5 Shots, Pretrain mean {:.2f}, DFC mean {:.2f}, FO mean {:.2f}'.format(mean_2[0]*100/6,mean_2[1]*100/6,mean_2[2]*100/6))
                ax2.grid(visible=True, axis='y')
            #ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

            #plt.grid(visible=True, axis='both')
            #plt.title('All test curves for accuracy of {:}'.format(names_acc[k]))
            image_name = os.path.join(output_folder,
                                      'All test curves for accuracy of {:} shots {:}.png'.format(names_acc[k], shots))
            plt.savefig(image_name)
            plt.show()

    if show_lr:

        all_acc_plot = [total_acc_plot,total_acc_plot_2_mc,total_acc_plot_all_mc]
        for k in range(len(names_acc)):
            num_shots = 2
            labels = []
            labels2 = []
            fig = plt.figure(figsize=(25, 15))
            ax = plt.subplot(111)
            model_name= ['DFC']
            region = -1
            shots_to = 0
            i1 = 0
            i2 = 0
            shots = 1
            new_lr = -1
            lr_mean = 0
            for i in range(len(all_acc_plot[k])):
                new_region = i%7
                if new_region == 0:
                    lr_mean = 0
                    new_lr += 1
                lr_mean += np.mean(all_acc_plot[k][i])
                ax.errorbar(i,np.mean(all_acc_plot[k][i]),yerr=([[np.mean(all_acc_plot[k][i])-np.min(all_acc_plot[k][i])],[np.max(all_acc_plot[k][i])-np.mean(all_acc_plot[k][i])]]),marker='^')#, label='{:} GS & {:} shots'.format(GS, num_shots))
                if (i+1)%7 == 0 and i != 0:
                    ax.axvline(i,color='gray')
                    lr_mean = lr_mean/7
                    ax.hlines(y=lr_mean, xmin=(i-6), xmax=i, linewidth=2, color='r')

                labels.append('Region {:}, lr {:}'.format(new_region, lr_list[new_lr]))
                plt.grid()
            ax.set_xticks([i for i in range(int(np.floor(len(test_file_names))))]) #90
            ax.set_xticklabels(labels, rotation=45, ha='right')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
            ax.tick_params(labelbottom=True, labelleft=True, labelright=True)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.025))
            ax.title.set_text('1 Shots')
            ax.grid(visible=True, axis='y')

            #ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

            #plt.grid(visible=True, axis='both')
            #plt.title('All test curves for accuracy of {:}'.format(names_acc[k]))
            image_name = os.path.join(output_folder,
                                      'All test curves for accuracy of {:}.png'.format(names_acc[k]))
            plt.savefig(image_name)
            plt.show()


### TODO CM
for n in range(len(CM_filenames)):
    if len(CM_filenames) >= 1:
    # Confusion matrix
        for i in range(len(CM_filenames)):
            plt.figure()
            CM_plt = plt.imread(CM_filenames[i])
            plt.imshow(CM_plt)
            plt.show()