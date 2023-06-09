

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
train_file_names = ["/Users/cheva/Desktop/DATA/redo_1704/OVA_all_region_1MT_train_lr0.005.csv","/Users/cheva/Desktop/DATA/redo_1704/OVA_all_region_1MT_train_lr0.0001.csv"]#'/Users/cheva/Desktop/DATA/DFC_train/OVA_DFC_results_train_1shot_1way.csv']
lr_list = [5e-3,1e-4]
#for lr in lr_list:
    #train_file_names.append('/Users/cheva/Desktop/DATA/other_regions/retrained_DFC/training/OVA_test_contrastive_maml_4shot_4cway_lr{:}.csv'.format(lr))

train_file_names=[]
output_folder = '/Users/cheva/Desktop/DATA/redo_1704'
if len(train_file_names) >= 1:
    for n in range(len(train_file_names)):
        train_file_name = train_file_names[n]
        data = genfromtxt(train_file_name, delimiter=',')
        acc = data[0,1:].tolist()
        loss = data[1,1:].tolist()
        N = [10]
        for i in range(len(N)):
            log_acc = running_mean(acc, N[i])
            plt.figure()
            plt.plot(log_acc)
            plt.title('Accuracy for running mean over {:} samples at training.\n Max accuracy: {:0.1f}, Min accuracy: {:0.1f}, lr: {:}'.format(N[i], np.max(log_acc)*100, np.min(log_acc)*100, np.mean(log_acc)*100,lr_list[n]))
            plt.hlines(np.max(log_acc),0,len(acc), colors='red')
            image_name = os.path.join(output_folder,'Accuracies for lr {:}.png'.format(lr_list[n]))
            plt.savefig(image_name)
            plt.show()


        plt.figure()
        plt.plot(loss)
        plt.title(
            'Loss at training.\n Max loss: {:0.1f}, Min loss: {:0.1f}, lr: {:}'.format(np.max(loss) , np.min(loss) , lr_list[n]))
        image_name = os.path.join(output_folder,
                                  'Loss for lr {:}.png'.format(lr_list[n]))
        plt.savefig(image_name)
        plt.show()


test_file_names = []

regions = [0,1,2,3,4,5,6]
model = 92
shots = [1,5,10]
for s in shots:
    for i in regions:
        test_file_names.append(
            #'/Users/cheva/Desktop/DATA/redo_1704/Region_{0}_results_test_{1}shot_SS0.32_model{2}.csv'.format(i,s,model))
            '/Users/cheva/Desktop/DATA/NCE/Region2_{0}_results_test_{1}shot_SS0.25_model{2}_NoNCE.csv'.format(i, s, model))
if len(test_file_names) >= 1:
    mean = 0
    std = 0
    region_acc = []
    region_std = []
    for n in range(len(test_file_names)):
        test_file_name = test_file_names[n]
        data = genfromtxt(test_file_name, delimiter=',')
        acc = data[1:].tolist()
        shot_id = shots[n//len(regions)]
        region_id = n%len(regions)
        region_acc.append(np.mean(acc)*100)
        mean += region_acc[-1]
        region_std.append(np.std(acc)*100)
        std += region_std[-1]
        #print("On region {:} and {:} shots, mean is {:.3f}, std is {:.3f}".format(region_id,shot_id, region_acc[-1], region_std[-1]))
        if region_id == 6:
            #print("The mean over all regions for {:} shots is {:.3f} and std {:.3f}".format(shot_id, mean/len(regions),std/len(regions)))
            print("Model {:} top 1".format(model))
            print("Shots: {:} mean: \n {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f}".format(shot_id, region_acc[0],region_acc[1],region_acc[2],region_acc[3],region_acc[4],region_acc[5],region_acc[6],mean/len(regions),std/len(regions)))
            mean = 0
            std = 0
            region_acc = []
            region_std = []


data_file_names_ref = ['/Users/cheva/Desktop/hist_record.csv']
data_file_names = ['/Users/cheva/Desktop/DATA/NCE/wnorm_hist.csv']
data_file_names = ['/Users/cheva/Desktop/DATA/NCE/hist_full_clean_ratio_ow.csv']
data_file_names = ['/Users/cheva/Desktop/DATA/NCE/hist_bigSF_ratio.csv']
data_file_names2 = ['/Users/cheva/Desktop/DATA/NCE/hist_bigSF_ratio2.csv']

data_file_names = []
data_file_names = ['/Users/cheva/Desktop/DATA/hist_FO.csv']

#data_file_names_ref = []
data_file_names_ref_use = False#len(data_file_names_ref) > 0
if len(data_file_names) >= 1:
    mean = 0
    std = 0
    region_acc = []
    region_std = []
    p_id = 10
    for n in range(len(data_file_names)):
        data_file_names = data_file_names[n]
        data = genfromtxt(data_file_names, delimiter=',')
        #data2 = genfromtxt(data_file_names2[0], delimiter=',')
        acc = data[:].tolist()#+data2[:].tolist()
        mean = np.mean(acc)

        #id = [i for i in range(len(data, axis=1))]
        #sort_id = [sorted, id]

        #ss = np.sort(sort_id, axis=0)
        print("Mean similarity: {:.4f} \nMax similarity:  {:.4f} \nMin similarity:  {:.4f}".format(mean, np.max(acc), np.min(acc)))
        sorted = np.sort(acc)
        print(" number of similartiy element: {:}".format(len(sorted)))
        len_10 = int(np.floor(len(sorted) / 10))
        limits = []
        for i in range(1, p_id):
            bottom = sorted[:(len_10 * i)]
            top = sorted[(len_10 * (10 - i)):]
            limits.append(sorted[(len_10 * i)])
            print("===================================================================")
            print("Max similarity of bottom {:}% {:}".format(i*10, np.max(bottom)))
            print("Min similarity of top {:}% {:}".format(i*10, np.min(top)))
            print("Nbr values where <= max bottom {:}%: {:}".format(i*10, len(np.where(sorted <= np.max(bottom))[0])))

        if data_file_names_ref_use:
            data_ref = genfromtxt(data_file_names_ref[0], delimiter=',')
            acc_ref = data_ref[:].tolist()

            sorted_ref = np.sort(acc_ref)
            for i in range(1,p_id):
                bottom = sorted_ref[:(len_10*i)]
                top = sorted_ref[(len_10*(10-i)):]
                print("===================================================================")
                print("Max similarity of bottom {:}% {:}".format(i*10, np.max(bottom)))
                print("Min similarity of top {:}% {:}".format(i*10, np.min(top)))
                print("Nbr values where <= max bottom {:}%: {:}".format(i*10, len(np.where(sorted <= np.max(bottom))[0])))
            gaus_max = np.max(sorted_ref)
            gaus = (sorted_ref)/gaus_max
            l = list(np.unique(gaus))
            gaus2 = [list(gaus).count(x) for x in l]
            max_val = np.max(gaus2)
            w = []
            for i in range(len(gaus2)):
                tmp = gaus2[i]
                tmp = max_val / (2 * tmp)
                w.append(tmp)

            w2 = np.multiply(gaus2,w)
            freq = 1.0/9
            plt.figure()
            plt.scatter(l,gaus2, s=1, color='blue')
            plt.scatter(l,w, s=1, color='red')
            plt.scatter(l,w2, s=1, color='green')
            a=[]
            for i in range(10):
                a.append(i*freq)
            #    plt.vlines(i*freq, 0,14, 'y')
            plt.show()

            # boolean array where the steps are
            # the index of the element right of the step
            steps = list(np.where(np.diff(l) > 0.03)[0] + 1)
            steps.insert(0,0)
            steps.append(len(w)-1)
            steps_w = []
            for ii in range(len(steps)-1):
                steps_w.append(np.mean(w[steps[ii]:steps[ii+1]]))

            plt.figure()
            plt.plot(w)
            plt.plot(l,'r')
            #plt.vlines(steps, 0,7, 'y')
            for i in range(len(steps_w)):
                plt.hlines(steps_w[i], steps[i],steps[i+1], colors='purple')
            plt.show()

            steps_s = list(np.where(np.diff(sorted) > 0.04)[0] + 1)
            steps_s.insert(0, 0)
            steps_s.append(len(w) - 1)
            steps_sw = []
            for ii in range(len(steps_s) - 1):
                steps_sw.append(len(list(sorted[steps_s[ii]:steps_s[ii + 1]])))

        else:
            colors = ['red','black','orange','blue','fuchsia', 'peru','springgreen','grey','purple','coral']
            plt.figure()
            #for i in range(len(limits)):
            #    plt.vlines(limits[i], 0, 20, colors=colors[i], label='Top {:} %'.format(i*10))
            plt.hist(sorted, bins=len(sorted))
            plt.vlines(np.min(sorted), 0, 40, label="Min", colors='r')
            plt.vlines(np.mean(sorted), 0, 40, label="Mean", colors='g')
            plt.vlines(np.max(sorted), 0, 40, label="Max", colors='purple')
            plt.legend()
            plt.show()




data_file_names_region = ['/Users/cheva/Desktop/DATA/NCE/hist_region_0.csv','/Users/cheva/Desktop/DATA/NCE/hist_region_1.csv',
                   '/Users/cheva/Desktop/DATA/NCE/hist_region_2.csv','/Users/cheva/Desktop/DATA/NCE/hist_region_3.csv',
                   '/Users/cheva/Desktop/DATA/NCE/hist_region_4.csv','/Users/cheva/Desktop/DATA/NCE/hist_region_5.csv',
                   '/Users/cheva/Desktop/DATA/NCE/hist_region_6.csv']
data_file_names_region = ['/Users/cheva/Desktop/DATA/NCE/hist_full_clean_ratio.csv']


data_file_names_ratio = ['/Users/cheva/Desktop/DATA/NCE/hist_ratio_0.csv', '/Users/cheva/Desktop/DATA/NCE/hist_ratio_1.csv', '/Users/cheva/Desktop/DATA/NCE/hist_ratio_2.csv']
#data_file_names_ratio = []
#total skip 1568
data_file_names_neg = ['/Users/cheva/Desktop/DATA/NCE/hist_neg_0.csv','/Users/cheva/Desktop/DATA/NCE/hist_neg_1.csv','/Users/cheva/Desktop/DATA/NCE/hist_neg_2.csv']

data_file_names_pos = ['/Users/cheva/Desktop/DATA/NCE/hist_pos_0.csv','/Users/cheva/Desktop/DATA/NCE/hist_pos_1.csv','/Users/cheva/Desktop/DATA/NCE/hist_pos_2.csv']
data_file_names_pos = ['/Users/cheva/Desktop/DATA/NCE/final_0.csv','/Users/cheva/Desktop/DATA/NCE/final_1.csv']
data_file_names_neg = ['/Users/cheva/Desktop/DATA/NCE/final_2.csv','/Users/cheva/Desktop/DATA/NCE/final_3.csv']
data_file_names_ratio = ['/Users/cheva/Desktop/DATA/NCE/final_4.csv','/Users/cheva/Desktop/DATA/NCE/final_5.csv']


data_file_names_neg = []
data_file_names_region = []
data_file_names_pos = []

if len(data_file_names_ratio) >= 1:
    sorted = []
    #for n in range(len(data_file_names_region)):
        #data_file_names = data_file_names_region[n]
        #data = genfromtxt(data_file_names, delimiter=',')

    for i in data_file_names_ratio:
        data = genfromtxt(i, delimiter=',')
        acc = data[:].tolist()
        sorted += acc
    sorted = [x for x in sorted if str(x) != 'nan' and x != 0.0]
    sorted_ratio = sorted.copy()
    sorted = np.sort(sorted)
    print("Ratio of similarities")

    print("Mean similarity: {:} \nMax similarity:  {:} \nMin similarity:  {:}".format(np.mean(sorted), np.max(sorted),
                                                                                               np.min(sorted)))
    standard = np.std(sorted)
    print("Mean and one std: {:}, {:}".format(np.mean(sorted) + standard,np.mean(sorted) - standard))
    print("Mean and two std: {:}, {:}".format(np.mean(sorted) + 2*standard,np.mean(sorted) - 2*standard))
    print("Mean and three std: {:}, {:}".format(np.mean(sorted) + 3*standard,np.mean(sorted) - 3*standard))

    len_10 = int(np.floor(len(sorted) / 10))
    limits = []
    for i in range(1, 1):#6):
        bottom = sorted[:(len_10 * i)]
        top = sorted[(len_10 * (10 - i)):]
        limits.append(sorted[(len_10 * i)])
        print("===================================================================")
        print("Max similarity of bottom {:}% {:}".format(i * 10, np.max(bottom)))
        print("Min similarity of top {:}% {:}".format(i * 10, np.min(top)))
        print("Nbr values where <= max bottom {:}%: {:}".format(i * 10, len(np.where(sorted <= np.max(bottom))[0])))

    plt.figure()
    print("===================================================================")

    hist = plt.hist(sorted, bins=len(sorted))#,label="Region {:}".format(n))
    plt.vlines(np.min(sorted), 0, 40, label="Min", colors='r')
    plt.vlines(np.mean(sorted), 0, 40, label="Mean",colors='g')
    plt.vlines(np.max(sorted), 0, 40, label="Max",colors='purple')
    #plt.vlines(np.mean(sorted)+standard, 0, 50, label="1 std", colors='yellow')
    #plt.vlines(np.mean(sorted)+2*standard, 0, 50, label="2 std", colors='orange')
    #plt.vlines(np.mean(sorted)+3*standard, 0, 50, label="3 std", colors='pink')
    #plt.vlines(np.mean(sorted)-standard, 0, 50, label="1 std", colors='yellow')
    #plt.vlines(np.mean(sorted)-2*standard, 0, 50, label="2 std", colors='orange')
    #plt.vlines(np.mean(sorted)-3*standard, 0, 50, label="3 std", colors='pink')
    plt.title("Similarity ratio")
    plt.xlabel("Similarity score")
    plt.ylabel("Number of iterations")
    plt.legend()
    plt.show()

if len(data_file_names_pos) >= 1:
    sorted = []
    for i in data_file_names_pos:
        data = genfromtxt(i, delimiter=',')
        acc = data[:].tolist()
        sorted += acc

    sorted = [x for x in sorted if str(x) != 'nan' and x != 0.0]
    sorted_pos = sorted.copy()
    sorted = np.sort(sorted)
    print("Intra-class similarity")

    print("Mean similarity: {:} \nMax similarity:  {:} \nMin similarity:  {:}".format(np.mean(sorted),
                                                                                               np.max(sorted),
                                                                                               np.min(sorted)))
    len_10 = int(np.floor(len(sorted) / 10))
    limits = []
    for i in range(1, 6):
        bottom = sorted[:(len_10 * i)]
        top = sorted[(len_10 * (10 - i)):]
        limits.append(sorted[(len_10 * i)])
        print("===================================================================")
        print("Max similarity of bottom {:}% {:}".format(i * 10, np.max(bottom)))
        print("Min similarity of top {:}% {:}".format(i * 10, np.min(top)))
        print("Nbr values where <= max bottom {:}%: {:}".format(i * 10, len(np.where(sorted <= np.max(bottom))[0])))

    plt.figure()
    hist = plt.hist(sorted, bins=len(sorted))#,label="Region {:}".format(n))
    plt.vlines(np.min(sorted), 0, 40, label="Min", colors='r')
    plt.vlines(np.mean(sorted), 0, 40, label="Mean", colors='g')
    plt.vlines(np.max(sorted), 0, 40, label="Max", colors='purple')
    plt.title("Intra-class similarity")
    plt.xlabel("Similarity score")
    plt.ylabel("Number of iterations")
    plt.legend()
    plt.show()

if len(data_file_names_neg) >= 1:
    print("Inter-class similarity")
    sorted = []
    for i in data_file_names_neg:
        data = genfromtxt(i, delimiter=',')
        acc = data[:].tolist()
        sorted += acc

    sorted = [x for x in sorted if str(x) != 'nan' and x != 0.0]

    print("Mean similarity: {:} \nMax similarity:  {:} \nMin similarity:  {:}".format(np.mean(sorted), np.max(sorted),
                                                                                               np.min(sorted)))
    len_10 = int(np.floor(len(sorted) / 10))
    limits = []
    for i in range(1, 6):
        bottom = sorted[:(len_10 * i)]
        top = sorted[(len_10 * (10 - i)):]
        limits.append(sorted[(len_10 * i)])
        print("===================================================================")
        print("Max similarity of bottom {:}% {:}".format(i * 10, np.max(bottom)))
        print("Min similarity of top {:}% {:}".format(i * 10, np.min(top)))
        print("Nbr values where <= max bottom {:}%: {:}".format(i * 10, len(np.where(sorted <= np.max(bottom))[0])))

    plt.figure()
    print("===================================================================")

    hist = plt.hist(sorted, bins=len(sorted))#,label="Region {:}".format(n))
    plt.vlines(np.min(sorted), 0, 40, label="Min", colors='r')
    plt.vlines(np.mean(sorted), 0, 40, label="Mean",colors='g')
    plt.vlines(np.max(sorted), 0, 40, label="Max",colors='purple')
    plt.title("Inter-class similarity")
    plt.xlabel("Similarity score")
    plt.ylabel("Number of iterations")
    plt.legend()
    plt.show()


one_shot = [0.16500000655651093, 0.33000001311302185, 0.1550000011920929, 0.2150000035762787, 0.2150000035762787, 0.22499999403953552, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026, 0.22583332657814026]

two_shot = [0.3149999976158142, 0.19499999284744263, 0.25, 0.27000001072883606, 0.1899999976158142, 0.22499999403953552, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934, 0.23250000178813934]

three_shot = [0.36000001430511475, 0.24500000476837158, 0.2549999952316284, 0.22499999403953552, 0.1550000011920929, 0.2800000011920929, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]



a = [ 0.14000000059604645, 0.20000000298023224, 0.20000000298023224, 0.3400000035762787, 0.2199999988079071]
b = [ 0.20000000298023224, 0.30000001192092896, 0.2199999988079071, 0.14000000059604645, 0.2199999988079071]
c=[ 0.2800000011920929, 0.23999999463558197, 0.18000000715255737, 0.2800000011920929, 0.23999999463558197]
print(np.mean(a), np.mean(b), np.mean(c))