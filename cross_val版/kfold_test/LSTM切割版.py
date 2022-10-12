from keras.models import Sequential
from keras.layers import Flatten,LSTM
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,KFold
import math
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
start = datetime.datetime.now()

if __name__ == '__main__':

    ACC_list = []
    TPR_list = []
    TNR_list = []
    PPV_list = []
    NPV_list = []
    MCC_list = []
    F1_list = []

    roundnum = 0
    KFold_time = 10

    a = np.load(
        "D:/nkust_1108_Lab\mimic_npyfile\mimic_mind\disease_count_20\8維/3times\Word2vecMIMICmindICD9_8_3times.npy")
    disease_count = 20
    vector = 8
    case_data_length = 10000
    control_data_length = 30000

    x = a[:(case_data_length + control_data_length)]
    y = np.vstack((np.repeat(np.array([[1]]), (case_data_length), axis=0),
                   np.repeat(np.array([[0]]), (control_data_length), axis=0)))
    kf = KFold(n_splits=KFold_time, shuffle=True)

    for train_index, test_index in kf.split(x):
        print("")
        print("")
        roundnum += 1
        print("roundnum:", roundnum)

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Sequential()
        model.add(LSTM(100, return_sequences=True, activation="tanh", dropout=0.2))
        model.add(LSTM(100, return_sequences=True, activation="tanh", dropout=0.0))
        model.add(LSTM(100, return_sequences=True, activation="tanh", dropout=0.0))
        model.add(LSTM(100, return_sequences=True, activation="tanh", dropout=0.0))
        model.add(LSTM(100, return_sequences=True, activation="tanh", dropout=0.0))
        model.add(Flatten())
        model.add(Dense(units=100, activation='tanh'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


        acc_list = []
        loss_list = []
        val_acc_list = []
        val_loss_list = []
        time = 0
        for _ in range(700):
            time += 1
            print("time:", time)
            his = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=2000, epochs=1,verbose=1,use_multiprocessing=True)  # 這段做驗證
            acc_list.extend(his.history['acc'])
            loss_list.extend(his.history['loss'])
            val_acc_list.extend(his.history['val_acc'])
            val_loss_list.extend(his.history['val_loss'])


        y_pred = model.predict(x_test)
        y_pred = y_pred.flatten()
        y_pred = np.where(y_pred > 0.5, 1, 0)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        print("TN:", TN)
        print("FP:", FP)
        print("FN:", FN)
        print("TP:", TP)
        print("")


        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        TPR = round(TPR, 4)
        TPR_list.append(TPR)
        print("TPR:", TPR)

        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        TNR = round(TNR, 4)
        TNR_list.append(TNR)
        print("TNR:", TNR)

        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        PPV = round(PPV, 4)
        PPV_list.append(PPV)
        print("PPV:", PPV)

        # Negative predictive value
        NPV = TN / (TN + FN)
        NPV = round(NPV, 4)
        NPV_list.append(NPV)
        print("NPV:", NPV)

        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        FPR = round(FPR, 4)
        print("FPR:", FPR)

        # False negative rate
        FNR = FN / (TP + FN)
        FNR = round(FNR, 4)
        print("FNR:", FNR)

        # False discovery rate
        FDR = FP / (TP + FP)
        FDR = round(FDR, 4)
        print("FDR:", FDR)

        # False omission rate
        FOR = FN / FP
        FOR = round(FOR, 4)
        print("FOR:", FOR)

        # F1score
        F1score = 2 * TP / (2 * TP + FP + FN)
        F1score = round(F1score, 4)
        F1_list.append(F1score)
        print("F1score:", F1score)

        # Matthews correlation coefficient
        MCC = math.sqrt(TPR * TNR * PPV * NPV) - math.sqrt(FNR * FPR * FOR * FDR)
        MCC = round(MCC, 4)
        MCC_list.append(MCC)
        print("MCC:", MCC)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        ACC = round(ACC, 4)
        ACC_list.append(ACC)
        print("ACC:", ACC)


    plt.plot(acc_list, color='blue', label='acc_list')
    plt.plot(loss_list, color='orange', label='loss_list')
    plt.savefig('acc.png')
    plt.plot(val_acc_list, color='blue', label='val_acc_list')
    plt.plot(val_loss_list, color='orange', label='val_loss_list')
    plt.savefig('val_acc.png')


    end = datetime.datetime.now()
    print("執行時間：", end - start)
    timeall = end - start
    timeall = str(timeall)

    file = open("acc.txt", "w")
    for i in range(len(acc_list)):
        temp = "acc:" + str(np.around(acc_list[i], decimals=4)) + " - " + "loss:" + str(
            np.around(loss_list[i], decimals=4)) + " - " + "val_acc:" + str(
            np.around(val_acc_list[i], decimals=4)) + " - " + "val_loss:" + str(np.around(val_loss_list[i], decimals=4))
        file.write(temp)
        file.write('\n')
    file.write('\n')
    file.write("TN:" + str(TN))
    file.write('\n')
    file.write("FP:" + str(FP))
    file.write('\n')
    file.write("FN:" + str(FN))
    file.write('\n')
    file.write("TP:" + str(TP))
    file.write('\n')
    file.write('\n')
    file.write("TPR:" + str(statistics.mean(TPR_list)))
    file.write('\n')
    file.write("TNR:" + str(statistics.mean(TNR_list)))
    file.write('\n')
    file.write("PPV:" + str(statistics.mean(PPV_list)))
    file.write('\n')
    file.write("NPV:" + str(statistics.mean(NPV_list)))
    file.write('\n')
    file.write("F1score:" + str(statistics.mean(F1_list)))
    file.write('\n')
    file.write("MCC:" + str(statistics.mean(MCC_list)))
    file.write('\n')
    file.write("ACC:" + str(statistics.mean(ACC_list)))
    file.write('\n')
    file.write('\n')
    file.write("執行時間:" + timeall)
    file.close()