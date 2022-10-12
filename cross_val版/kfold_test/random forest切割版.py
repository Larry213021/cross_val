import numpy as np
import os
import datetime
from sklearn import ensemble
from sklearn.model_selection import train_test_split,KFold
import statistics
from sklearn.metrics import confusion_matrix
import math

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

    time = 0
    KFold_time = 10

    df = np.load("Word2vecMIMICmindICD9_8_3times.npy")
    disease_count = 20
    vector = 8
    case_data_length = 10000
    control_data_length = 30000

    x = df[:(case_data_length + control_data_length)]
    y = np.vstack((np.repeat(np.array([[1]]), (case_data_length), axis=0),
                   np.repeat(np.array([[0]]), (control_data_length), axis=0)))
    kf = KFold(n_splits=KFold_time, shuffle=True)

    for train_index, test_index in kf.split(x):

        print("")
        print("")
        time += 1
        print("time:", time)

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        a = x_train.shape[0]
        x_train = x_train.flatten()
        x_train = x_train.reshape((a, (disease_count * vector)))

        b = x_test.shape[0]
        x_test = x_test.flatten()
        x_test = x_test.reshape((b, (disease_count * vector)))

        forest = ensemble.RandomForestClassifier(n_estimators = 200,max_depth = 100)

        forest.fit(x_train, y_train)

        y_pred = forest.predict(x_test)
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
        print("")
        print("")


    end = datetime.datetime.now()
    print("執行時間：", end - start)
    timeall = end - start
    timeall = str(timeall)

    file = open("acc.txt", "w")
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