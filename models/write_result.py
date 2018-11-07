import csv
download_dir = "./result/cifar_trainTest.csv"  # where you want the file to be downloaded to

def write_training_test_results(df_time, methods, df_name):    
    print ("Writing file to ", download_dir)
    csv = open(download_dir, "a")

    for method in methods:
        if(method == "OC-NN-Linear"):
            row = method + "," + str(df_time["tf_OneClass_NN-Linear-Train"] ) + "," + str(df_time["tf_OneClass_NN-Linear-Test"]) + "\n"
            csv.write(row)
        if(method=="OC-NN-Sigmoid"):
            row = method + "," + str(df_time["tf_OneClass_NN-Sigmoid-Train"]) + "," + str(df_time["tf_OneClass_NN-Sigmoid-Test"]) + "\n"
            csv.write(row)

        if (method == "CAE-OCSVM-Linear"):
            row = method + "," + str(df_time["cae_ocsvm-linear-Train"]) + "," + str(df_time["cae_ocsvm-linear-Test"]) + "\n"
            csv.write(row)

        if (method == "CAE-OCSVM-RBF"):
            row = method + "," + str(df_time["cae_ocsvm-rbf-Train"]) + "," + str(df_time["cae_ocsvm-rbf-Test"]) + "\n"
            csv.write(row)

        if (method == "AE2-SVDD-Linear"):
            row = method + "," + str(df_time["ae_svdd-linear-Train"]) + "," + str(df_time["ae_svdd-linear-Test"]) + "\n"
            csv.write(row)

        if (method == "AE2-SVDD-RBF"):
            row = method + "," + str(df_time["ae_svdd-rbf-Train"]) + "," + str(df_time["ae_svdd-rbf-Test"]) + "\n"
            csv.write(row)

        if (method == "OCSVM-Linear"):
            row = method + "," + str(df_time["sklearn-OCSVM-Linear-Train"]) + "," + str(df_time["sklearn-OCSVM-Linear-Test"]) + "\n"
            csv.write(row)

        if (method == "OCSVM-RBF"):
            row = method + "," + str(df_time["sklearn-OCSVM-RBF-Train"]) + "," + str(df_time["sklearn-OCSVM-RBF-Test"]) + "\n"
            csv.write(row)

        if (method == "RPCA_OCSVM"):
            row = method + "," + str(df_time["rpca_ocsvm-Train"]) + "," + str(df_time["rpca_ocsvm-Test"]) + "\n"
            csv.write(row)

        if (method == "Isolation_Forest"):
            row = method + "," + str(df_time["isolation-forest-Train"]) + "," + str(df_time["isolation-forest-Test"])+ "\n"
            csv.write(row)


    return
