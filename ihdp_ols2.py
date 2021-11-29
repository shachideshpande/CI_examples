"""
IHDP experiment using simple OLS approach


covariateNames <- c("bw", "b.head", "preterm", "birth.o", "nnhealth", "momage",
                        "sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll",
                        "cig", "first", "booze", "drugs", "work.dur", "prenatal",
                        "ark", "ein", "har", "mia", "pen", "tex", "was")

https://github.com/vdorie/npci/blob/master/examples/ihdp_sim/data.R
"""
import argparse
import torch
import random
import numpy as np
from numpy import load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

REPLICATES=1000


# IHDP data loader
# input of the original train split provided by the .npz files at https://www.fredjo.com
def generate_data_ihdp_npz(replicates):
    data_train = load('Dataset/ihdp_train.npz')
    data_test = load('Dataset/ihdp_test.npz')
    
    for i in range(replicates):
        
        yield (i, (data_train['x'][...,i], data_train['t'][...,i], data_train['yf'][...,i], data_train['mu0'][...,i], data_train['mu1'][...,i]), (data_test['x'][...,i], data_test['t'][...,i], data_test['yf'][...,i], data_test['mu0'][...,i], data_test['mu1'][...,i]))

    
    
# Compute true ATE values using mu1 and mu0 values for train-val and test splits
def compute_true_ate(train_val_mu0, train_val_mu1, test_mu0, test_mu1):

    # print("Calculating true ate")
    # print("within y1=", train_val_mu1.mean(), "y0=", train_val_mu0.mean())
    # print("outside y1=", test_mu1.mean(), "y0=", test_mu0.mean())

    return (train_val_mu1-train_val_mu0).mean(), (test_mu1-test_mu0).mean()

# Main function
def main(args):
    
    # Load IHDP data
    data_loader = generate_data_ihdp_npz(REPLICATES)

    # store ATE error results here 
    result_eps_ate = np.zeros((REPLICATES,2))


    for i, train_split, test_split in data_loader:
        print("Replicate "+str(i)+"===============")
        
        # extract required train-val-test splits
        (x_train, t_train, y_train, mu0_train, mu1_train) = train_split
        (x_test, t_test, y_test, mu0_test, mu1_test) = test_split


        # Fit linear regression ols1 for data where treatment=1 and ols0 for data where treatment=0
        ols1 = LinearRegression().fit(x_train[t_train==1], y_train[t_train==1])
        ols0 = LinearRegression().fit(x_train[t_train==0], y_train[t_train==0])
        

        # Evaluate.
        
        # Compute true ate values for train+val and test datasets separately
        true_ate_trainval, true_ate_test = compute_true_ate(mu0_train, mu1_train, mu0_test, mu1_test)
        
        # Print and store results for the current IHDP replicate
        print("[Within] Predicted y1 and y0=", ols1.predict(x_train).mean(), ols0.predict(x_train).mean())
        print("[Outside] Predicted y1 and y0=", ols1.predict(x_test).mean(), ols0.predict(x_test).mean())
        est_ate_orig_trainval = abs((ols1.predict(x_train)-ols0.predict(x_train)).mean()-true_ate_trainval)
        est_ate_orig_test = abs((ols1.predict(x_test)-ols0.predict(x_test)).mean()-true_ate_test)

        print("Epsilon ATE (within sample)= {:0.3g}".format(est_ate_orig_trainval.item()))
        print("Epsilon ATE (outside sample)= {:0.3g}".format(est_ate_orig_test.item()))

        result_eps_ate[i][0] = abs(est_ate_orig_trainval)
        result_eps_ate[i][1] = abs(est_ate_orig_test)
        
   
    print("===============================")
    print("Final Results")
    print("===============================")
    print("Epsilon ATE (within sample):")
    print("Mean = {:0.3g}".format(result_eps_ate.mean(axis=0)[0]))
    print("Std dev = {:0.3g}".format(result_eps_ate.std(axis=0)[0]))

    print("Epsilon ATE (outside sample):")
    print("Mean = {:0.3g}".format(result_eps_ate.mean(axis=0)[1]))   
    print("Std dev = {:0.3g}".format(result_eps_ate.std(axis=0)[1]))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("OLS2-check")

    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
