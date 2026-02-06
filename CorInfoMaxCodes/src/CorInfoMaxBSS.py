import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from IPython.display import Latex, Math, clear_output, display
from numba import njit
from tqdm import tqdm

##### IMPORT MY UTILITY SCRIPTS #######
from BSSbase import BSSBaseClass

mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15

############# Correlative Information Based Blind Source Separation Neural Network ####################################
class OnlineCorInfoMax2by2(BSSBaseClass):

    def __init__(
        self,
        lambday=0.999,
        lambdae=0.999,
        muW=1e-3,
        epsilon=1e-3,
        W=None,
        By=None,
        neural_OUTPUT_COMP_TOL=1e-6,
        update_off_diagonal_By=False,
        set_ground_truth=False,
        S=None,
        A=None,
    ):
        # HERE WE CONSIDER ONLY 2 BY 2 MIXING SYSTEM FOR DEBUGGING PURPOSES. THIS IS NOT A GENERAL IMPLEMENTATION. THE CODE ABOVE IS THE GENERAL ONE.
        s_dim = 2
        x_dim = 2
        if W is not None:
            assert W.shape == (
                s_dim,
                x_dim,
            ), "The shape of the initial guess W must be (s_dim, x_dim) = (%d,%d)" % (
                s_dim,
                x_dim,
            )
            W = W
        else:
            W = np.eye(s_dim, x_dim)

        if By is not None:
            assert By.shape == (
                s_dim,
                s_dim,
            ), "The shape of the initial guess By must be (s_dim, s_dim) = (%d,%d)" % (
                s_dim,
                s_dim,
            )
            By = By
        else:
            By = 5 * np.eye(s_dim)

        self.s_dim = s_dim
        self.x_dim = x_dim
        self.lambday = lambday
        self.lambdae = lambdae
        self.muW = muW
        self.gamy = (1 - lambday) / lambday
        self.game = (1 - lambdae) / lambdae
        self.epsilon = epsilon
        self.W = W
        self.By = By
        self.neural_OUTPUT_COMP_TOL = neural_OUTPUT_COMP_TOL
        self.update_off_diagonal_By = update_off_diagonal_By
        ### Ground Truth Sources and Mixing Matrix For Debugging
        self.set_ground_truth = set_ground_truth
        self.S = S  # Sources
        self.A = A  # Mixing Matrix
        self.SIR_list = []
        self.SNR_list = []

    ############################################################################################
    ############### REQUIRED FUNCTIONS FOR DEBUGGING ###########################################
    ############################################################################################
    def evaluate_for_debug(self, W, A, S, X, mean_normalize_estimation=False):
        s_dim = self.s_dim
        Y_ = W @ X
        if mean_normalize_estimation:
            Y_ = Y_ - Y_.mean(axis=1).reshape(-1, 1)
        Y_ = self.signed_and_permutation_corrected_sources(S, Y_)
        coef_ = ((Y_ * S).sum(axis=1) / (Y_ * Y_).sum(axis=1)).reshape(-1, 1)
        Y_ = coef_ * Y_

        SINR = 10 * np.log10(self.CalculateSINRjit(Y_, S, False)[0])
        SNR = self.snr_jit(S, Y_)

        T = W @ A
        Tabs = np.abs(T)
        P = np.zeros((s_dim, s_dim))

        for SourceIndex in range(s_dim):
            Tmax = np.max(Tabs[SourceIndex, :])
            Tabs[SourceIndex, :] = Tabs[SourceIndex, :] / Tmax
            P[SourceIndex, :] = Tabs[SourceIndex, :] > 0.999

        GG = P.T @ T
        _, SGG, _ = np.linalg.svd(
            GG
        )  # SGG is the singular values of overall matrix Wf @ A

        return SINR, SNR, SGG, Y_, P

    def plot_for_debug(self, SIR_list, SNR_list, P, debug_iteration_point, YforPlot):
        pl.clf()
        pl.subplot(2, 2, 1)
        pl.plot(np.array(SIR_list), linewidth=5)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        pl.ylabel("SIR (dB)", fontsize=45)
        pl.title("SIR Behaviour", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 2)
        pl.plot(np.array(SNR_list), linewidth=5)
        pl.grid()
        pl.title("Component SNR Check", fontsize=45)
        pl.ylabel("SNR (dB)", fontsize=45)
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 3)
        pl.plot(np.array(self.SV_list), linewidth=5)
        pl.grid()
        pl.title(
            "Singular Value Check, Overall Matrix Rank: "
            + str(np.linalg.matrix_rank(P)),
            fontsize=45,
        )
        pl.xlabel("Number of Iterations / {}".format(debug_iteration_point), fontsize=45)
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        pl.subplot(2, 2, 4)
        pl.plot(YforPlot, linewidth=5)
        pl.title("Y last 25", fontsize=45)
        pl.grid()
        pl.xticks(fontsize=45)
        pl.yticks(fontsize=45)

        clear_output(wait=True)
        display(pl.gcf())

    def compute_overall_mapping(self, return_mapping=False):
        W, By, gamy, game, epsilon = self.W, self.By, self.gamy, self.game, self.epsilon
        Wf = np.linalg.solve((game / epsilon) * np.eye(self.s_dim) - gamy * By, (game / epsilon) * W)
        if return_mapping:
            return Wf
        else:
            return None

    ###############################################################
    ############### NEURAL DYNAMICS ALGORITHMS ####################
    ###############################################################
    @staticmethod
    @njit
    def run_neural_dynamics_antisparse(
        x,
        y,
        W,
        By,
        gamy,
        game,
        epsilon,
        lr_start=0.9,
        lr_stop=1e-15,
        lr_rule="divide_by_loop_index",
        lr_decay_multiplier=0.01,
        neural_dynamic_iterations=100,
        neural_OUTPUT_COMP_TOL=1e-7,
    ):
        yke = np.dot(W, x)
        for j in range(neural_dynamic_iterations):
            # mu_y = max(lr_start / (j + 1), lr_stop)
            if lr_rule == "constant":
                mu_y = lr_start
            elif lr_rule == "divide_by_loop_index":
                mu_y = max(lr_start / (j + 1), lr_stop)
            elif lr_rule == "divide_by_slow_loop_index":
                mu_y = max(lr_start / (j * lr_decay_multiplier + 1), lr_stop)
            y_old = y.copy()
            e = yke - y
            grady = gamy * By @ y + (game / epsilon) * e
            y = y + mu_y * (grady)
            y = np.clip(y, -1, 1)

            if np.linalg.norm(y - y_old) < neural_OUTPUT_COMP_TOL * np.linalg.norm(y):
                break
        return y

    ####################################################################
    ## FIT BATCH FUNCTIONS IF ALL THE OBSERVATIONS ARE AVAILABLE      ##
    ## THESE FUNCTIONS ALSO FIT IN ONLINE MANNER.                     ##
    ####################################################################
    def fit_batch_antisparse(
        self,
        X,
        n_epochs=1,
        neural_dynamic_iterations=250,
        neural_lr_start=0.9,
        neural_lr_stop=1e-15,
        synaptic_lr_rule="constant",
        synaptic_lr_decay_divider=5000,
        neural_loop_lr_rule="divide_by_loop_index",
        neural_lr_decay_multiplier=0.005,
        use_error_corr_structured_connectivity=False,
        shuffle=False,
        debug_iteration_point=1000,
        plot_in_jupyter=False,
    ):

        lambday, lambdae, muW, gamy, game, W, By, epsilon = (
            self.lambday,
            self.lambdae,
            self.muW,
            self.gamy,
            self.game,
            self.W,
            self.By,
            self.epsilon,
        )
        neural_dynamic_tol = self.neural_OUTPUT_COMP_TOL
        debugging = self.set_ground_truth
        SIR_list = self.SIR_list
        SNR_list = self.SNR_list
        self.SV_list = []

        assert X.shape[0] == self.x_dim, "You must input the transpose"

        samples = X.shape[1]

        if debugging:
            S = self.S
            A = self.A
            if plot_in_jupyter:
                plt.figure(figsize=(45, 30), dpi=80)

        Y = np.random.randn(self.s_dim, samples)

        if shuffle:
            idx = np.random.permutation(samples)  # random permutation
        else:
            idx = np.arange(samples)

        for k in range(n_epochs):

            for i_sample in tqdm(range(samples)):
                x_current = X[:, idx[i_sample]]
                y = np.zeros(self.s_dim)

                # Output recurrent weights
                y = self.run_neural_dynamics_antisparse(
                    x_current,
                    y,
                    W,
                    By,
                    gamy,
                    game,
                    epsilon,
                    lr_start=neural_lr_start,
                    lr_stop=neural_lr_stop,
                    lr_rule=neural_loop_lr_rule,
                    lr_decay_multiplier=neural_lr_decay_multiplier,
                    neural_dynamic_iterations=neural_dynamic_iterations,
                    neural_OUTPUT_COMP_TOL=neural_dynamic_tol,
                )

                e = y - W @ x_current

                if synaptic_lr_rule == "constant":
                    muW_ = muW
                elif synaptic_lr_rule == "divide_by_log_index":
                    muW_ = np.max(
                        [
                            muW
                            / (1 + np.log(2 + (i_sample // synaptic_lr_decay_divider))),
                            1e-3,
                        ]
                    )
                elif synaptic_lr_rule == "divide_by_index":
                    muW_ = np.max([muW / (i_sample // synaptic_lr_decay_divider), 1e-3])

                W = W + muW_ * np.outer(e, x_current)

                ### Original update rule is as follows (see the commented three lines below)
                # z = By @ y
                # z_update = np.outer(z, z)
                # By = (1 / lambday) * (By - gamy * z_update)

                By_copy = By.copy()
                By[0, 0] = (1 / lambday) * (By_copy[0, 0] - gamy * (By_copy[0,0] * By_copy[0,0] * y[0] * y[0] + 
                                                            2 * By_copy[0,0] * By_copy[0, 1]* y[0] * y[1] +
                                                            By_copy[0, 1] * By_copy[0, 1] * y[1] * y[1]))

                By[1, 1] = (1 / lambday) * (By_copy[1, 1] - gamy * (By_copy[1, 0] * By_copy[1, 0] * y[0] * y[0] + 
                                                            2 * By_copy[1, 0] * By_copy[1, 1]* y[0] * y[1] +
                                                            By_copy[1, 1] * By_copy[1, 1] * y[1] * y[1]))
                if self.update_off_diagonal_By:
                    By[0, 1] = (1 / lambday) * (By_copy[0, 1] - gamy * (By_copy[0,0] * By_copy[1,0] * y[0] * y[0] + 
                                                                By_copy[0,0] * By_copy[1, 1]* y[0] * y[1] +
                                                                By_copy[0,1] * By_copy[1, 0]* y[0] * y[1] +
                                                                By_copy[0, 1] * By_copy[1, 1] * y[1] * y[1]))
                    
                    # By[1, 0] = By[0, 1]

                    By[1, 0] = (1 / lambday) * (By_copy[1, 0] - gamy * (By_copy[0,0] * By_copy[1,0] * y[0] * y[0] + 
                                                                By_copy[0,0] * By_copy[1, 1]* y[0] * y[1] +
                                                                By_copy[0,1] * By_copy[1, 0]* y[0] * y[1] +
                                                                By_copy[0, 1] * By_copy[1, 1] * y[1] * y[1]))

                # Record the seperated signal
                Y[:, idx[i_sample]] = y

                if debugging:
                    if ((i_sample % debug_iteration_point) == 0) | (
                        i_sample == samples - 1
                    ):  # & (i_sample >= debug_iteration_point):
                        self.W = W
                        self.By = By
                        try:
                            Wf = self.compute_overall_mapping(return_mapping=True)

                            SINR, SNR, SGG, Y_, P = self.evaluate_for_debug(
                                Wf, A, S, X, False
                            )
                            self.SV_list.append(abs(SGG))

                            SIR_list.append(SINR)
                            SNR_list.append(SNR)

                            self.SNR_list = SNR_list
                            self.SIR_list = SIR_list

                            if plot_in_jupyter:
                                YforPlot = Y[:, idx[i_sample - 25 : i_sample]].T
                                self.plot_for_debug(
                                    SIR_list,
                                    SNR_list,
                                    P,
                                    debug_iteration_point,
                                    YforPlot,
                                )
                        except Exception as e:
                            print(str(e))
        self.W = W
        self.By = By

