###########################################################################
################## TODO: Tune the weights for each cost ###################
###########################################################################
terminalCartpole = DifferentialActionModelCartpole()
terminalCartpoleDAM = DifferentialActionModelNumDiff(terminalCartpole,withGaussApprox=True)
terminalCartpoleIAM = IntegratedActionModelEuler(terminalCartpoleDAM)

terminalCartpole.costWeights[0] = 100
terminalCartpole.costWeights[1] = 100
terminalCartpole.costWeights[2] = 1.
terminalCartpole.costWeights[3] = 0.1
terminalCartpole.costWeights[4] = 0.01
terminalCartpole.costWeights[5] = 0.0001
problem = ShootingProblem(x0, [ cartpoleIAM ]*T, terminalCartpoleIAM)
