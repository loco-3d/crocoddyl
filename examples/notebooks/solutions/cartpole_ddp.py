##########################################################################
################## TODO: Create the DDP solver and run it ###############
###########################################################################

# Creating the DDP solver
ddp = SolverDDP(problem)
# ddp.callback = [ CallbackDDPVerbose() ]

# Solving this problem
xs, us, done = ddp.solve(maxiter=1000)
print done
