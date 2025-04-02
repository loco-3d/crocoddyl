# flake8: noqa: F821
# #########################################################################
# ################# TODO: Create the DDP solver and run it ###############
# ##########################################################################

# Creating the DDP solver
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving this problem
done = ddp.solve()
print(done)
