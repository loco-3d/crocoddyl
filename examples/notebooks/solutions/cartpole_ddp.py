# #########################################################################
# ################# TODO: Create the DDP solver and run it ###############
# ##########################################################################

# Creating the DDP solver
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackDDPVerbose()])

# Solving this problem
done = ddp.solve(1000)
print(done)
