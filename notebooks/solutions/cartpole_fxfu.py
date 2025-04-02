# flake8: noqa: F821
cartpoleData = cartpoleND.createData()
cartpoleND.calc(cartpoleData, x, u)
cartpoleND.calcDiff(cartpoleData, x, u)
print(cartpoleData.Fx)
print(cartpoleData.Fu)
