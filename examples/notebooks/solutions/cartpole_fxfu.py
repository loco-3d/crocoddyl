cartpoleData = cartpoleND.createData()
cartpoleND.calcDiff(cartpoleData, x, u)
print(cartpoleData.Fx)
print(cartpoleData.Fu)
